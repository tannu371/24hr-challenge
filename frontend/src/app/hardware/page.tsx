"use client";

import { useEffect, useState } from "react";
import {
  fetchBackends, fetchCachedHw, fetchTrials, importCachedHw, pollHwJob, submitHwJob,
} from "@/lib/api";
import DownloadTray from "@/components/DownloadTray";
import PageHeader from "@/components/PageHeader";
import SelectedBadge from "@/components/SelectedBadge";

type Backend = {
  name: string; qubits: number; queue_length: number;
  status: string; operational: boolean; simulator: boolean;
};

type Cached = {
  name: string; file: string;
  meta: Record<string, unknown>;
  results_summary: Record<string, unknown>;
  params: Record<string, unknown>;
};

type QaoaTrial = {
  id: number; kind: string; created_at: string; summary: Record<string, unknown>;
};

export default function HardwarePage() {
  const [backends, setBackends] = useState<Backend[] | null>(null);
  const [backendsErr, setBackendsErr] = useState<string | null>(null);
  const [cached, setCached] = useState<Cached[]>([]);
  const [trials, setTrials] = useState<QaoaTrial[]>([]);
  const [selectedTrial, setSelectedTrial] = useState<number | null>(null);
  const [selectedBackend, setSelectedBackend] = useState<string>("");
  const [shots, setShots] = useState(4096);
  const [readout, setReadout] = useState(true);
  const [dd, setDd] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<{ job_id: string; trial_id: number } | null>(null);
  const [poll, setPoll] = useState<Record<string, unknown> | null>(null);
  const [submitErr, setSubmitErr] = useState<string | null>(null);

  useEffect(() => {
    fetchBackends().then(
      (r) => setBackends(r.backends),
      (e: unknown) => setBackendsErr(e instanceof Error ? e.message : String(e)),
    );
    fetchCachedHw().then((r) => setCached(r.cached));
    fetchTrials().then((rs) => setTrials(rs.filter((t) => t.kind === "qaoa_sim")));
  }, []);

  // Poll every 5 s while the job is pending.
  useEffect(() => {
    if (!job) return;
    let stopped = false;
    const tick = async () => {
      try {
        const data = await pollHwJob(job.job_id);
        if (stopped) return;
        setPoll(data);
        const status = String(data["status"] ?? "").toUpperCase();
        if (status === "DONE" || status === "COMPLETED" || status === "ERROR" || status === "CANCELLED") return;
        setTimeout(tick, 5000);
      } catch (e: unknown) {
        if (!stopped) setSubmitErr(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    return () => { stopped = true; };
  }, [job]);

  async function submit() {
    if (!selectedTrial || !selectedBackend) return;
    setSubmitting(true); setSubmitErr(null); setJob(null); setPoll(null);
    try {
      const out = await submitHwJob({
        trial_id: selectedTrial,
        backend_name: selectedBackend,
        shots,
        error_mitigation: { readout, dynamical_decoupling: dd },
      });
      setJob({ job_id: out.job_id, trial_id: out.trial_id });
    } catch (e: unknown) {
      setSubmitErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  const tokenMissing = backendsErr?.includes("503");
  const pollResults = (poll?.results as Record<string, unknown> | undefined);
  const pollStatus = String(poll?.status ?? "").toUpperCase();

  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow="Phase 4"
        title="IBM Quantum hardware"
        subtitle="Submit a transpiled QAOA circuit to a real backend, poll for completion, ingest counts. The token never leaves the backend; the cache panel below works without one."
      />

      <div className="grid grid-cols-12 gap-5">
        <section className="card p-5 col-span-12 lg:col-span-7 flex flex-col gap-4">
          <h2 className="display text-lg">Live submission</h2>

          {tokenMissing ? (
            <div className="text-[12px] card p-3" style={{ background: "var(--background-alt)", color: "var(--muted)" }}>
              IBM token not configured. Add <code className="mono">IBM_QUANTUM_TOKEN</code> to <code className="mono">backend/.env</code> and restart the backend to enable live submission. Cached results below still work.
            </div>
          ) : backends ? (
            <div className="flex flex-col gap-1">
              <span className="label-cap">backend</span>
              <select value={selectedBackend} onChange={(e) => setSelectedBackend(e.target.value)}>
                <option value="">— pick a backend —</option>
                {backends.map((b) => (
                  <option key={b.name} value={b.name}>
                    {b.name} · {b.qubits}q · queue {b.queue_length} · {b.status}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="label-cap">loading backends…</div>
          )}
          {backendsErr && !tokenMissing && (
            <div className="text-[12px]" style={{ color: "var(--danger)" }}>{backendsErr}</div>
          )}

          <div className="flex flex-col gap-1">
            <span className="label-cap">source qaoa_sim trial (re-uses θ⋆ + circuit)</span>
            <select value={selectedTrial ?? ""} onChange={(e) => setSelectedTrial(Number(e.target.value) || null)}>
              <option value="">— pick a trial —</option>
              {trials.map((t) => (
                <option key={t.id} value={t.id}>
                  #{t.id} · {String(t.summary["energy_star"] ?? t.summary["cost"] ?? "—")} · {new Date(t.created_at).toLocaleString()}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3 items-end">
            <div className="flex flex-col gap-1">
              <span className="label-cap">shots</span>
              <input type="number" min={64} max={200000} value={shots} onChange={(e) => setShots(Number(e.target.value) || 4096)} />
            </div>
            <div className="flex flex-col gap-1">
              <label className="flex items-center gap-2 text-[12px]">
                <input type="checkbox" checked={readout} onChange={(e) => setReadout(e.target.checked)} /> readout mitigation
              </label>
              <label className="flex items-center gap-2 text-[12px]">
                <input type="checkbox" checked={dd} onChange={(e) => setDd(e.target.checked)} /> dynamical decoupling
              </label>
            </div>
          </div>

          <button
            className="btn primary self-start"
            disabled={submitting || !selectedTrial || !selectedBackend || tokenMissing}
            onClick={submit}
          >{submitting ? "submitting…" : "Submit job"}</button>

          {submitErr && (
            <div className="text-[12px]" style={{ color: "var(--danger)" }}>{submitErr}</div>
          )}

          {job && (
            <div className="card p-3 mt-2" style={{ background: "var(--background-alt)" }}>
              <div className="label-cap">job</div>
              <div className="mono text-[12px]">id: {job.job_id}</div>
              <div className="mono text-[12px]">trial: #{job.trial_id}</div>
              <div className="mono text-[12px]">status: {pollStatus || "queued"}</div>
              {poll && (
                <div className="mono text-[11px]" style={{ color: "var(--muted)" }}>
                  queue position: {String(poll["queue_position"] ?? "—")}
                </div>
              )}
              {pollResults && pollStatus === "DONE" && (
                <div className="mt-2">
                  <div className="label-cap">ingested</div>
                  <div className="mono text-[12px]">cost: {Number(pollResults["cost"]).toFixed(4)}</div>
                  <div className="mono text-[12px]">approx_ratio: {Number(pollResults["approx_ratio"] ?? 0).toFixed(3)}</div>
                  <SelectedBadge selected={pollResults["selected"] as number[]} names={pollResults["selected_names"] as string[]} />
                  <div className="mt-2"><DownloadTray trialId={job.trial_id} /></div>
                </div>
              )}
            </div>
          )}
        </section>

        <section className="card p-5 col-span-12 lg:col-span-5">
          <h2 className="display text-lg">Cached results</h2>
          <p className="text-[12px] mt-1 mb-3" style={{ color: "var(--muted)" }}>
            Snapshots in <code className="mono">/artifacts</code>. Ones tagged <code className="mono">stand_in: true</code> are synthetic placeholders that mimic shot+readout noise — replace with real runs as they come in.
          </p>
          <ul className="flex flex-col gap-2">
            {cached.map((c) => (
              <li key={c.name} className="card p-3">
                <div className="flex justify-between items-baseline">
                  <div className="display">{c.name}</div>
                  {Boolean(c.meta?.["stand_in"]) && (
                    <span className="label-cap" style={{ color: "var(--danger)" }}>stand-in</span>
                  )}
                </div>
                <div className="mono text-[11px]" style={{ color: "var(--muted)" }}>
                  backend: {String(c.meta?.["backend"] ?? "—")} · shots: {String(c.meta?.["shots"] ?? "—")}
                </div>
                <div className="mono text-[11px] mt-1">
                  cost: {Number(c.results_summary["cost"]).toFixed(4)} · approx_ratio: {Number(c.results_summary["approx_ratio"] ?? 0).toFixed(3)}
                </div>
                <button
                  className="btn cyan mt-2"
                  onClick={async () => {
                    const out = await importCachedHw(c.name);
                    window.location.href = `/trials/${out.trial_id}`;
                  }}
                >
                  Import to trials →
                </button>
              </li>
            ))}
            {cached.length === 0 && (
              <li className="label-cap">no cached snapshots in /artifacts</li>
            )}
          </ul>
        </section>
      </div>
    </div>
  );
}
