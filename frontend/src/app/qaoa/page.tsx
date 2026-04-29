"use client";

import { useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { API_BASE, runQaoa, type QaoaParams } from "@/lib/api";
import { useProblem } from "@/lib/problem-context";
import DownloadTray from "@/components/DownloadTray";
import Gauge from "@/components/Gauge";
import PageHeader from "@/components/PageHeader";
import SelectedBadge from "@/components/SelectedBadge";
import Slider from "@/components/Slider";

type Tick = {
  restart: number; iter: number; energy: number;
  gamma: number[]; beta: number[];
};

export default function QaoaPage() {
  const { params } = useProblem();
  const [p, setP] = useState(2);
  const [maxIter, setMaxIter] = useState(150);
  const [restarts, setRestarts] = useState(5);
  const [mixer, setMixer] = useState<"x" | "xy_ring">("x");
  const [initState, setInitState] = useState<"uniform" | "dicke">("uniform");
  const [optimizer, setOptimizer] = useState<"COBYLA" | "SPSA" | "L-BFGS-B">("COBYLA");

  const [busy, setBusy] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [ticks, setTicks] = useState<Tick[]>([]);
  const [out, setOut] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const cancelRef = useRef<AbortController | null>(null);

  async function runOnce() {
    setBusy(true); setErr(null); setOut(null);
    try {
      const payload: QaoaParams = {
        ...params, p, mixer, init_state: initState,
        optimizer, max_iter: maxIter, n_restarts: restarts, qaoa_seed: 0,
        compute_classical_optimum: true,
      };
      setOut(await runQaoa(payload));
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runStream() {
    setBusy(true); setStreaming(true); setErr(null); setOut(null); setTicks([]);
    const ac = new AbortController();
    cancelRef.current = ac;

    try {
      const payload: QaoaParams = {
        ...params, p, mixer, init_state: initState,
        optimizer, max_iter: maxIter, n_restarts: restarts, qaoa_seed: 0,
        compute_classical_optimum: true,
      };
      const resp = await fetch(`${API_BASE}/qaoa/run/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload),
        signal: ac.signal,
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let eventName: string | null = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trimEnd();
          buffer = buffer.slice(nl + 1);
          if (!line) { eventName = null; continue; }
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          else if (line.startsWith("data:")) {
            try {
              const data = JSON.parse(line.slice(5).trim());
              if (eventName === "tick") {
                const t = data as Tick;
                setTicks((prev) => [...prev, t]);
              } else if (eventName === "done") {
                setOut(data);
              } else if (eventName === "error") {
                setErr(String((data as { message?: string }).message ?? "stream error"));
              }
            } catch { /* ignore malformed */ }
          }
        }
      }
    } catch (e: unknown) {
      if ((e as { name?: string }).name !== "AbortError") {
        setErr(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false); setStreaming(false); cancelRef.current = null;
    }
  }

  const trialId = out?.trial_id as number | undefined;
  const top = (out?.top_bitstrings as Array<{ bitstring: string; probability: number; cost: number; K: number }> | undefined) ?? [];
  const approxRatio = out?.approx_ratio as number | null | undefined;

  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow="Phase 3"
        title="Quantum (Aer simulator)"
        subtitle="Hand-rolled QAOA on Qiskit primitives. Live energy stream over SSE; the best restart wins."
      />

      <div className="grid grid-cols-12 gap-5">
        <section className="card p-5 col-span-12 lg:col-span-4 flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-3">
            <Slider label="p — depth" value={p} onChange={setP} min={1} max={6} />
            <Slider label="max_iter" value={maxIter} onChange={setMaxIter} min={10} max={500} step={10} />
            <Slider label="restarts" value={restarts} onChange={setRestarts} min={1} max={20} />
            <div className="flex flex-col gap-1">
              <span className="label-cap">mixer</span>
              <select value={mixer} onChange={(e) => setMixer(e.target.value as "x" | "xy_ring")}>
                <option value="x">X (standard)</option>
                <option value="xy_ring">XY-ring (preserves K)</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="label-cap">initial state</span>
              <select value={initState} onChange={(e) => setInitState(e.target.value as "uniform" | "dicke")}>
                <option value="uniform">|+⟩^N</option>
                <option value="dicke">Dicke-K</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="label-cap">optimizer</span>
              <select value={optimizer} onChange={(e) => setOptimizer(e.target.value as "COBYLA" | "SPSA" | "L-BFGS-B")}>
                <option value="COBYLA">COBYLA</option>
                <option value="SPSA">SPSA</option>
                <option value="L-BFGS-B">L-BFGS-B</option>
              </select>
            </div>
          </div>

          <div className="flex gap-2 mt-2">
            <button className="btn primary" disabled={busy} onClick={runStream}>
              {streaming ? "streaming…" : busy ? "running…" : "Run with live stream"}
            </button>
            <button className="btn" disabled={busy} onClick={runOnce}>
              Run (no stream)
            </button>
            {streaming && (
              <button className="btn ghost" onClick={() => cancelRef.current?.abort()}>
                cancel
              </button>
            )}
          </div>

          {err && <div className="text-[12px]" style={{ color: "var(--danger)" }}>{err}</div>}

          {mixer === "xy_ring" && initState !== "dicke" && (
            <div className="text-[11px] mt-1" style={{ color: "var(--danger)" }}>
              Heads up: XY-ring mixer is only Hamming-weight-preserving when paired with the Dicke-K initial state.
            </div>
          )}
        </section>

        <section className="card p-5 col-span-12 lg:col-span-8 flex flex-col gap-3">
          <div className="label-cap">energy stream</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={ticks.map((t, i) => ({ i, energy: t.energy, restart: t.restart }))}>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
              <XAxis dataKey="i" stroke="var(--muted)" fontSize={10} />
              <YAxis stroke="var(--muted)" fontSize={10} />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }}
                formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
              />
              <Line type="monotone" dataKey="energy" dot={false} stroke="var(--cyan)" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex justify-between mono text-[11px]" style={{ color: "var(--muted)" }}>
            <span>{ticks.length} iterations streamed</span>
            {ticks.length > 0 && (
              <span>last energy: {ticks[ticks.length - 1].energy.toFixed(4)}</span>
            )}
          </div>
        </section>

        {out && (
          <section className="card p-5 col-span-12 grid grid-cols-12 gap-5">
            <div className="col-span-12 md:col-span-5">
              <div className="flex items-baseline gap-3 mb-3">
                <h2 className="display text-lg">result</h2>
                <span className="label-cap">trial #{trialId}</span>
              </div>
              <div className="flex gap-4 mb-3">
                <div>
                  <div className="label-cap">final energy ⟨H_C⟩</div>
                  <div className="mono text-lg" style={{ color: "var(--cyan)" }}>
                    {(out.energy_star as number).toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="label-cap">selected cost</div>
                  <div className="mono text-lg" style={{ color: "var(--accent)" }}>
                    {(out.cost as number)?.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="label-cap">classical opt</div>
                  <div className="mono text-lg">
                    {out.classical_optimum != null
                      ? (out.classical_optimum as number).toFixed(4)
                      : "—"}
                  </div>
                </div>
              </div>
              <SelectedBadge
                selected={out.selected as number[]}
                names={out.selected_names as string[]}
              />
              <div className="mt-4">
                <Gauge value={typeof approxRatio === "number" ? approxRatio : 0} />
              </div>
              {trialId && (
                <details className="mt-3">
                  <summary className="label-cap cursor-pointer">downloads</summary>
                  <div className="mt-2"><DownloadTray trialId={trialId} /></div>
                </details>
              )}
            </div>

            <div className="col-span-12 md:col-span-7">
              <div className="label-cap mb-1">top-10 bitstring distribution</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={top}>
                  <XAxis dataKey="bitstring" stroke="var(--muted)" fontSize={9} angle={-30} textAnchor="end" height={50} />
                  <YAxis stroke="var(--muted)" fontSize={10} />
                  <Tooltip
                    contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }}
                    formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
                  />
                  <Bar dataKey="probability" fill="var(--gold)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
