"use client";

import { use, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { deleteTrial, fetchTrial } from "@/lib/api";
import DownloadTray from "@/components/DownloadTray";
import Gauge from "@/components/Gauge";
import PageHeader from "@/components/PageHeader";
import SelectedBadge from "@/components/SelectedBadge";
import { getPinnedTrials, togglePin, unpinTrial } from "@/lib/problem-context";

type Trial = {
  id: number; kind: string; created_at: string;
  params: Record<string, unknown>; results: Record<string, unknown>;
};

type Props = { params: Promise<{ id: string }> };

export default function TrialDetailPage({ params }: Props) {
  const { id: idStr } = use(params);
  const id = Number(idStr);
  const router = useRouter();
  const [trial, setTrial] = useState<Trial | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pinned, setPinned] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    fetchTrial(id).then(setTrial, (e) => setErr(e instanceof Error ? e.message : String(e)));
    setPinned(getPinnedTrials().includes(id));
  }, [id]);

  async function onDelete() {
    if (!window.confirm(
      `Delete trial #${id} (${trial?.kind ?? "?"})?\n\nThis cannot be undone. Generated artifacts and download links will stop working.`
    )) return;
    setDeleting(true);
    try {
      await deleteTrial(id);
      unpinTrial(id);
      router.push("/trials");
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
      setDeleting(false);
    }
  }

  if (err) return <div className="text-[12px]" style={{ color: "var(--danger)" }}>{err}</div>;
  if (!trial) return <div className="label-cap">loading…</div>;

  const isQaoa = trial.kind === "qaoa_sim" || trial.kind === "qaoa_hw";
  const cost = trial.results["cost"] as number | undefined;
  const energy = (trial.results["energy_star"] ?? trial.results["energy"]) as number | undefined;
  const ratio = trial.results["approx_ratio"] as number | null | undefined;
  const top = (trial.results["top_bitstrings"] as Array<{ bitstring: string; probability?: number; count?: number; cost: number }> | undefined) ?? [];

  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow={`Trial #${id}`}
        title={trial.kind}
        subtitle={new Date(trial.created_at).toLocaleString()}
        actions={
          <>
            <button
              className="btn"
              style={{ color: pinned ? "var(--accent)" : undefined }}
              onClick={() => { setPinned(!pinned); togglePin(id); }}
            >{pinned ? "★ pinned" : "☆ pin to judge mode"}</button>
            <button
              className="btn"
              style={{ color: "var(--danger)", borderColor: "var(--danger)" }}
              disabled={deleting}
              onClick={onDelete}
            >{deleting ? "deleting…" : "Delete trial"}</button>
          </>
        }
      />

      <div className="grid grid-cols-12 gap-5">
        <section className="card p-5 col-span-12 md:col-span-5 flex flex-col gap-3">
          <h2 className="display text-lg">params</h2>
          <pre className="mono text-[11px] whitespace-pre-wrap" style={{ color: "var(--muted)" }}>
            {JSON.stringify(trial.params, null, 2)}
          </pre>
        </section>

        <section className="card p-5 col-span-12 md:col-span-7 flex flex-col gap-3">
          <h2 className="display text-lg">result</h2>
          <div className="flex flex-wrap gap-4">
            {cost !== undefined && (
              <div>
                <div className="label-cap">cost</div>
                <div className="mono text-lg" style={{ color: "var(--accent)" }}>{cost.toFixed(4)}</div>
              </div>
            )}
            {energy !== undefined && (
              <div>
                <div className="label-cap">energy ⟨H_C⟩</div>
                <div className="mono text-lg" style={{ color: "var(--cyan)" }}>{energy.toFixed(4)}</div>
              </div>
            )}
            <div>
              <div className="label-cap">K</div>
              <div className="mono text-lg">{String(trial.results["K"] ?? "—")}</div>
            </div>
          </div>
          <SelectedBadge
            selected={trial.results["selected"] as number[]}
            names={trial.results["selected_names"] as string[]}
          />
          {typeof ratio === "number" && <Gauge value={ratio} />}
          <div className="mt-3">
            <DownloadTray trialId={id} kind={trial.kind} />
          </div>
        </section>

        {isQaoa && top.length > 0 && (
          <section className="card p-5 col-span-12">
            <div className="label-cap mb-2">top bitstrings</div>
            <table className="tx">
              <thead>
                <tr>
                  <th>bitstring</th>
                  <th>probability</th>
                  <th>cost</th>
                  <th>K</th>
                </tr>
              </thead>
              <tbody>
                {top.slice(0, 10).map((b, i) => (
                  <tr key={i} style={{ cursor: "default" }}>
                    <td className="mono">{b.bitstring}</td>
                    <td className="mono">{(b.probability ?? (b.count ? Number(b.count) : 0)).toFixed?.(4) ?? "—"}</td>
                    <td className="mono">{b.cost?.toFixed(4)}</td>
                    <td className="mono">{(b as { K?: number }).K ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>
    </div>
  );
}
