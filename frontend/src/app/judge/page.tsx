"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE, fetchTrial } from "@/lib/api";
import Gauge from "@/components/Gauge";
import SelectedBadge from "@/components/SelectedBadge";
import { getPinnedTrials } from "@/lib/problem-context";

type Trial = {
  id: number; kind: string; created_at: string;
  params: Record<string, unknown>; results: Record<string, unknown>;
};

export default function JudgePage() {
  const [pins, setPins] = useState<number[]>([]);
  const [trials, setTrials] = useState<Trial[]>([]);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const ids = getPinnedTrials();
    setPins(ids);
    Promise.all(ids.map(fetchTrial)).then(setTrials);
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === " " || e.key === "ArrowRight") {
        e.preventDefault();
        setIdx((i) => Math.min(i + 1, Math.max(0, trials.length - 1)));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIdx((i) => Math.max(0, i - 1));
      } else if (e.key === "Escape") {
        window.history.back();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [trials.length]);

  const total = trials.length;
  const t = trials[idx];

  return (
    <div className="fixed inset-0 flex flex-col" style={{ background: "var(--background)", zIndex: 50 }}>
      <header className="flex justify-between items-baseline px-10 py-6 border-b">
        <div className="display text-2xl">Judge Mode</div>
        <div className="flex gap-4 items-baseline">
          <span className="label-cap">space → next · ← back · esc to leave</span>
          <span className="mono text-[12px]" style={{ color: "var(--muted)" }}>
            {total > 0 ? `${idx + 1} / ${total}` : "0 trials pinned"}
          </span>
        </div>
      </header>

      <main className="flex-1 px-10 py-6 overflow-auto">
        {total === 0 ? (
          <EmptyJudge pinCount={pins.length} />
        ) : t ? (
          <Slide trial={t} stepIdx={idx + 1} total={total} />
        ) : (
          <div className="label-cap">loading slide…</div>
        )}
      </main>

      <footer className="border-t px-10 py-3 flex justify-between items-center">
        <button className="btn" disabled={idx === 0} onClick={() => setIdx((i) => Math.max(0, i - 1))}>← back</button>
        <div className="flex gap-1">
          {trials.map((_, i) => (
            <button
              key={i}
              onClick={() => setIdx(i)}
              style={{
                width: 8, height: 8, borderRadius: 4,
                background: i === idx ? "var(--accent)" : "var(--border)",
                border: 0, cursor: "pointer",
              }}
              aria-label={`go to slide ${i + 1}`}
            />
          ))}
        </div>
        <button className="btn primary" disabled={idx >= total - 1} onClick={() => setIdx((i) => Math.min(total - 1, i + 1))}>next →</button>
      </footer>
    </div>
  );
}

function EmptyJudge({ pinCount }: { pinCount: number }) {
  return (
    <div className="max-w-2xl">
      <p className="text-base" style={{ color: "var(--muted)" }}>
        No pinned trials yet. {pinCount === 0 && "Open the Trials table and click ☆ to pin trials you want in this reel."}
      </p>
    </div>
  );
}

function Slide({ trial, stepIdx, total }: { trial: Trial; stepIdx: number; total: number }) {
  const isQaoa = trial.kind === "qaoa_sim" || trial.kind === "qaoa_hw";
  const params = trial.params;
  const results = trial.results;
  const cost = results["cost"] as number | undefined;
  const energy = (results["energy_star"] ?? results["energy"]) as number | undefined;
  const ratio = (results["approx_ratio"] as number | null) ?? 0;

  const setupBullets = useMemo(() => {
    const N = params["N"]; const K = params["K"]; const lam = params["lambda"];
    const out = [
      `N = ${N} assets, K = ${K}`,
      `λ = ${typeof lam === "number" ? lam.toFixed(2) : String(lam)}, P_K = ${params["P_K"]}, P_R = ${params["P_R"]}`,
    ];
    if (isQaoa) {
      out.push(`p = ${params["p"]} layers, mixer = ${params["mixer"]}, init = ${params["init_state"]}`);
      if (trial.kind === "qaoa_hw" && params["backend"]) out.push(`backend: ${params["backend"]} · shots ${params["shots"]}`);
    }
    return out;
  }, [params, isQaoa, trial.kind]);

  return (
    <div className="grid grid-cols-12 gap-8 h-full">
      <section className="col-span-12 md:col-span-5 flex flex-col gap-4 justify-center">
        <div className="label-cap">step {stepIdx} of {total}</div>
        <h1 className="display text-5xl leading-tight">
          {trial.kind === "qaoa_sim" && "QAOA on simulator"}
          {trial.kind === "qaoa_hw" && "QAOA on IBM hardware"}
          {trial.kind === "classical_brute" && "Brute-force baseline"}
          {trial.kind === "classical_sa" && "Simulated annealing"}
          {trial.kind === "classical_markowitz" && "Markowitz relaxation"}
        </h1>
        <ul className="flex flex-col gap-2">
          {setupBullets.map((b, i) => (
            <li key={i} className="text-[14px]" style={{ color: "var(--muted)" }}>· {b}</li>
          ))}
        </ul>
        <div className="mt-3 grid grid-cols-2 gap-4">
          {cost !== undefined && (
            <div>
              <div className="label-cap">cost</div>
              <div className="mono text-3xl" style={{ color: "var(--accent)" }}>{cost.toFixed(4)}</div>
            </div>
          )}
          {energy !== undefined && (
            <div>
              <div className="label-cap">energy</div>
              <div className="mono text-3xl" style={{ color: "var(--cyan)" }}>{energy.toFixed(4)}</div>
            </div>
          )}
        </div>
        {isQaoa && typeof ratio === "number" && (
          <Gauge value={ratio} width={320} />
        )}
        <SelectedBadge selected={results["selected"] as number[]} names={results["selected_names"] as string[]} />
      </section>

      <section className="col-span-12 md:col-span-7 flex flex-col gap-3 justify-center">
        <div className="label-cap">key chart</div>
        <KeyChart trialId={trial.id} kind={trial.kind} />
      </section>
    </div>
  );
}

function KeyChart({ trialId, kind }: { trialId: number; kind: string }) {
  // Pick the most informative pre-rendered plot for each kind.
  const plot = (() => {
    if (kind === "qaoa_sim" || kind === "qaoa_hw") return "histogram";
    if (kind === "classical_sa") return "trajectory";
    if (kind === "classical_markowitz") return "comparison";
    return "comparison";
  })();
  const src = `${API_BASE}/export/plot/${trialId}/${plot}.svg`;
  return (
    /* eslint-disable-next-line @next/next/no-img-element */
    <img
      key={src}
      src={src}
      alt={`${plot} plot`}
      style={{
        width: "100%",
        maxHeight: "60vh",
        objectFit: "contain",
        border: "1px solid var(--border)",
        borderRadius: 12,
        background: "var(--card)",
      }}
    />
  );
}
