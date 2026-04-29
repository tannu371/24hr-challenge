"use client";

import { useState } from "react";
import { runLandscape, type LandscapeParams } from "@/lib/api";
import { useProblem } from "@/lib/problem-context";
import Heatmap from "@/components/Heatmap";
import PageHeader from "@/components/PageHeader";
import Slider from "@/components/Slider";

type LandscapeResp = Awaited<ReturnType<typeof runLandscape>>;

export default function LandscapePage() {
  const { params } = useProblem();
  const [grid, setGrid] = useState<LandscapeResp | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [resG, setResG] = useState(41);
  const [resB, setResB] = useState(41);
  const [mixer, setMixer] = useState<"x" | "xy_ring">("x");
  const [initState, setInitState] = useState<"uniform" | "dicke">("uniform");
  const [picked, setPicked] = useState<{ r: number; c: number; gamma: number; beta: number; energy: number } | null>(null);

  async function compute() {
    setBusy(true); setErr(null);
    try {
      const payload: LandscapeParams = {
        ...params,
        mixer, init_state: initState,
        n_gamma: resG, n_beta: resB,
        gamma_max: Math.PI, beta_max: Math.PI,
      };
      const out = await runLandscape(payload);
      setGrid(out);
      setPicked({
        r: out.argmin.i, c: out.argmin.j,
        gamma: out.argmin.gamma, beta: out.argmin.beta,
        energy: out.argmin.energy,
      });
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  function onCellClick(r: number, c: number) {
    if (!grid) return;
    setPicked({
      r, c,
      gamma: grid.gamma[r], beta: grid.beta[c],
      energy: grid.energy[r][c],
    });
  }

  const heatLabels = grid
    ? {
        rowLabels: grid.gamma.map((g, i) => (i % 5 === 0 ? g.toFixed(2) : "")),
        colLabels: grid.beta.map((b, i) => (i % 5 === 0 ? b.toFixed(2) : "")),
      }
    : { rowLabels: undefined, colLabels: undefined };

  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow="Phase 3.5"
        title="Landscape playground"
        subtitle="2D scan of ⟨H_C⟩(γ, β) at p=1. Click any cell to set an initial point — Phase 4 will let you launch the optimiser from there."
      />

      <div className="grid grid-cols-12 gap-5">
        <section className="card p-5 col-span-12 lg:col-span-4 flex flex-col gap-3">
          <Slider label="γ resolution" value={resG} onChange={setResG} min={11} max={101} step={2} />
          <Slider label="β resolution" value={resB} onChange={setResB} min={11} max={101} step={2} />
          <div className="flex flex-col gap-1">
            <span className="label-cap">mixer</span>
            <select value={mixer} onChange={(e) => setMixer(e.target.value as "x" | "xy_ring")}>
              <option value="x">X mixer</option>
              <option value="xy_ring">XY-ring</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <span className="label-cap">initial state</span>
            <select value={initState} onChange={(e) => setInitState(e.target.value as "uniform" | "dicke")}>
              <option value="uniform">|+⟩^N</option>
              <option value="dicke">Dicke-K</option>
            </select>
          </div>

          <button className="btn primary self-start mt-2" disabled={busy} onClick={compute}>
            {busy ? "scanning…" : "Compute landscape"}
          </button>
          {err && <div className="text-[12px]" style={{ color: "var(--danger)" }}>{err}</div>}

          {picked && (
            <div className="mt-4 card p-3" style={{ background: "var(--background-alt)" }}>
              <div className="label-cap mb-1">selected point</div>
              <div className="mono text-[12px]">γ = {picked.gamma.toFixed(3)}</div>
              <div className="mono text-[12px]">β = {picked.beta.toFixed(3)}</div>
              <div className="mono text-[12px]">⟨H_C⟩ = {picked.energy.toFixed(4)}</div>
              <button
                className="btn cyan mt-2"
                onClick={() => {
                  // Hand off to /qaoa via localStorage so the QAOA page can pick it up.
                  try {
                    window.localStorage.setItem(
                      "qportf.qaoaWarmStart",
                      JSON.stringify({ p: 1, theta0: [picked.gamma, picked.beta] })
                    );
                  } catch { /* ignore */ }
                  window.location.href = "/qaoa";
                }}
              >
                Use as warm start →
              </button>
            </div>
          )}
        </section>

        <section className="card p-5 col-span-12 lg:col-span-8">
          {!grid ? (
            <div className="label-cap">No landscape yet — set parameters and click Compute.</div>
          ) : (
            <>
              <div className="flex justify-between mb-3">
                <div className="label-cap">⟨H_C⟩(γ, β) — N={grid.N}, K={grid.K}, mixer={grid.mixer}</div>
                <div className="mono text-[11px]" style={{ color: "var(--muted)" }}>
                  argmin γ={grid.argmin.gamma.toFixed(3)} β={grid.argmin.beta.toFixed(3)}
                </div>
              </div>
              <Heatmap
                data={grid.energy}
                rowLabels={heatLabels.rowLabels}
                colLabels={heatLabels.colLabels}
                width={620}
                height={520}
                diverging
                onCellClick={onCellClick}
                highlight={picked ? { r: picked.r, c: picked.c } : undefined}
                title="rows = γ (top→bottom), cols = β (left→right)"
                formatValue={(v) => v.toFixed(4)}
              />
            </>
          )}
        </section>
      </div>
    </div>
  );
}
