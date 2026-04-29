"use client";

import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { runBrute, runMarkowitz, runSA } from "@/lib/api";
import { useProblem } from "@/lib/problem-context";
import PageHeader from "@/components/PageHeader";
import SelectedBadge from "@/components/SelectedBadge";
import Slider from "@/components/Slider";
import DownloadTray from "@/components/DownloadTray";

type AnyResult = Record<string, unknown> & { trial_id?: number };

export default function ClassicalPage() {
  const { params } = useProblem();

  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow="Phase 2"
        title="Classical baselines"
        subtitle="Brute force gives the global optimum (small N), SA is the strong heuristic competitor on the QUBO, and the convex Markowitz relaxation gives an upper bound + the efficient frontier."
      />

      <div className="grid grid-cols-12 gap-5">
        <BruteCard params={params} />
        <SACard params={params} />
        <MarkowitzCard params={params} />
      </div>
    </div>
  );
}

function ResultPanel({ result, kind }: { result: AnyResult; kind: string }) {
  const cost = result.cost as number | undefined;
  const trialId = result.trial_id as number | undefined;
  return (
    <div className="mt-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-3">
        <div>
          <div className="label-cap">cost</div>
          <div className="mono text-lg" style={{ color: "var(--accent)" }}>
            {cost !== undefined ? cost.toFixed(4) : "—"}
          </div>
        </div>
        <div>
          <div className="label-cap">K</div>
          <div className="mono text-lg">{(result.K as number) ?? "—"}</div>
        </div>
        <div>
          <div className="label-cap">runtime</div>
          <div className="mono text-lg">
            {(result.runtime_s as number)?.toFixed(3) ?? "—"} s
          </div>
        </div>
        {trialId && (
          <div className="ml-auto label-cap">trial #{trialId}</div>
        )}
      </div>
      <SelectedBadge
        selected={result.selected as number[] | undefined}
        names={result.selected_names as string[] | undefined}
      />
      {trialId && (
        <details>
          <summary className="label-cap cursor-pointer">downloads</summary>
          <div className="mt-2"><DownloadTray trialId={trialId} kind={kind} /></div>
        </details>
      )}
    </div>
  );
}

/* -------- Brute -------- */
function BruteCard({ params }: { params: ReturnType<typeof useProblem>["params"] }) {
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<AnyResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  return (
    <section className="card p-5 col-span-12 lg:col-span-4 flex flex-col">
      <div className="flex justify-between items-baseline">
        <h2 className="display text-lg">Brute force</h2>
        <span className="label-cap">/classical/brute</span>
      </div>
      <p className="text-[12px] mt-1 mb-3" style={{ color: "var(--muted)" }}>
        Enumerates all C(N,K) cardinality-K bitstrings. Optimal for N ≤ 20.
      </p>
      <button
        className="btn primary self-start"
        disabled={busy}
        onClick={async () => {
          setBusy(true); setErr(null);
          try { setOut(await runBrute(params)); }
          catch (e: unknown) { setErr(e instanceof Error ? e.message : String(e)); }
          finally { setBusy(false); }
        }}
      >{busy ? "running…" : "Run brute"}</button>

      {err && <div className="text-[12px] mt-3" style={{ color: "var(--danger)" }}>{err}</div>}
      {out && <ResultPanel result={out} kind="classical_brute" />}

      {Array.isArray(out?.energy_distribution) && (
        <div className="mt-4">
          <div className="label-cap mb-1">energy distribution</div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart
              data={(out.energy_distribution as number[]).slice(0, 80).map((v, i) => ({ i, v }))}
            >
              <XAxis dataKey="i" hide />
              <YAxis stroke="var(--muted)" fontSize={9} />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }}
                formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
              />
              <Bar dataKey="v" fill="var(--cyan)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

/* -------- SA -------- */
function SACard({ params }: { params: ReturnType<typeof useProblem>["params"] }) {
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<AnyResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [T0, setT0] = useState(1.0);
  const [Tmin, setTmin] = useState(1e-3);
  const [sweeps, setSweeps] = useState(200);
  const [restarts, setRestarts] = useState(20);
  const [move, setMove] = useState<"flip" | "swap">("swap");

  return (
    <section className="card p-5 col-span-12 lg:col-span-4 flex flex-col">
      <div className="flex justify-between items-baseline">
        <h2 className="display text-lg">Simulated annealing</h2>
        <span className="label-cap">/classical/sa</span>
      </div>
      <p className="text-[12px] mt-1 mb-3" style={{ color: "var(--muted)" }}>
        Metropolis spin-flip on the QUBO with multistart. Strong classical competitor.
      </p>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <Slider label="T0" value={T0} onChange={setT0} min={0.05} max={5} step={0.05} display={(v) => v.toFixed(2)} />
        <Slider label="T_min" value={Math.log10(Tmin)} onChange={(v) => setTmin(Math.pow(10, v))} min={-5} max={0} step={0.1} display={(v) => `10^${v.toFixed(1)}`} />
        <Slider label="sweeps" value={sweeps} onChange={setSweeps} min={20} max={1000} step={10} />
        <Slider label="restarts" value={restarts} onChange={setRestarts} min={1} max={50} />
        <div className="flex flex-col gap-1 col-span-2">
          <span className="label-cap">move</span>
          <select value={move} onChange={(e) => setMove(e.target.value as "flip" | "swap")}>
            <option value="swap">swap (preserves K)</option>
            <option value="flip">flip (full hypercube)</option>
          </select>
        </div>
      </div>

      <button
        className="btn primary self-start"
        disabled={busy}
        onClick={async () => {
          setBusy(true); setErr(null);
          try {
            const res = await runSA({
              ...params, T0, T_min: Tmin, sweeps, restarts, move, init: "random_K", sa_seed: 0,
            } as Parameters<typeof runSA>[0]);
            setOut(res);
          } catch (e: unknown) { setErr(e instanceof Error ? e.message : String(e)); }
          finally { setBusy(false); }
        }}
      >{busy ? "annealing…" : "Run SA"}</button>

      {err && <div className="text-[12px] mt-3" style={{ color: "var(--danger)" }}>{err}</div>}
      {out && <ResultPanel result={out} kind="classical_sa" />}

      {Array.isArray(out?.runs) && (
        <div className="mt-4">
          <div className="label-cap mb-1">restart trajectories</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
              <XAxis stroke="var(--muted)" fontSize={9} />
              <YAxis stroke="var(--muted)" fontSize={9} />
              <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }} />
              {(out.runs as Array<{ trajectory_per_sweep: number[] }>).map((r, i) => (
                <Line
                  key={i}
                  data={r.trajectory_per_sweep.map((v, j) => ({ j, v }))}
                  dataKey="v"
                  type="monotone"
                  dot={false}
                  stroke={i === (out.best_restart as number) ? "var(--accent)" : "var(--cyan)"}
                  strokeOpacity={i === (out.best_restart as number) ? 1 : 0.35}
                  strokeWidth={i === (out.best_restart as number) ? 2 : 1}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

/* -------- Markowitz -------- */
function MarkowitzCard({ params }: { params: ReturnType<typeof useProblem>["params"] }) {
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<AnyResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [nLambda, setNLambda] = useState(50);

  return (
    <section className="card p-5 col-span-12 lg:col-span-4 flex flex-col">
      <div className="flex justify-between items-baseline">
        <h2 className="display text-lg">Markowitz (relaxed)</h2>
        <span className="label-cap">/classical/markowitz</span>
      </div>
      <p className="text-[12px] mt-1 mb-3" style={{ color: "var(--muted)" }}>
        Convex relaxation w∈ℝᴺ via cvxpy. Project to top-K. Sweeps λ for the frontier.
      </p>

      <Slider label="frontier resolution" value={nLambda} onChange={setNLambda} min={10} max={120} />

      <button
        className="btn primary self-start mt-3"
        disabled={busy}
        onClick={async () => {
          setBusy(true); setErr(null);
          try {
            const res = await runMarkowitz({
              ...params,
              frontier: true,
              frontier_n_lambda: nLambda,
              frontier_lambda_min: 0.05,
              frontier_lambda_max: 20.0,
            });
            setOut(res);
          } catch (e: unknown) { setErr(e instanceof Error ? e.message : String(e)); }
          finally { setBusy(false); }
        }}
      >{busy ? "solving…" : "Run Markowitz"}</button>

      {err && <div className="text-[12px] mt-3" style={{ color: "var(--danger)" }}>{err}</div>}
      {out && <ResultPanel result={out} kind="classical_markowitz" />}

      {Array.isArray(out?.frontier) && (
        <div className="mt-4">
          <div className="label-cap mb-1">efficient frontier (vol vs return)</div>
          <ResponsiveContainer width="100%" height={180}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
              <XAxis
                type="number"
                dataKey="vol"
                stroke="var(--muted)"
                fontSize={10}
                domain={["dataMin", "dataMax"]}
                name="vol"
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <YAxis
                type="number"
                dataKey="return"
                stroke="var(--muted)"
                fontSize={10}
                domain={["dataMin", "dataMax"]}
                name="return"
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }}
                formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
              />
              <Scatter
                data={(out.frontier as Array<{ vol: number; return: number; lambda: number }>).map((d) => ({ vol: d.vol, return: d.return, l: d.lambda }))}
                fill="var(--gold)"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
