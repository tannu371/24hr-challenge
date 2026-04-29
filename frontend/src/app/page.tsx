"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fetchProblem, type ProblemResponse } from "@/lib/api";
import { useProblem } from "@/lib/problem-context";
import Heatmap from "@/components/Heatmap";
import PageHeader from "@/components/PageHeader";
import Slider from "@/components/Slider";

const log10 = (v: number) => Math.log10(v);
const pow10 = (v: number) => Math.pow(10, v);

export default function ProblemPage() {
  const { params, setParams, reset } = useProblem();
  const [data, setData] = useState<ProblemResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [csvText, setCsvText] = useState<string>("");
  const [mode, setMode] = useState<"synthetic" | "csv">("synthetic");

  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        setLoading(true);
        setError(null);
        const payload = {
          ...params,
          csv_data: mode === "csv" ? csvText : null,
        };
        const resp = await fetchProblem(payload);
        setData(resp);
        if (resp.N !== params.N) setParams({ N: resp.N });
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, csvText, mode]);

  const muRows = useMemo(
    () => data?.mu.map((v, i) => ({ name: data.asset_names[i], mu: v })) ?? [],
    [data]
  );

  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow="Phase 1"
        title="Problem"
        subtitle="Define the asset universe, multi-objective scalarisation, and target cardinality. Everything downstream — classical, quantum, hardware — uses these settings."
        actions={
          <button className="btn ghost" onClick={() => reset()}>reset</button>
        }
      />

      <div className="grid grid-cols-12 gap-5">
        <section className="card p-5 col-span-12 md:col-span-5 flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <button
              className="btn"
              style={{
                background: mode === "synthetic" ? "var(--accent)" : undefined,
                color: mode === "synthetic" ? "#fff" : undefined,
              }}
              onClick={() => setMode("synthetic")}
            >Synthetic</button>
            <button
              className="btn"
              style={{
                background: mode === "csv" ? "var(--accent)" : undefined,
                color: mode === "csv" ? "#fff" : undefined,
              }}
              onClick={() => setMode("csv")}
            >Upload CSV</button>
          </div>

          {mode === "synthetic" ? (
            <Slider
              label="N — assets"
              value={params.N}
              onChange={(v) => setParams({ N: v, K: Math.min(params.K, v) })}
              min={4}
              max={20}
            />
          ) : (
            <div className="flex flex-col gap-1">
              <span className="label-cap">CSV (header + daily prices)</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  f.text().then(setCsvText);
                }}
                className="text-[12px]"
              />
              <textarea
                rows={6}
                className="mono text-[11px]"
                placeholder={"P0,P1,P2,P3\n100,101,99,100\n…"}
                value={csvText}
                onChange={(e) => setCsvText(e.target.value)}
              />
            </div>
          )}

          <Slider
            label="λ — risk aversion"
            value={params.lambda}
            onChange={(v) => setParams({ lambda: v })}
            min={0}
            max={20}
            step={0.05}
            display={(v) => v.toFixed(2)}
          />
          <Slider
            label="K — target cardinality"
            value={params.K}
            onChange={(v) => setParams({ K: v })}
            min={1}
            max={Math.max(2, data?.N ?? params.N)}
          />
          <Slider
            label="log₁₀(P_K) — cardinality penalty"
            value={log10(Math.max(1e-3, params.P_K))}
            onChange={(v) => setParams({ P_K: pow10(v) })}
            min={-1}
            max={4}
            step={0.05}
            display={(v) => `10^${v.toFixed(2)} = ${pow10(v).toFixed(2)}`}
          />
          <Slider
            label="log₁₀(P_R) — risk-threshold penalty"
            value={log10(Math.max(1e-3, params.P_R))}
            onChange={(v) => setParams({ P_R: pow10(v) })}
            min={-1}
            max={4}
            step={0.05}
            display={(v) => `10^${v.toFixed(2)} = ${pow10(v).toFixed(2)}`}
          />
          <Slider
            label="θ — risk-cap (annual variance)"
            value={params.theta_risk}
            onChange={(v) => setParams({ theta_risk: v })}
            min={0.0}
            max={0.20}
            step={0.005}
            display={(v) => `${v.toFixed(3)}  (≈ ${(Math.sqrt(v) * 100).toFixed(1)}% vol)`}
          />
          <Slider
            label="seed"
            value={params.seed}
            onChange={(v) => setParams({ seed: v })}
            min={0}
            max={999}
          />

          {loading && <div className="label-cap">recomputing…</div>}
          {error && (
            <div className="text-[12px]" style={{ color: "var(--danger)" }}>
              {error}
            </div>
          )}
        </section>

        <section className="col-span-12 md:col-span-7 flex flex-col gap-5">
          <div className="card p-5">
            <div className="flex justify-between mb-3">
              <div className="label-cap">μ — annualised expected return</div>
              <div className="mono text-[11px]" style={{ color: "var(--muted)" }}>
                {data ? `${data.mode}, N=${data.N}` : ""}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={muRows}>
                <XAxis dataKey="name" stroke="var(--muted)" fontSize={10} />
                <YAxis stroke="var(--muted)" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)", border: "1px solid var(--border)",
                    borderRadius: 6, fontSize: 12,
                  }}
                  formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
                />
                <Bar dataKey="mu" fill="var(--gold)" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card p-5">
            <div className="flex justify-between mb-2">
              <div className="label-cap">Σ — covariance heatmap</div>
            </div>
            {data && (
              <Heatmap
                data={data.Sigma}
                rowLabels={data.asset_names}
                colLabels={data.asset_names}
                width={520}
                height={Math.max(280, 24 * data.N + 60)}
                diverging
              />
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
