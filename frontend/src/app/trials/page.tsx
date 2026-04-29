"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { deleteTrial, fetchTrials } from "@/lib/api";
import PageHeader from "@/components/PageHeader";
import { getPinnedTrials, togglePin, unpinTrial } from "@/lib/problem-context";

type Row = {
  id: number; kind: string; created_at: string;
  summary: Record<string, unknown>;
};

type SortKey = "id" | "kind" | "cost" | "energy" | "approx_ratio";

export default function TrialsPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [pins, setPins] = useState<number[]>([]);
  const [filter, setFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    fetchTrials().then(setRows);
    setPins(getPinnedTrials());
  }, []);

  async function onDelete(trialId: number, kind: string) {
    if (!window.confirm(`Delete trial #${trialId} (${kind})?`)) return;
    try {
      await deleteTrial(trialId);
      setRows((rs) => rs.filter((r) => r.id !== trialId));
      setPins(unpinTrial(trialId));
    } catch (e: unknown) {
      window.alert(e instanceof Error ? e.message : String(e));
    }
  }

  const filtered = useMemo(() => {
    let out = rows;
    if (filter !== "all") out = out.filter((r) => r.kind === filter);
    out = [...out].sort((a, b) => {
      const av = readSortable(a, sortKey);
      const bv = readSortable(b, sortKey);
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === "asc" ? cmp : -cmp;
    });
    return out;
  }, [rows, filter, sortKey, sortDir]);

  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow="Phase 6.7"
        title="Trials"
        subtitle="Every solve is recorded. Pin trials with ★ to add them to the Judge Mode reel."
        actions={
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">all kinds</option>
            <option value="classical_brute">brute</option>
            <option value="classical_sa">SA</option>
            <option value="classical_markowitz">Markowitz</option>
            <option value="qaoa_sim">QAOA (sim)</option>
            <option value="qaoa_hw">QAOA (hw)</option>
          </select>
        }
      />

      <div className="card p-4">
        <table className="tx">
          <thead>
            <tr>
              <th onClick={() => toggle("id")}>{header("id", sortKey, sortDir)}</th>
              <th onClick={() => toggle("kind")}>{header("kind", sortKey, sortDir)}</th>
              <th>created</th>
              <th onClick={() => toggle("cost")}>{header("cost", sortKey, sortDir)}</th>
              <th onClick={() => toggle("energy")}>{header("energy", sortKey, sortDir)}</th>
              <th onClick={() => toggle("approx_ratio")}>{header("approx_ratio", sortKey, sortDir)}</th>
              <th>K</th>
              <th>pin</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.id} onClick={() => (window.location.href = `/trials/${r.id}`)}>
                <td className="mono">#{r.id}</td>
                <td>{r.kind}</td>
                <td className="mono text-[11px]">{new Date(r.created_at).toLocaleString()}</td>
                <td className="mono">{fmt(r.summary["cost"])}</td>
                <td className="mono">{fmt(r.summary["energy_star"] ?? r.summary["energy"])}</td>
                <td className="mono">{fmt(r.summary["approx_ratio"])}</td>
                <td className="mono">{String(r.summary["K"] ?? "—")}</td>
                <td onClick={(e) => { e.stopPropagation(); setPins(togglePin(r.id)); }}>
                  <span style={{ color: pins.includes(r.id) ? "var(--accent)" : "var(--muted)", fontSize: 14 }}>
                    {pins.includes(r.id) ? "★" : "☆"}
                  </span>
                </td>
                <td
                  onClick={(e) => { e.stopPropagation(); onDelete(r.id, r.kind); }}
                  title="delete trial"
                  style={{ color: "var(--muted)", cursor: "pointer" }}
                >
                  <span style={{ fontSize: 14 }}>✕</span>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td colSpan={9} className="label-cap">no trials yet — run something on Classical or QAOA</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-4">
        <Link className="btn cyan" href="/judge">Open Judge Mode →</Link>
      </div>
    </div>
  );

  function toggle(k: SortKey) {
    if (sortKey === k) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(k); setSortDir("desc"); }
  }
}

function readSortable(r: Row, key: SortKey): number | string {
  if (key === "id") return r.id;
  if (key === "kind") return r.kind;
  if (key === "energy") return Number(r.summary["energy_star"] ?? r.summary["energy"] ?? Number.POSITIVE_INFINITY);
  if (key === "cost") return Number(r.summary["cost"] ?? Number.POSITIVE_INFINITY);
  if (key === "approx_ratio") return Number(r.summary["approx_ratio"] ?? -1);
  return 0;
}

function header(name: SortKey, sortKey: SortKey, dir: "asc" | "desc") {
  return (
    <span style={{ cursor: "pointer" }}>
      {name}{sortKey === name ? (dir === "asc" ? " ↑" : " ↓") : ""}
    </span>
  );
}

function fmt(v: unknown): string {
  if (typeof v === "number") return v.toFixed(4);
  if (v === null || v === undefined) return "—";
  return String(v);
}
