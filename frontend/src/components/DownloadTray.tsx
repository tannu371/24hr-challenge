"use client";

/* Per-trial download tray. Buttons are gated by trial kind so users never see
   downloads that would 4xx because the underlying data doesn't exist for that
   kind (e.g. histogram on a Markowitz trial). */

import { API_BASE } from "@/lib/api";

type Kind =
  | "classical_brute"
  | "classical_sa"
  | "classical_markowitz"
  | "qaoa_sim"
  | "qaoa_hw"
  | string;

type Props = {
  trialId: number;
  /** trial kind — drives which buttons are shown */
  kind?: Kind;
  /** legacy boolean — if `kind` not given, treat as qaoa_sim/hw vs classical */
  isQaoa?: boolean;
};

export default function DownloadTray({ trialId, kind, isQaoa }: Props) {
  const u = (path: string) => `${API_BASE}/export${path}`;

  const k: Kind = kind ?? (isQaoa === false ? "classical_brute" : "qaoa_sim");
  const isQuantum = k === "qaoa_sim" || k === "qaoa_hw";
  const hasTrajectory = isQuantum || k === "classical_sa";
  // brute has energy_distribution we can histogram; QAOA has top_bitstrings
  const hasHistogram = isQuantum || k === "classical_brute";

  return (
    <div className="flex flex-wrap gap-2">
      {isQuantum && <>
        <a className="btn" href={u(`/qasm/${trialId}`)} download>QASM 3</a>
        <a className="btn" href={u(`/qiskit/${trialId}`)} download>Qiskit .py</a>
        <a className="btn" href={u(`/pennylane/${trialId}`)} download>PennyLane .py</a>
        <a className="btn" href={u(`/circuit/${trialId}.svg`)} download>circuit.svg</a>
        <a className="btn" href={u(`/circuit/${trialId}.svg?transpiled=true`)} download>circuit (transpiled).svg</a>
      </>}
      {hasTrajectory && <>
        <a className="btn" href={u(`/plot/${trialId}/trajectory.svg`)} download>trajectory.svg</a>
        <a className="btn" href={u(`/plot/${trialId}/trajectory.csv`)} download>trajectory.csv</a>
      </>}
      {hasHistogram && <>
        <a className="btn" href={u(`/plot/${trialId}/histogram.svg`)} download>histogram.svg</a>
        <a className="btn" href={u(`/plot/${trialId}/histogram.csv`)} download>histogram.csv</a>
      </>}
      <a className="btn" href={u(`/plot/${trialId}/comparison.svg`)} download>comparison.svg</a>
      {isQuantum && (
        <a className="btn" href={u(`/plot/${trialId}/landscape.svg`)} download>landscape.svg</a>
      )}
      <a className="btn primary" href={u(`/bundle/${trialId}`)} download>bundle.zip</a>
    </div>
  );
}
