/* Thin client for the FastAPI backend.

   We talk to the backend through NEXT_PUBLIC_API_BASE (default
   http://127.0.0.1:8765) so the IBM token never has to leave the backend
   process — see /backend/.env. */

export const API_BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE) ||
  "http://127.0.0.1:8765";

export type ProblemParams = {
  N: number;
  lambda: number;            // λ₂ — variance weight
  lambda_return?: number;    // λ₁ — return weight (default 1.0)
  K: number;
  P_K: number;
  P_S?: number;
  sector_caps?: Record<number, number>;
  P_R: number;
  theta_risk: number;
  transaction_costs?: number[];
  seed: number;
  csv_data?: string | null;
};

export type ProblemResponse = {
  mode: "synthetic" | "csv";
  N: number;
  K: number;
  asset_names: string[];
  sectors: number[];
  mu: number[];
  Sigma: number[][];
  qubo_Q: number[][];
  qubo_offset: number;
  ising_J: number[][];
  ising_h: number[];
  ising_offset: number;
  weights: Record<string, unknown>;
};

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = await resp.text();
    try { detail = JSON.parse(detail).detail ?? detail; } catch { /* ignore */ }
    throw new Error(`POST ${path} → ${resp.status}: ${detail}`);
  }
  return resp.json() as Promise<T>;
}

export async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) {
    let detail = await resp.text();
    try { detail = JSON.parse(detail).detail ?? detail; } catch { /* ignore */ }
    throw new Error(`GET ${path} → ${resp.status}: ${detail}`);
  }
  return resp.json() as Promise<T>;
}

/* Problem builder. */
export const fetchProblem = (p: ProblemParams) =>
  postJson<ProblemResponse>("/problem", p);

/* Classical solvers. */
export const runBrute = (p: ProblemParams) =>
  postJson<Record<string, unknown>>("/classical/brute", p);
export const runSA = (p: ProblemParams & {
  T0: number; T_min: number; sweeps: number; restarts: number;
  move: "flip" | "swap"; init: "random" | "random_K"; sa_seed: number;
}) => postJson<Record<string, unknown>>("/classical/sa", p);
export const runMarkowitz = (p: ProblemParams & {
  frontier: boolean; frontier_n_lambda: number;
  frontier_lambda_min: number; frontier_lambda_max: number;
}) => postJson<Record<string, unknown>>("/classical/markowitz", p);

/* QAOA simulator. */
export type QaoaParams = ProblemParams & {
  p: number;
  mixer: "x" | "xy_ring";
  init_state: "uniform" | "dicke";
  optimizer: "COBYLA" | "SPSA" | "L-BFGS-B";
  max_iter: number;
  n_restarts: number;
  qaoa_seed: number;
  compute_classical_optimum?: boolean;
  n_top_bitstrings?: number;  // surface more bitstrings for off-shell diagnostics
};
export const runQaoa = (p: QaoaParams) =>
  postJson<Record<string, unknown>>("/qaoa/run", p);

export type LandscapeParams = ProblemParams & {
  mixer: "x" | "xy_ring";
  init_state: "uniform" | "dicke";
  n_gamma: number;
  n_beta: number;
  gamma_max: number;
  beta_max: number;
};
export const runLandscape = (p: LandscapeParams) =>
  postJson<{
    N: number; K: number; mixer: string; init_state: string;
    gamma: number[]; beta: number[]; energy: number[][];
    argmin: { gamma: number; beta: number; i: number; j: number; energy: number };
  }>("/qaoa/landscape", p);

/* Trials. */
export const fetchTrials = () => getJson<Array<{
  id: number; kind: string; created_at: string; summary: Record<string, unknown>;
}>>("/trials");
export const fetchTrial = (id: number) => getJson<{
  id: number; kind: string; created_at: string;
  params: Record<string, unknown>; results: Record<string, unknown>;
}>(`/trials/${id}`);
export async function deleteTrial(id: number): Promise<void> {
  const r = await fetch(`${API_BASE}/trials/${id}`, { method: "DELETE" });
  if (!r.ok) {
    let detail = await r.text();
    try { detail = JSON.parse(detail).detail ?? detail; } catch {}
    throw new Error(`DELETE /trials/${id} → ${r.status}: ${detail}`);
  }
}

/* Hardware. */
export const fetchBackends = () =>
  getJson<{ backends: Array<{
    name: string; qubits: number; queue_length: number;
    status: string; operational: boolean; simulator: boolean;
  }>; instance: string }>("/hw/backends");
export const submitHwJob = (p: {
  trial_id: number; backend_name: string; shots: number;
  error_mitigation: { readout?: boolean; dynamical_decoupling?: boolean };
}) => postJson<{ trial_id: number; job_id: string; backend: string }>("/hw/submit", p);
export const pollHwJob = (job_id: string) =>
  getJson<Record<string, unknown>>(`/hw/job/${job_id}`);
export const fetchCachedHw = () =>
  getJson<{ cached: Array<{ name: string; file: string; meta: Record<string, unknown>; results_summary: Record<string, unknown>; params: Record<string, unknown> }> }>("/hw/cached");
export const importCachedHw = (name: string) =>
  postJson<{ trial_id: number; name: string }>("/hw/cached/import", { name });
export const fetchCachedHwOne = (name: string) =>
  getJson<Record<string, unknown>>(`/hw/cached/${name}`);

/* Exports — return URLs for direct download */
export const exportUrl = (kind: string, trialId: number, suffix = "") =>
  `${API_BASE}/export/${kind}/${trialId}${suffix}`;
