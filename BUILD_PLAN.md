# Hybrid Quantum–Classical Portfolio Optimizer — Build Plan

A small-tasks breakdown for turning your 26-slide deck into an interactive web app that lets you (and the judges) tweak every classical and quantum parameter, run on simulator and real IBM hardware, and download circuits/code/plots per trial.

---

## 0. Do these BEFORE you write any code

- [ ] **Revoke the IBM Quantum token you pasted in chat.** Generate a new one. It only ever lives in `backend/.env` (gitignored). Add `.env` to `.gitignore` first thing.
- [ ] Decide stack. Recommended given your existing skills:
  - Backend: **FastAPI + Qiskit + qiskit-ibm-runtime + PennyLane + cvxpy + SQLite**
  - Frontend: **Next.js + TypeScript + Tailwind** (reuse the deck's CSS tokens), or vanilla Vite SPA if you want to move faster
- [ ] Repo layout:
  ```
  /backend         FastAPI app, solvers, exports
  /frontend        Next.js app
  /artifacts       cached hardware results for offline demo
  BUILD_PLAN.md
  ```

## Architecture (one picture worth memorizing)

```
Frontend (browser)  ──HTTPS──►  FastAPI backend  ──qiskit-ibm-runtime──►  IBM Quantum
  sliders, plots,                qiskit / pennylane                       hardware
  downloads, judge                cvxpy, numpy
  mode                            SQLite (trials)
```

Token NEVER leaves the backend. Frontend talks only to your FastAPI server.

---

## Phase 1 — Backend skeleton & problem definition

**Task 1.1 — FastAPI scaffold**
- FastAPI app, CORS for `http://localhost:3000`, `/healthz`, `python-dotenv`.
- Done when: `uvicorn` boots and `/healthz` returns 200.

**Task 1.2 — Problem builder**
- `POST /problem` accepts `{N, lambda, K, P_K, P_R, seed?, csv_data?}`.
- Returns `{mu, Sigma, qubo_Q, qubo_offset, ising_J, ising_h, ising_offset}`.
- Synthetic mode: seeded log-return generator → sample mean μ and covariance Σ.
- CSV mode: parse uploaded daily prices → log returns → μ, Σ.
- Done when: response matches a hand-computed value for N=4, fixed seed.

**Task 1.3 — QUBO ↔ Ising conversion utilities**
- Pure functions, no Qiskit dependency. Substitute `x_i = (1 - z_i)/2` correctly with offsets.
- Done when: unit tests verify round-trip on a 6-asset case (`pytest backend/tests/test_qubo.py`).

**Task 1.4 — Trials store (SQLite)**
- Every solve writes `{trial_id, kind, params_json, results_json, created_at}`.
- `kind ∈ {classical_brute, classical_sa, classical_markowitz, qaoa_sim, qaoa_hw}`.
- `GET /trials` (newest first), `GET /trials/{id}` (full record).
- Done when: every run from later phases auto-records here.

---

## Phase 2 — Classical baselines

**Task 2.1 — Brute force (small N)**
- `POST /classical/brute` enumerates all $\binom{N}{K}$ feasible portfolios for N ≤ 20.
- Returns optimum + full energy distribution for plotting.
- Done when: matches analytical optimum on N=8 fixed seed.

**Task 2.2 — Simulated annealing**
- `POST /classical/sa` with `{T0, T_min, sweeps, restarts, seed}`.
- Metropolis spin-flip on Ising form.
- Returns final state + trajectory (energy per sweep) for plotting.
- Done when: hits brute-force optimum ≥95% of the time on N=12 with 20 restarts.

**Task 2.3 — Continuous Markowitz**
- `POST /classical/markowitz` solves the convex relaxation with `cvxpy`.
- Returns weights for given λ, plus efficient frontier sweep over a λ range.
- Done when: frontier renders smoothly across 50 λ values.

---

## Phase 3 — Quantum (Aer simulator)

**Task 3.1 — Cost Hamiltonian builder**
- Convert Ising `(J, h)` → Qiskit `SparsePauliOp`.
- Sanity check: ⟨z|H_C|z⟩ matches direct Ising evaluation on a few computational basis states.

**Task 3.2 — QAOA ansatz with mixer choice**
- Parameterized depth p; two mixer options:
  - **X-mixer** (standard, but wastes amplitude on infeasible bitstrings — the slide-17 problem).
  - **XY-ring mixer** (preserves Hamming weight → respects budget constraint natively).
- Initial state: uniform superposition or Dicke-K state (pair the latter with XY-mixer).
- Returns a `QuantumCircuit` and a parameter binder.

**Task 3.3 — Energy evaluator on Aer statevector**
- Given (γ, β) vectors, return ⟨H_C⟩. This is what the outer optimizer calls.

**Task 3.4 — Outer optimization loop**
- `POST /qaoa/run` accepts `{p, mixer, init_state, optimizer, max_iter, n_restarts, seed}`.
- Optimizers: COBYLA, SPSA, L-BFGS-B with multistart (the three you tried in the deck).
- Stream progress over Server-Sent Events: `{iter, energy, gamma, beta}` per step.
- Returns: best params, final energy, sampled distribution (1024 shots), top-10 bitstrings, approximation ratio (when brute-force optimum exists).

**Task 3.5 — Landscape sweep (real version of your slide-18 playground)**
- `POST /qaoa/landscape` for p=1: 2D grid of ⟨H_C⟩ over (γ, β). Returns the matrix.

---

## Phase 4 — Real hardware (IBM Quantum)

**Task 4.1 — Backend discovery**
- `GET /hw/backends` → list available IBM backends with `{name, qubits, queue_length, status}` via `QiskitRuntimeService.backends()`.

**Task 4.2 — Job submission**
- `POST /hw/submit` with `{trial_id, backend_name, shots, error_mitigation: {readout, dynamical_decoupling}}`.
- Transpile current QAOA circuit to the backend, submit via `Sampler` primitive, return `job_id`.
- Persist `job_id ↔ trial_id` mapping in SQLite.

**Task 4.3 — Job polling**
- `GET /hw/job/{job_id}` → `{status, queue_pos, est_start, results?}`.
- Frontend polls every 5s.

**Task 4.4 — Result ingestion**
- On completion, parse counts, compute energy, write into trial record, set `status=complete`.

**Task 4.5 — Hardware result cache (the demo safety net)**
- `GET /hw/cached` returns prebuilt hardware results for N=8 and N=12.
- You run hardware trials in advance, snapshot trial records, ship the JSON in `/artifacts`.
- This is the single most important task for surviving demo day if the queue is backed up.

---

## Phase 5 — Code & artifact exports

Every export takes `trial_id` and returns a file download.

- **Task 5.1 — QASM 3** → `GET /export/qasm/{trial_id}`
- **Task 5.2 — Standalone Qiskit `.py`** → self-contained script that reproduces the run with current params baked in
- **Task 5.3 — Standalone PennyLane `.py`** → same QAOA reimplemented with `qml.qaoa.cost_layer` / `mixer_layer`
- **Task 5.4 — Circuit diagrams** → SVG + PNG via `circuit.draw('mpl')`, both pre- and post-transpilation
- **Task 5.5 — Plots** → energy trajectory, probability histogram, landscape heatmap, classical-vs-quantum bar chart — each as SVG + PNG + raw CSV
- **Task 5.6 — Full bundle** → `GET /export/bundle/{trial_id}` returns a zip with everything

---

## Phase 6 — Frontend

Reuse the deck's CSS tokens (`--gold`, `--cyan`, Fraunces, IBM Plex). Single-page app with anchor sections mirroring the deck flow.

**Task 6.1 — Layout shell**
- Same typography and accent system as the deck.
- Dark/light toggle (you already have this in the deck JS).
- Sidebar nav: Problem · Classical · Quantum (Sim) · Hardware · Trials · Exports · **Judge Mode**.

**Task 6.2 — Problem panel**
- Sliders: N (4–20), λ (0–1), K (1–N), log₁₀(P_K) (−1 to 4), log₁₀(P_R) (−1 to 4).
- CSV upload OR synthetic with seed.
- Live preview: μ as bar chart, Σ as heatmap.

**Task 6.3 — Classical panel**
- Three cards (brute / SA / Markowitz), each with its own params.
- "Run" → result card (selected portfolio, objective, runtime).
- Efficient frontier plot after Markowitz.

**Task 6.4 — Quantum simulator panel**
- Sliders: p (1–6), max_iter (10–500).
- Dropdowns: mixer, initial state, optimizer.
- Live energy curve during optimization (consume the SSE stream from Task 3.4).
- Final card: bitstring distribution (bar chart of top 10), approximation ratio gauge.

**Task 6.5 — Landscape playground**
- Real version of slide 18, but with actual QAOA energy from `/qaoa/landscape`.
- 2D heatmap of ⟨H_C⟩(γ, β), click to set initial point, then run optimization from there.

**Task 6.6 — Hardware panel**
- Backend dropdown with live queue lengths.
- Shots input, error mitigation toggles.
- Submit → job card with status/polling.
- "Use cached result" button (Task 4.5) for the offline-demo path.

**Task 6.7 — Trials table**
- All runs sortable by time, kind, N, p, energy, approximation ratio.
- Click a row → detail view with params, plots, downloads.
- Pin trials → they appear in Judge Mode.

**Task 6.8 — Judge Mode**
- Fullscreen, minimal chrome. Shows a curated sequence of pinned trials with one-click navigation.
- Each step: setup → result → key chart → next.
- Goal: press space, walk the room through your best trials in 5–7 minutes.
- This is the feature that makes the project demo-friendly. Don't skip it.

**Task 6.9 — Download tray**
- Per-trial buttons for QASM / Qiskit / PennyLane / circuit SVG / plots / full bundle.
- One-click copy of code blocks.

---

## Phase 7 — Polish & demo prep

- **Task 7.1** — Error states & graceful degradation. If IBM unreachable → show cached results with a clear banner. If optimization diverges → show trajectory so far instead of erroring.
- **Task 7.2** — README, `.env.example`, Dockerfile. Anyone (including a judge) should be able to clone + `docker-compose up`.
- **Task 7.3** — Pre-recorded hardware bundle. Run 5 representative trials on real hardware before demo day, commit the JSON to `/artifacts`. Judge Mode pulls from here by default.
- **Task 7.4** — Dry-run demo end-to-end. 7-minute target. Time yourself, trim what doesn't earn its place.

---

## Notes on a few traps

- **IBM Quantum Platform migration.** Use `qiskit-ibm-runtime` (the new SDK), not the deprecated `qiskit-ibmq-provider`. Auth is via `QiskitRuntimeService.save_account(token=...)` once, then it reads from the cache. Keep that cache file out of git.
- **Transpilation matters on real hardware.** A 16-qubit QAOA at p=3 transpiled naively can balloon to thousands of 2-qubit gates. Use `optimization_level=3` and pick a hardware-efficient ansatz if depth becomes a problem.
- **Aer statevector tops out around 25 qubits** on a typical laptop. Above that, switch to MPS or sampling backends — your slide 21 already acknowledges this wall.
- **SSE for live progress > polling** for the optimization curve. It's one less round trip per iteration and the UI stays smooth.
- **For the judge demo, never depend on a live IBM queue.** Cached results + a clearly-labeled "live submission" button you only press if the queue is short.
