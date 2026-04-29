# Quantum-Classical Portfolio Optimisation

24-hour evaluation challenge — PROJECT-Q × Zuntenium pilot (April 2026).

A working prototype of a constrained, multi-objective portfolio
optimisation problem solved with both classical baselines (brute force,
greedy, simulated annealing with two move sets, convex relaxation) and a
hand-rolled QAOA hybrid solver, plus a critical comparison.

The full mathematical formulation, results, and critical evaluation are
in **[RESEARCH_NOTE.md](RESEARCH_NOTE.md)**. This README only documents
the code layout and how to run it.

## Repository layout

```
backend/             FastAPI service — wraps portfolio/ as HTTP endpoints
  app/               routers, services, config
  tests/             pytest suite (41 tests)
  scripts/           seed_hw_cache.py — generates demo /artifacts
  .env.example       template for IBM Quantum credentials
frontend/            Next.js + TypeScript + Tailwind UI
  src/app/           Problem · Classical · QAOA · Landscape · Hardware · Trials · Judge
artifacts/           cached hardware snapshots for offline demo
portfolio/           original Python package (data, formulation, classical, quantum)
experiments/         CLI runner from the original submission
tests/               original portfolio-package tests
BUILD_PLAN.md        7-phase build plan (this is what was built today)
RESEARCH_NOTE.md     math, methods, results, evaluation
requirements.txt     backend Python deps
```

## Interactive UI quickstart (the 24-hour build)

Two terminals — backend on :8765, frontend on :3000.

```bash
# 1. Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp backend/.env.example backend/.env       # then paste your IBM token (optional)
cd backend && uvicorn app.main:app --reload --host 127.0.0.1 --port 8765

# 2. Frontend (new terminal)
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

The IBM token only ever lives in `backend/.env`. The frontend never sees it.

### Running tests

```bash
source .venv/bin/activate
cd backend && python -m pytest tests/ -q     # 41 tests, ~30 s
```

### Cached hardware snapshots

Three stand-in artifacts ship in `/artifacts` so the Hardware tab demos cleanly without an IBM token. They simulate Aer + readout bit-flip noise and are clearly tagged `meta.stand_in: true`. Replace with real runs:

```bash
# 1. Submit a real job from the Hardware tab in the UI (needs a token)
# 2. Once it ingests, copy backend/trials.db → /artifacts manually, OR re-seed:
cd backend && python -m scripts.seed_hw_cache
```

### Live deck plots

The four data-claim plots in `quantum_portfolio_deck.html` (slides 15, 19, 20, 21) are `<img>` tags pointing at SVGs in `/artifacts/deck_plots/`, generated from the live solvers. Regenerate after math/scaling changes:

```bash
cd backend && python -m scripts.generate_deck_plots
```

This runs brute / SA / Markowitz / QAOA on representative N values and writes:
- `slide_15_optimizers.svg` — COBYLA vs SPSA vs L-BFGS-B trajectories at N=10 K=4 p=2
- `slide_19_approx_ratio.svg` — bar chart, 5 methods × N∈{8, 12, 14}
- `slide_20_convergence.svg` — SA per-flip vs QAOA per-eval vs CVX one-shot at N=12
- `slide_21_walltime.svg` — wall-clock per operation across N∈{8, 10, …, 18}

There's also a separate verification harness that reproduces the deck's quantitative claims (slides 13–17 + 18–24): `python -m scripts.run_deck_trials` and `python -m scripts.verify_deck_claims`. Reports go to `/artifacts/deck_trials/`.

### Trust the metric: stress test

If `approx_ratio = 1.0` for every method on the deck-19 chart looks suspicious, run the stress test:

```bash
cd backend && python -m scripts.stress_test_methods
```

Five sections deliberately try to break the metric:
- **random K-feasible portfolios** — no optimization, ratio ≈ 0.40 mean
- **SA with tiny budgets** — 1 flip → ratio 0.83; 100 flips → ratio 1.00
- **QAOA without top-M rescue** — argmax decode gives catastrophic ratios (the QAOA distribution at random θ concentrates on infeasible bitstrings)
- **CVX across 7 seeds** — ratios swing from 0.83 to 1.00 depending on instance
- **Degenerate inputs** (`x=0`, `x=1`) — ratios reflect cardinality-penalty disasters

The 1.0s on the deck plots are *earned* by multistart + top-M decoding on tractable instances, not given by construction. Report at `artifacts/deck_trials/STRESS_TEST.md`.



## Original CLI quick start (preserved from the original submission)

```bash
pip install -r requirements.txt
python -m tests.test_formulation         # ~ 1 s, 4 tests
python -m experiments.run_main           # ~ 10 min on a laptop CPU
```

After the experiment finishes, `experiments/results/` contains:

| File | What |
|---|---|
| `summary.txt`            | Tabular comparison of every solver. |
| `convergence.png`        | Per-iteration trajectories (SA + QAOA at p=1/2/3). |
| `qaoa_distribution.png`  | Top-24 most-probable bitstrings under the QAOA distribution; the optimum is highlighted. |
| `frontier.png`           | Risk/return scatter of every C(N,K) feasible portfolio with each solver's pick overlaid. |
| `scaling.png`            | Wall-time vs. N for brute force vs. QAOA. |
| `depth_study.txt`        | QAOA approximation ratio and P(opt) as a function of circuit depth p = 1..5. |

## Choosing the problem

Edit `experiments/run_main.py:main_experiment` to change asset count,
sector count, the prior portfolio, or any of the scalarisation weights
in `ObjectiveWeights`. The QUBO and QAOA circuit are rebuilt
automatically.

## What's interesting / honest about this submission

* The QUBO build is checked against a closed-form objective on 256
  random bitstrings to floating-point precision (see tests).
* The optimum-on-the-K-shell property is proven empirically (`brute_force`
  and `brute_force_full` agree on the optimum) — this is how we verify
  the cardinality penalty multiplier is large enough.
* Two SA variants are included so the classical baseline is *fair*: a
  naive bit-flip SA that fights the cardinality penalty, and a swap-move
  SA that respects the constraint and finds the optimum trivially.
* The QAOA result is *not* hidden behind a one-shot "pick highest
  probability bitstring": we implement the standard sampler-style
  decode (top-64 most-probable, lowest QUBO cost wins) **and** report the
  raw success probability `P(opt)` separately.
* Convergence / scalability / depth results are reported as observed,
  not curated — including the non-monotone-in-depth behaviour that QAOA
  shows on this instance.

The conclusion of this work is that **no quantum advantage is
demonstrated**, and `RESEARCH_NOTE.md §6` explains exactly why we believe
that is the correct answer for this problem at this scale on today's
hardware.
