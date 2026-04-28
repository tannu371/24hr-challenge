# Hybrid Quantum-Classical Portfolio Optimisation

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
portfolio/
  data.py            two-factor synthetic asset universe (PSD covariance)
  formulation.py     multi-objective cost; QUBO; QUBO -> Ising; -> Pauli
  classical.py       brute_force, greedy, SA (flip + swap), Markowitz QP
  quantum.py         hand-rolled QAOA on qiskit primitives
  analysis.py        tables and plots
experiments/
  run_main.py        single-instance + depth study + scaling study
  results/           generated artefacts (PNG plots, summary.txt)
tests/
  test_formulation.py  algebraic equivalence checks (QUBO == Ising == cost fn)
RESEARCH_NOTE.md     full write-up: math, methods, results, evaluation
requirements.txt
```

## Quick start

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
