"""Classical baselines.

We deliberately implement four very different classical approaches so that
the quantum result is judged against a *strong* set of references rather than
a single straw-man:

1. ``brute_force``       -- exhaustive enumeration over the cardinality-K
                            shell. Gives the global optimum at the cost of
                            C(N,K) cost evaluations. Only feasible for small N.

2. ``brute_force_full``  -- enumerate the full 2^N hypercube. Treats the
                            cardinality penalty exactly as the QAOA does
                            (i.e. soft, via the cost). Useful for verifying
                            that the *penalty-augmented* QUBO has its global
                            minimum on the K-cardinality shell.

3. ``greedy``            -- forward-selection heuristic on the QUBO.

4. ``simulated_annealing`` -- a textbook SA on the QUBO. This is the *real*
                            classical competitor to QAOA on Ising-like
                            problems and is the benchmark to beat.

5. ``markowitz_continuous`` -- Lagrangian / convex relaxation: drop the
                            cardinality constraint, allow w in [0,1]^N with
                            sum w = 1, and solve the resulting QP with cvxpy.
                            Then project onto the K-asset shell by picking
                            the top-K weights.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .formulation import (
    PortfolioProblem,
    build_qubo,
    qubo_cost,
)


@dataclass
class SolverResult:
    name: str
    x: np.ndarray
    cost: float
    wall_time: float
    n_evaluations: int
    history: Optional[List[float]] = None
    extra: Optional[dict] = None


# ---------------------------------------------------------------------------
# Exhaustive enumeration
# ---------------------------------------------------------------------------

def brute_force(problem: PortfolioProblem) -> SolverResult:
    """Enumerate all C(N, K_target) cardinality-K bitstrings."""
    n = problem.universe.n_assets
    K = problem.K_target
    Q, c = build_qubo(problem)

    t0 = time.perf_counter()
    best_x, best_cost = None, np.inf
    n_eval = 0
    for combo in itertools.combinations(range(n), K):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        f = qubo_cost(Q, c, x)
        n_eval += 1
        if f < best_cost:
            best_cost, best_x = f, x
    wall = time.perf_counter() - t0
    return SolverResult("brute_force_K", best_x, best_cost, wall, n_eval)


def brute_force_full(problem: PortfolioProblem) -> SolverResult:
    """Enumerate the full 2^N hypercube (no cardinality restriction)."""
    n = problem.universe.n_assets
    Q, c = build_qubo(problem)

    t0 = time.perf_counter()
    best_x, best_cost = None, np.inf
    n_eval = 0
    for k in range(2 ** n):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
        f = qubo_cost(Q, c, x)
        n_eval += 1
        if f < best_cost:
            best_cost, best_x = f, x
    wall = time.perf_counter() - t0
    return SolverResult("brute_force_full", best_x, best_cost, wall, n_eval)


# ---------------------------------------------------------------------------
# Greedy forward selection on the QUBO
# ---------------------------------------------------------------------------

def greedy(problem: PortfolioProblem) -> SolverResult:
    """Add the asset with the largest immediate cost reduction until K assets.

    O(N^2 * K) cost evaluations. Misses non-monotone interactions (which is
    the point: greedy is a proper *lower-quality* baseline).
    """
    n = problem.universe.n_assets
    K = problem.K_target
    Q, c = build_qubo(problem)

    x = np.zeros(n, dtype=int)
    history = [qubo_cost(Q, c, x)]
    t0 = time.perf_counter()
    n_eval = 1

    for _ in range(K):
        best_idx, best_f = None, np.inf
        for i in range(n):
            if x[i] == 1:
                continue
            x[i] = 1
            f = qubo_cost(Q, c, x)
            n_eval += 1
            if f < best_f:
                best_f, best_idx = f, i
            x[i] = 0
        x[best_idx] = 1
        history.append(qubo_cost(Q, c, x))

    wall = time.perf_counter() - t0
    return SolverResult("greedy", x, history[-1], wall, n_eval, history=history)


# ---------------------------------------------------------------------------
# Simulated annealing on the QUBO
# ---------------------------------------------------------------------------

def simulated_annealing(
    problem: PortfolioProblem,
    n_steps: int = 5000,
    T0: float = 1.0,
    T1: float = 1e-3,
    seed: int = 0,
    init: str = "random_K",
    move: str = "flip",
) -> SolverResult:
    """Metropolis-Hastings on the QUBO with geometric cooling.

    `init`:
        'random'     - uniform random binary
        'random_K'   - random cardinality-K start (gives SA a fair head start)
    `move`:
        'flip'       - single-bit flip. Explores the full hypercube but
                       must climb the cardinality penalty wall.
        'swap'       - swap one held asset with one un-held asset (preserves
                       K). The constraint-aware classical move set; this is
                       the *strong* SA baseline.
    """
    rng = np.random.default_rng(seed)
    n = problem.universe.n_assets
    K = problem.K_target
    Q, c = build_qubo(problem)

    if init == "random":
        x = rng.integers(0, 2, size=n)
    elif init == "random_K":
        x = np.zeros(n, dtype=int)
        x[rng.choice(n, size=K, replace=False)] = 1
    else:
        raise ValueError(init)

    f = qubo_cost(Q, c, x)
    best_x, best_f = x.copy(), f
    history = [f]

    cooling = (T1 / T0) ** (1.0 / max(n_steps - 1, 1))
    T = T0

    t0 = time.perf_counter()
    n_eval = 1
    for _ in range(n_steps):
        if move == "flip":
            i = rng.integers(0, n)
            x[i] ^= 1
            f_new = qubo_cost(Q, c, x)
            n_eval += 1
            dE = f_new - f
            accept = (dE <= 0) or (rng.random() < np.exp(-dE / max(T, 1e-12)))
            if accept:
                f = f_new
                if f < best_f:
                    best_f, best_x = f, x.copy()
            else:
                x[i] ^= 1  # reject
        elif move == "swap":
            held = np.where(x == 1)[0]
            unheld = np.where(x == 0)[0]
            if len(held) == 0 or len(unheld) == 0:
                history.append(best_f)
                T *= cooling
                continue
            i = held[rng.integers(0, len(held))]
            j = unheld[rng.integers(0, len(unheld))]
            x[i], x[j] = 0, 1
            f_new = qubo_cost(Q, c, x)
            n_eval += 1
            dE = f_new - f
            accept = (dE <= 0) or (rng.random() < np.exp(-dE / max(T, 1e-12)))
            if accept:
                f = f_new
                if f < best_f:
                    best_f, best_x = f, x.copy()
            else:
                x[i], x[j] = 1, 0
        else:
            raise ValueError(move)

        history.append(best_f)
        T *= cooling
    wall = time.perf_counter() - t0

    return SolverResult(
        f"sim_annealing_{move}", best_x, best_f, wall, n_eval, history=history
    )


# ---------------------------------------------------------------------------
# Continuous Markowitz (relaxation) via cvxpy + project to top-K
# ---------------------------------------------------------------------------

def markowitz_continuous(problem: PortfolioProblem) -> SolverResult:
    """Drop cardinality, solve mean-variance QP, project to top-K by weight.

    The relaxation gives an *upper* bound on what a continuous portfolio can
    achieve and a sensible warm-start. The projection step is heuristic: it
    is well known that the cardinality-constrained mean-variance problem is
    NP-hard, so the round-off is generically suboptimal. This is precisely
    why combinatorial heuristics (and, hopefully, quantum) are interesting.
    """
    import cvxpy as cp

    u = problem.universe
    n = u.n_assets
    K = problem.K_target
    w_lams = problem.weights

    t0 = time.perf_counter()

    w = cp.Variable(n, nonneg=True)
    risk = cp.quad_form(w, cp.psd_wrap(u.sigma))
    ret = u.mu @ w
    # Use the same return/variance trade-off as the binary model so the
    # comparison is on equal terms (modulo cardinality).
    objective = cp.Minimize(-w_lams.lam_return * ret + w_lams.lam_variance * risk)
    constraints = [cp.sum(w) == 1.0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    w_star = np.array(w.value).flatten()

    # Project: pick K largest weights, set to 1/K.
    top_K = np.argsort(-w_star)[:K]
    x = np.zeros(n, dtype=int)
    x[top_K] = 1

    Q, c = build_qubo(problem)
    f = qubo_cost(Q, c, x)
    wall = time.perf_counter() - t0

    return SolverResult(
        "markowitz_relaxed_topK",
        x, f, wall, 1,
        extra=dict(continuous_w=w_star, continuous_obj=float(prob.value)),
    )
