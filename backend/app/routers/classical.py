"""Classical baselines (Phase 2).

Three solvers, one router:

* POST /classical/brute      — exhaustive enumeration over the K-shell.
* POST /classical/sa         — Metropolis simulated annealing with multi-restart.
* POST /classical/markowitz  — cvxpy convex relaxation + λ-sweep efficient frontier.

Every successful run is auto-recorded in the SQLite trials store.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from portfolio.classical import (
    brute_force,
    markowitz_continuous,
    simulated_annealing,
)
from portfolio.formulation import build_qubo, qubo_cost

from ..services.problem_builder import build_portfolio_problem, canonical_params
from ..services.trials_store import TrialsStore


router = APIRouter(prefix="/classical", tags=["classical"])


def _approx_ratio_vs_brute(method_cost: float, problem) -> tuple[Optional[float], Optional[float]]:
    """Compute (classical_optimum, approx_ratio = method/brute) on small problems.

    Skipped above N=18 (C(18,9) ≈ 48k — still cheap, but not worth blocking the
    API for). Returns (None, None) when skipped or when brute is degenerate.
    """
    n = problem.universe.n_assets
    if n > 18:
        return None, None
    bf = brute_force(problem)
    opt = float(bf.cost)
    if abs(opt) < 1e-15:
        return opt, None
    return opt, float(method_cost) / opt


# ---------------------------------------------------------------------------
# Shared base payload
# ---------------------------------------------------------------------------


class ProblemFields(BaseModel):
    """Re-used across every solver endpoint (deck-faithful naming)."""

    N: int = Field(default=8, ge=2, le=24)
    lambda_: float = Field(default=2.5, ge=0.0, alias="lambda")
    lambda_return: float = Field(default=1.0, ge=0.0)
    K: int = Field(default=3, ge=1)
    P_K: float = Field(default=5.0, ge=0.0)
    P_S: float = Field(default=0.0, ge=0.0)
    sector_caps: Optional[dict[int, int]] = None
    P_R: float = Field(default=0.5, ge=0.0)
    theta_risk: float = Field(default=0.04, ge=0.0)
    transaction_costs: Optional[list[float]] = None
    seed: Optional[int] = 7
    n_periods: int = Field(default=504, ge=30)
    csv_data: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# 2.1 — Brute force
# ---------------------------------------------------------------------------


class BruteRequest(ProblemFields):
    return_distribution: bool = Field(
        default=True,
        description="If true, include the full QUBO-cost histogram of every C(N,K) bitstring.",
    )
    distribution_max_n: int = Field(
        default=20,
        description="Hard cap on N for which we will enumerate. C(20,10) ≈ 185k — already heavy.",
    )


@router.post("/brute")
def classical_brute(req: BruteRequest) -> dict:
    if req.N > req.distribution_max_n:
        raise HTTPException(
            status_code=400,
            detail=f"brute force is restricted to N ≤ {req.distribution_max_n} (you asked for {req.N})",
        )

    payload = req.model_dump(by_alias=True)
    problem, mode = build_portfolio_problem(payload)

    t0 = time.perf_counter()
    res = brute_force(problem)
    runtime_s = time.perf_counter() - t0

    Q, c = build_qubo(problem)

    # Full energy distribution over all C(N,K) bitstrings (used for plot).
    energies = []
    if req.return_distribution:
        from itertools import combinations
        n = problem.universe.n_assets
        K = problem.K_target
        for combo in combinations(range(n), K):
            x = np.zeros(n, dtype=int)
            x[list(combo)] = 1
            energies.append(qubo_cost(Q, c, x))
        energies.sort()

    selected = [int(i) for i in np.where(res.x == 1)[0]]
    selected_names = [problem.universe.asset_names[i] for i in selected]

    # Brute *is* the optimum, by construction.
    classical_opt = float(res.cost)
    approx_ratio = None if abs(classical_opt) < 1e-15 else 1.0

    results = {
        "kind": "classical_brute",
        "cost": float(res.cost),
        "selected": selected,
        "selected_names": selected_names,
        "K": int(res.x.sum()),
        "classical_optimum": classical_opt,
        "approx_ratio": approx_ratio,
        "n_evaluations": int(res.n_evaluations),
        "runtime_s": float(runtime_s),
        "energy_distribution": energies if req.return_distribution else None,
        "best_x": res.x.tolist(),
    }

    trial_id = TrialsStore().record(
        kind="classical_brute",
        params=canonical_params(payload, mode, problem),
        results={**results, "energy_distribution_size": len(energies) if req.return_distribution else 0},
    )
    return {"trial_id": trial_id, **results}


# ---------------------------------------------------------------------------
# 2.2 — Simulated annealing
# ---------------------------------------------------------------------------


class SARequest(ProblemFields):
    T0: float = Field(default=1.0, gt=0.0)
    T_min: float = Field(default=1e-3, gt=0.0)
    sweeps: int = Field(default=200, ge=1, description="Sweeps over the N-bit state. Total Metropolis steps = sweeps * N.")
    restarts: int = Field(default=20, ge=1)
    sa_seed: int = Field(default=0, alias="sa_seed_")
    move: str = Field(default="flip", pattern="^(flip|swap)$")
    init: str = Field(default="random_K", pattern="^(random|random_K)$")

    model_config = ConfigDict(populate_by_name=True)


@router.post("/sa")
def classical_sa(req: SARequest) -> dict:
    payload = req.model_dump(by_alias=True)
    problem, mode = build_portfolio_problem(payload)
    n = problem.universe.n_assets
    n_steps = req.sweeps * n

    runs = []
    t0 = time.perf_counter()
    for r in range(req.restarts):
        res = simulated_annealing(
            problem,
            n_steps=n_steps,
            T0=req.T0,
            T1=req.T_min,
            seed=int(req.sa_seed + r),
            init=req.init,
            move=req.move,
        )
        # Down-sample the per-step history to a per-sweep trajectory for plot.
        history = np.asarray(res.history, dtype=float)
        if len(history) > req.sweeps + 1:
            idx = np.linspace(0, len(history) - 1, req.sweeps + 1).astype(int)
            sweep_history = history[idx].tolist()
        else:
            sweep_history = history.tolist()

        runs.append({
            "restart": r,
            "seed": int(req.sa_seed + r),
            "final_cost": float(res.cost),
            "final_x": res.x.tolist(),
            "trajectory_per_sweep": sweep_history,
            "n_evaluations": int(res.n_evaluations),
            "wall_time_s": float(res.wall_time),
        })
    total_runtime = time.perf_counter() - t0

    best = min(runs, key=lambda r: r["final_cost"])
    best_x = np.array(best["final_x"], dtype=int)
    selected = [int(i) for i in np.where(best_x == 1)[0]]
    selected_names = [problem.universe.asset_names[i] for i in selected]

    classical_opt, approx_ratio = _approx_ratio_vs_brute(best["final_cost"], problem)

    results = {
        "kind": "classical_sa",
        "cost": best["final_cost"],
        "selected": selected,
        "selected_names": selected_names,
        "K": int(best_x.sum()),
        "classical_optimum": classical_opt,
        "approx_ratio": approx_ratio,
        "best_restart": best["restart"],
        "n_restarts": req.restarts,
        "sweeps": req.sweeps,
        "n_evaluations": sum(r["n_evaluations"] for r in runs),
        "runtime_s": float(total_runtime),
        "runs": runs,
        "best_x": best["final_x"],
    }

    full_results = dict(results)
    full_results["restart_costs"] = [r["final_cost"] for r in runs]
    trial_id = TrialsStore().record(
        kind="classical_sa",
        params=canonical_params(payload, mode, problem) | {
            "T0": req.T0, "T_min": req.T_min, "sweeps": req.sweeps,
            "restarts": req.restarts, "move": req.move, "init": req.init,
            "sa_seed": req.sa_seed,
        },
        results=full_results,
    )
    return {"trial_id": trial_id, **results}


# ---------------------------------------------------------------------------
# 2.3 — Continuous Markowitz + efficient frontier sweep
# ---------------------------------------------------------------------------


class MarkowitzRequest(ProblemFields):
    frontier: bool = Field(default=True)
    frontier_n_lambda: int = Field(default=50, ge=2, le=200)
    frontier_lambda_min: float = Field(default=0.05, gt=0.0)
    frontier_lambda_max: float = Field(default=20.0, gt=0.0)


@router.post("/markowitz")
def classical_markowitz(req: MarkowitzRequest) -> dict:
    payload = req.model_dump(by_alias=True)
    problem, mode = build_portfolio_problem(payload)
    n = problem.universe.n_assets

    t0 = time.perf_counter()
    res = markowitz_continuous(problem)
    primary_runtime = time.perf_counter() - t0

    continuous_w = res.extra["continuous_w"] if res.extra else np.zeros(n)
    continuous_w = np.asarray(continuous_w).tolist()
    selected = [int(i) for i in np.where(res.x == 1)[0]]
    selected_names = [problem.universe.asset_names[i] for i in selected]

    # Efficient frontier sweep — vary lam_variance over a log-spaced grid;
    # everything else (return weight, penalties) stays put. The frontier is
    # plotted in (variance, return) coordinates of the *continuous* weights.
    frontier = None
    if req.frontier:
        lambdas = np.geomspace(
            req.frontier_lambda_min, req.frontier_lambda_max, req.frontier_n_lambda
        )
        points = []
        original_lam = problem.weights.lam_variance
        for lam in lambdas:
            problem.weights.lam_variance = float(lam)
            try:
                fr = markowitz_continuous(problem)
            except Exception:
                continue
            w = np.asarray(fr.extra["continuous_w"])
            ret = float(problem.universe.mu @ w)
            var = float(w @ problem.universe.sigma @ w)
            points.append({
                "lambda": float(lam),
                "return": ret,
                "variance": var,
                "vol": float(np.sqrt(max(var, 0.0))),
                "weights": w.tolist(),
            })
        problem.weights.lam_variance = original_lam
        frontier = points

    total_runtime = time.perf_counter() - t0

    classical_opt, approx_ratio = _approx_ratio_vs_brute(float(res.cost), problem)

    results = {
        "kind": "classical_markowitz",
        "cost": float(res.cost),
        "selected": selected,
        "selected_names": selected_names,
        "K": int(res.x.sum()),
        "classical_optimum": classical_opt,
        "approx_ratio": approx_ratio,
        "continuous_weights": continuous_w,
        "continuous_objective": float(res.extra["continuous_obj"]) if res.extra else None,
        "best_x": res.x.tolist(),
        "primary_runtime_s": primary_runtime,
        "runtime_s": total_runtime,
        "frontier": frontier,
    }

    summary_results = {
        k: v for k, v in results.items()
        if k not in ("frontier", "continuous_weights")
    }
    summary_results["frontier_n_points"] = len(frontier) if frontier else 0
    trial_id = TrialsStore().record(
        kind="classical_markowitz",
        params=canonical_params(payload, mode, problem) | {
            "frontier": req.frontier,
            "frontier_n_lambda": req.frontier_n_lambda,
            "frontier_lambda_min": req.frontier_lambda_min,
            "frontier_lambda_max": req.frontier_lambda_max,
        },
        results=summary_results,
    )
    return {"trial_id": trial_id, **results}
