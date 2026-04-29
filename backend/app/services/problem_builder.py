"""Shared utility: turn a frontend payload into a PortfolioProblem.

Used by /problem and by every solver endpoint so the parameter naming
(`lambda`, `lambda_return`, `P_K`, `P_R`, `P_S`, `theta_risk`,
``sector_caps``, ``transaction_costs``, ``K``, `N`, `seed`, `csv_data`)
is consistent across the API surface.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from portfolio.formulation import ObjectiveWeights, PortfolioProblem

from .csv_returns import universe_from_price_csv
from .synthetic import generate_log_returns_universe


def build_portfolio_problem(payload: dict[str, Any]) -> tuple[PortfolioProblem, str]:
    """Construct (problem, mode) from a request dict that has the standard fields.

    Mode is "csv" if csv_data is non-empty, else "synthetic".
    """
    csv_data: Optional[str] = payload.get("csv_data")
    if csv_data is not None and str(csv_data).strip():
        universe = universe_from_price_csv(csv_data)
        mode = "csv"
    else:
        universe = generate_log_returns_universe(
            n_assets=int(payload.get("N", 8)),
            seed=int(payload.get("seed", 7) or 7),
            n_periods=int(payload.get("n_periods", 504)),
        )
        mode = "synthetic"

    K_target = int(payload.get("K", max(1, universe.n_assets // 3)))
    if not (1 <= K_target <= universe.n_assets):
        raise ValueError(f"K={K_target} out of range for N={universe.n_assets}")

    lam_var = float(payload.get("lambda", payload.get("lambda_", 2.5)))
    lam_ret = float(payload.get("lambda_return", 1.0))
    P_K = float(payload.get("P_K", 5.0))
    P_R = float(payload.get("P_R", 0.5))
    P_S = float(payload.get("P_S", 0.0))
    theta_risk = float(payload.get("theta_risk", 0.04))

    weights = ObjectiveWeights(
        lam_return=lam_ret,
        lam_variance=lam_var,
        P_K=P_K,
        P_S=P_S,
        P_R=P_R,
        theta_risk=theta_risk,
    )

    sector_caps_raw = payload.get("sector_caps") or {}
    sector_caps = {int(k): int(v) for k, v in sector_caps_raw.items()}

    tc_raw = payload.get("transaction_costs")
    if tc_raw is not None and len(tc_raw) == universe.n_assets:
        transaction_costs = np.asarray(tc_raw, dtype=float)
    else:
        transaction_costs = np.zeros(universe.n_assets, dtype=float)

    return (
        PortfolioProblem(
            universe=universe,
            K_target=K_target,
            weights=weights,
            sector_caps=sector_caps,
            transaction_costs=transaction_costs,
        ),
        mode,
    )


def canonical_params(payload: dict[str, Any], mode: str, problem: PortfolioProblem) -> dict[str, Any]:
    """Project a request payload to a stable shape for the trials store."""
    return {
        "mode": mode,
        "N": problem.universe.n_assets,
        "K": problem.K_target,
        "lambda": problem.weights.lam_variance,
        "lambda_return": problem.weights.lam_return,
        "P_K": problem.weights.P_K,
        "P_R": problem.weights.P_R,
        "P_S": problem.weights.P_S,
        "theta_risk": problem.weights.theta_risk,
        "sector_caps": dict(problem.sector_caps) if problem.sector_caps else {},
        "transaction_costs": (
            problem.transaction_costs.tolist()
            if problem.transaction_costs is not None and bool(problem.transaction_costs.any()) else []
        ),
        "seed": payload.get("seed"),
        "n_periods": payload.get("n_periods", 504) if mode == "synthetic" else None,
        "asset_names": list(problem.universe.asset_names),
    }
