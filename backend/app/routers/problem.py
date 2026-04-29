"""Problem-definition endpoint (Task 1.2).

Builds a portfolio problem in two modes:

* **synthetic**: seeded log-return generator → sample mean μ and covariance Σ.
* **csv**: parse uploaded daily prices → log returns → μ, Σ.

Returns the QUBO and Ising forms produced by `portfolio.formulation` — the
deck-faithful version (slides 4–8): K-scaling on variance and return,
cardinality penalty, optional sector caps, risk-threshold penalty as a
diagonal-variance proxy, and linear transaction costs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from portfolio.formulation import build_qubo, qubo_to_ising

from ..services.problem_builder import build_portfolio_problem


router = APIRouter(prefix="", tags=["problem"])


class ProblemRequest(BaseModel):
    """Inputs for /problem (deck-faithful naming).

    * ``N``               — number of assets (synthetic mode only; ignored if CSV given).
    * ``lambda``          — λ₂ in the deck: weight on (1/K²)·xᵀΣx.
    * ``lambda_return``   — λ₁ in the deck: weight on (1/K)·μᵀx. Default 1.0.
    * ``K``               — target cardinality.
    * ``P_K``             — cardinality penalty (Σx_i − K)².
    * ``P_S``             — sector-cap penalty multiplier; off (0.0) by default.
    * ``sector_caps``     — {sector_index: L_s} mapping, applied with P_S>0.
    * ``P_R``             — risk-threshold penalty (deck term ⑤).
    * ``theta_risk``      — θ in the deck: target diagonal-variance proxy.
    * ``transaction_costs`` — optional length-N vector of per-asset linear costs.
    * ``seed``            — RNG seed for the synthetic log-return generator.
    * ``csv_data``        — optional CSV string of daily prices.
    """

    N: int = Field(default=8, ge=2, le=24)
    lambda_: float = Field(default=2.5, ge=0.0, alias="lambda")
    lambda_return: float = Field(default=1.0, ge=0.0)
    K: int = Field(default=3, ge=1)
    P_K: float = Field(default=5.0, ge=0.0)
    P_S: float = Field(default=0.0, ge=0.0)
    sector_caps: Optional[Dict[int, int]] = None
    P_R: float = Field(default=0.5, ge=0.0)
    theta_risk: float = Field(default=0.04, ge=0.0)
    transaction_costs: Optional[List[float]] = None
    seed: Optional[int] = 7
    n_periods: int = Field(default=504, ge=30, description="trading days in synthetic series")
    csv_data: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class ProblemResponse(BaseModel):
    mode: str
    N: int
    K: int
    asset_names: list[str]
    sectors: list[int]
    mu: list[float]
    Sigma: list[list[float]]
    qubo_Q: list[list[float]]
    qubo_offset: float
    ising_J: list[list[float]]
    ising_h: list[float]
    ising_offset: float
    weights: dict


@router.post("/problem", response_model=ProblemResponse)
def build_problem(req: ProblemRequest) -> ProblemResponse:
    payload = req.model_dump(by_alias=True)
    try:
        problem, mode = build_portfolio_problem(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    Q, offset = build_qubo(problem)
    ising = qubo_to_ising(Q, offset)

    return ProblemResponse(
        mode=mode,
        N=problem.universe.n_assets,
        K=problem.K_target,
        asset_names=list(problem.universe.asset_names),
        sectors=problem.universe.sectors.tolist(),
        mu=problem.universe.mu.tolist(),
        Sigma=problem.universe.sigma.tolist(),
        qubo_Q=Q.tolist(),
        qubo_offset=float(offset),
        ising_J=ising.J.tolist(),
        ising_h=ising.h.tolist(),
        ising_offset=float(ising.offset),
        weights=dict(
            lambda_=problem.weights.lam_variance,
            lambda_return=problem.weights.lam_return,
            P_K=problem.weights.P_K,
            P_R=problem.weights.P_R,
            P_S=problem.weights.P_S,
            theta_risk=problem.weights.theta_risk,
            sector_caps=dict(problem.sector_caps) if problem.sector_caps else {},
            transaction_costs=problem.transaction_costs.tolist()
                if problem.transaction_costs is not None else [],
        ),
    )
