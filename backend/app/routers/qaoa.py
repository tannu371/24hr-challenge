"""Phase 3 — QAOA endpoints (Aer simulator).

* POST /qaoa/run         — multistart hybrid optimisation, returns full result.
* GET  /qaoa/run/stream  — same call, but streams per-iteration progress over SSE.
* POST /qaoa/landscape   — 2D grid of ⟨H_C⟩(γ, β) for p=1.

The streamed and one-shot endpoints share the same params; the streamed one
takes its parameters as URL query/body so it can be opened with EventSource.
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from portfolio.classical import brute_force
from portfolio.formulation import build_qubo

from ..services.problem_builder import build_portfolio_problem, canonical_params
from ..services.qaoa import landscape_p1, run_qaoa_optimisation
from ..services.trials_store import TrialsStore


router = APIRouter(prefix="/qaoa", tags=["qaoa"])


# ---------------------------------------------------------------------------
# Shared payload
# ---------------------------------------------------------------------------


class QAOAFields(BaseModel):
    # Problem (deck-faithful)
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

    # QAOA
    p: int = Field(default=2, ge=1, le=8)
    mixer: str = Field(default="x", pattern="^(x|xy_ring)$")
    init_state: str = Field(default="uniform", pattern="^(uniform|dicke)$")
    optimizer: str = Field(default="COBYLA", pattern="^(COBYLA|SPSA|L-BFGS-B)$")
    max_iter: int = Field(default=200, ge=1, le=2000)
    n_restarts: int = Field(default=5, ge=1, le=50)
    qaoa_seed: int = Field(default=0)
    compute_classical_optimum: bool = Field(default=True)
    n_top_bitstrings: int = Field(default=10, ge=1, le=4096,
        description="How many top-probability bitstrings to surface. Bump this for small-N off-shell leakage diagnostics.")

    model_config = ConfigDict(populate_by_name=True)


class LandscapeRequest(BaseModel):
    N: int = Field(default=6, ge=2, le=14)
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
    mixer: str = Field(default="x", pattern="^(x|xy_ring)$")
    init_state: str = Field(default="uniform", pattern="^(uniform|dicke)$")
    n_gamma: int = Field(default=41, ge=5, le=121)
    n_beta: int = Field(default=41, ge=5, le=121)
    gamma_max: float = Field(default=math.pi, gt=0.0)
    beta_max: float = Field(default=math.pi, gt=0.0)

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Helpers shared by /qaoa/run (one-shot) and /qaoa/run/stream (SSE)
# ---------------------------------------------------------------------------


def _classical_optimum_if_small(problem) -> Optional[float]:
    """Brute-force the optimum for plotting approximation ratio. Skipped above N=18."""
    n = problem.universe.n_assets
    K = problem.K_target
    if n > 18:
        return None
    if math.comb(n, K) > 200_000:
        return None
    res = brute_force(problem)
    return float(res.cost)


def _result_payload(req: QAOAFields, problem, mode: str, qres) -> dict:
    selected = (
        [int(i) for i in np.where(qres.selected_x == 1)[0]]
        if qres.selected_x is not None else []
    )
    selected_names = [problem.universe.asset_names[i] for i in selected]
    return {
        "kind": "qaoa_sim",
        "p": req.p,
        "mixer": req.mixer,
        "init_state": req.init_state,
        "optimizer": req.optimizer,
        "max_iter": req.max_iter,
        "n_restarts": req.n_restarts,
        "energy_star": qres.energy_star,
        "theta_star": qres.theta_star.tolist(),
        "best_restart": qres.best_restart,
        "n_evaluations": qres.n_evaluations,
        "wall_time_s": qres.wall_time_s,
        "selected": selected,
        "selected_names": selected_names,
        "K": int(qres.selected_x.sum()) if qres.selected_x is not None else 0,
        "cost": qres.selected_cost,
        "top_bitstrings": qres.top_bitstrings,
        "approx_ratio": qres.approx_ratio,
        "classical_optimum": qres.classical_optimum,
        "history_per_restart": qres.history_per_restart,
    }


# ---------------------------------------------------------------------------
# 3.4 — POST /qaoa/run (one-shot, no streaming)
# ---------------------------------------------------------------------------


@router.post("/run")
def qaoa_run(req: QAOAFields) -> dict:
    payload = req.model_dump(by_alias=True)
    problem, mode = build_portfolio_problem(payload)

    classical_opt = _classical_optimum_if_small(problem) if req.compute_classical_optimum else None
    qres = run_qaoa_optimisation(
        problem=problem,
        p=req.p,
        mixer=req.mixer,
        init_state=req.init_state,
        optimizer=req.optimizer,
        max_iter=req.max_iter,
        n_restarts=req.n_restarts,
        seed=req.qaoa_seed,
        classical_optimum=classical_opt,
        n_top_bitstrings=req.n_top_bitstrings,
    )

    body = _result_payload(req, problem, mode, qres)

    trial_id = TrialsStore().record(
        kind="qaoa_sim",
        params=canonical_params(payload, mode, problem) | {
            "p": req.p, "mixer": req.mixer, "init_state": req.init_state,
            "optimizer": req.optimizer, "max_iter": req.max_iter,
            "n_restarts": req.n_restarts, "qaoa_seed": req.qaoa_seed,
        },
        results=body,
    )
    return {"trial_id": trial_id, **body}


# ---------------------------------------------------------------------------
# 3.4 — SSE streaming variant
# ---------------------------------------------------------------------------


@router.post("/run/stream")
async def qaoa_run_stream(req: QAOAFields):
    """Same params as /qaoa/run; streams per-iteration progress events.

    Event schema:
        event: tick   data: {"restart", "iter", "energy", "gamma", "beta"}
        event: done   data: <full /qaoa/run response>
        event: error  data: {"message"}
    """
    payload = req.model_dump(by_alias=True)
    problem, mode = build_portfolio_problem(payload)
    classical_opt = _classical_optimum_if_small(problem) if req.compute_classical_optimum else None

    q: queue.Queue = queue.Queue()
    SENTINEL = object()

    def _runner():
        try:
            qres = run_qaoa_optimisation(
                problem=problem,
                p=req.p,
                mixer=req.mixer,
                init_state=req.init_state,
                optimizer=req.optimizer,
                max_iter=req.max_iter,
                n_restarts=req.n_restarts,
                seed=req.qaoa_seed,
                classical_optimum=classical_opt,
                n_top_bitstrings=req.n_top_bitstrings,
                on_iter=lambda evt: q.put(("tick", evt)),
            )
            body = _result_payload(req, problem, mode, qres)
            trial_id = TrialsStore().record(
                kind="qaoa_sim",
                params=canonical_params(payload, mode, problem) | {
                    "p": req.p, "mixer": req.mixer, "init_state": req.init_state,
                    "optimizer": req.optimizer, "max_iter": req.max_iter,
                    "n_restarts": req.n_restarts, "qaoa_seed": req.qaoa_seed,
                    "stream": True,
                },
                results=body,
            )
            q.put(("done", {"trial_id": trial_id, **body}))
        except Exception as exc:  # pragma: no cover — surfaced as an SSE event
            q.put(("error", {"message": repr(exc)}))
        finally:
            q.put((SENTINEL, None))

    threading.Thread(target=_runner, daemon=True).start()

    async def event_stream():
        while True:
            evt = await asyncio.to_thread(q.get)
            kind, data = evt
            if kind is SENTINEL:
                break
            yield {"event": kind, "data": json.dumps(data)}

    return EventSourceResponse(event_stream())


# ---------------------------------------------------------------------------
# 3.5 — POST /qaoa/landscape
# ---------------------------------------------------------------------------


@router.post("/landscape")
def qaoa_landscape(req: LandscapeRequest) -> dict:
    payload = req.model_dump(by_alias=True)
    problem, _mode = build_portfolio_problem(payload)
    grid = landscape_p1(
        problem,
        mixer=req.mixer,
        init_state=req.init_state,
        n_gamma=req.n_gamma,
        n_beta=req.n_beta,
        gamma_max=req.gamma_max,
        beta_max=req.beta_max,
    )
    return {
        "N": problem.universe.n_assets,
        "K": problem.K_target,
        "mixer": req.mixer,
        "init_state": req.init_state,
        **grid,
    }
