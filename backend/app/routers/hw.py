"""Phase 4 — IBM Quantum hardware endpoints.

* GET  /hw/backends         — live IBM backend listing (4.1)
* POST /hw/submit            — transpile + submit a stored qaoa_sim trial (4.2)
* GET  /hw/job/{job_id}      — poll status, ingest on completion (4.3 + 4.4)
* GET  /hw/cached            — list cached snapshots from /artifacts
* GET  /hw/cached/{name}     — full cached record (4.5)
* POST /hw/cached/import     — copy a cached snapshot into the trials table

No endpoint ever returns the IBM token to the caller.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from portfolio.classical import brute_force

from ..services import hw, hw_cache
from ..services.problem_builder import build_portfolio_problem
from ..services.trials_store import TrialsStore


router = APIRouter(prefix="/hw", tags=["hardware"])


# ---------------------------------------------------------------------------
# 4.1 — backend listing
# ---------------------------------------------------------------------------


@router.get("/backends")
def list_backends() -> dict:
    if not hw.credentials_configured():
        raise HTTPException(
            status_code=503,
            detail="IBM Quantum token not configured. Add IBM_QUANTUM_TOKEN to backend/.env "
                   "and restart the server.",
        )
    try:
        backends = hw.list_backends()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IBM service error: {exc!r}")
    return {
        "backends": [b.__dict__ for b in backends],
        "instance": "configured",
    }


# ---------------------------------------------------------------------------
# 4.2 — POST /hw/submit
# ---------------------------------------------------------------------------


class HwSubmitRequest(BaseModel):
    trial_id: int = Field(..., description="Source qaoa_sim trial whose θ⋆ + circuit will be reused.")
    backend_name: str
    shots: int = Field(default=4096, ge=64, le=200000)
    error_mitigation: dict = Field(default_factory=dict)


@router.post("/submit")
def hw_submit(req: HwSubmitRequest) -> dict:
    if not hw.credentials_configured():
        raise HTTPException(
            status_code=503,
            detail="IBM Quantum token not configured. Add IBM_QUANTUM_TOKEN to backend/.env.",
        )
    store = TrialsStore()
    src = store.get(req.trial_id)
    if src is None:
        raise HTTPException(status_code=404, detail=f"trial {req.trial_id} not found")
    if src["kind"] != "qaoa_sim":
        raise HTTPException(
            status_code=400,
            detail=f"can only submit hardware jobs from qaoa_sim trials (got {src['kind']!r})",
        )

    params = src["params"]
    results = src["results"]
    theta = results.get("theta_star")
    p = results.get("p") or params.get("p")
    mixer = results.get("mixer") or params.get("mixer", "x")
    init_state = results.get("init_state") or params.get("init_state", "uniform")
    if theta is None or p is None:
        raise HTTPException(status_code=400, detail="source trial is missing θ⋆ or p")

    problem, _mode = build_portfolio_problem(params)

    try:
        out = hw.submit_qaoa_job(
            problem=problem, p=int(p), mixer=mixer, init_state=init_state,
            theta=list(theta), backend_name=req.backend_name,
            shots=req.shots, error_mitigation=req.error_mitigation,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"submission failed: {exc!r}")

    classical_opt = None
    try:
        if problem.universe.n_assets <= 18:
            classical_opt = float(brute_force(problem).cost)
    except Exception:
        pass

    new_trial_id = store.record(
        kind="qaoa_hw",
        params={**params, "p": int(p), "mixer": mixer, "init_state": init_state,
                "shots": req.shots, "error_mitigation": req.error_mitigation,
                "source_trial_id": req.trial_id, "backend": req.backend_name},
        results={
            "status": "queued",
            "job_id": out["job_id"],
            "backend": out["backend_name"],
            "shots": req.shots,
            "p": int(p),
            "mixer": mixer,
            "init_state": init_state,
            "theta_star": list(theta),
            "classical_optimum": classical_opt,
            "source_trial_id": req.trial_id,
        },
    )
    return {"trial_id": new_trial_id, "job_id": out["job_id"], "backend": out["backend_name"]}


# ---------------------------------------------------------------------------
# 4.3 + 4.4 — GET /hw/job/{job_id}
# ---------------------------------------------------------------------------


@router.get("/job/{job_id}")
def hw_job(job_id: str) -> dict:
    if not hw.credentials_configured():
        raise HTTPException(
            status_code=503,
            detail="IBM Quantum token not configured. Add IBM_QUANTUM_TOKEN to backend/.env.",
        )
    store = TrialsStore()
    trial = store.find_one(kind="qaoa_hw", where_results={"job_id": job_id})

    try:
        info = hw.poll_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"poll failed: {exc!r}")

    response = {
        "job_id": job_id,
        "status": info["status"],
        "backend": info.get("backend"),
        "queue_position": info.get("queue_position"),
        "est_start": info.get("est_start"),
        "trial_id": trial["id"] if trial else None,
    }

    if "counts" in info and trial is not None:
        # Ingest into the linked trial — Task 4.4.
        params = trial["params"]
        problem, _mode = build_portfolio_problem(params)
        prev = trial["results"]
        ingested = hw.ingest_counts(
            problem=problem,
            counts=info["counts"],
            p=int(prev["p"]),
            mixer=prev["mixer"],
            init_state=prev["init_state"],
            theta=prev["theta_star"],
            classical_optimum=prev.get("classical_optimum"),
        )
        merged = {**prev, **ingested, "status": "complete", "job_id": job_id,
                  "backend": info.get("backend") or prev.get("backend")}
        store.update_results(trial["id"], merged)
        response["results"] = merged
    elif trial is not None:
        response["results"] = trial["results"]

    return response


# ---------------------------------------------------------------------------
# 4.5 — cached results
# ---------------------------------------------------------------------------


@router.get("/cached")
def cached_index() -> dict:
    return {"cached": hw_cache.list_cached()}


@router.get("/cached/{name}")
def cached_get(name: str) -> dict:
    blob = hw_cache.get_cached(name)
    if blob is None:
        raise HTTPException(status_code=404, detail=f"cached snapshot {name!r} not found")
    return blob


class CachedImportRequest(BaseModel):
    name: str


@router.post("/cached/import")
def cached_import(req: CachedImportRequest) -> dict:
    """Copy a cached snapshot into the trials table so it shows up in /trials."""
    blob = hw_cache.get_cached(req.name)
    if blob is None:
        raise HTTPException(status_code=404, detail=f"cached snapshot {req.name!r} not found")

    params = blob.get("params", {}) | {"cached_name": req.name, "stand_in": blob.get("meta", {}).get("stand_in", False)}
    results = dict(blob.get("results", {}))
    results.setdefault("status", "complete")
    results["from_cache"] = True
    trial_id = TrialsStore().record(kind="qaoa_hw", params=params, results=results)
    return {"trial_id": trial_id, "name": req.name}
