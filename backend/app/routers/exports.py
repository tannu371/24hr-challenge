"""Phase 5 — export endpoints.

* GET /export/qasm/{trial_id}                 — QASM 3 (text/plain)
* GET /export/qiskit/{trial_id}               — standalone Qiskit .py
* GET /export/pennylane/{trial_id}            — standalone PennyLane .py
* GET /export/circuit/{trial_id}.{fmt}        — circuit diagram (svg|png), transpiled=true|false
* GET /export/plot/{trial_id}/{kind}.{fmt}    — plot (kind ∈ trajectory/histogram/landscape/comparison)
* GET /export/bundle/{trial_id}               — zip with everything
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ..services import exports


router = APIRouter(prefix="/export", tags=["export"])


@router.get("/qasm/{trial_id}")
def export_qasm(trial_id: int) -> Response:
    try:
        text = exports.export_qasm3(trial_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, KeyError) else 400, detail=str(exc))
    return Response(
        text, media_type="application/x.qasm",
        headers={"Content-Disposition": f'attachment; filename="trial_{trial_id}.qasm"'},
    )


@router.get("/qiskit/{trial_id}")
def export_qiskit(trial_id: int) -> Response:
    try:
        text = exports.export_qiskit_script(trial_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, KeyError) else 400, detail=str(exc))
    return Response(
        text, media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="qaoa_trial_{trial_id}.py"'},
    )


@router.get("/pennylane/{trial_id}")
def export_pennylane(trial_id: int) -> Response:
    try:
        text = exports.export_pennylane_script(trial_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, KeyError) else 400, detail=str(exc))
    return Response(
        text, media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="qaoa_trial_{trial_id}_pennylane.py"'},
    )


@router.get("/circuit/{trial_id}.{fmt}")
def export_circuit(trial_id: int, fmt: str, transpiled: bool = Query(default=False)) -> Response:
    if fmt not in {"svg", "png"}:
        raise HTTPException(status_code=400, detail="fmt must be svg or png")
    try:
        data = exports.render_circuit(trial_id, fmt=fmt, transpiled=transpiled)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, KeyError) else 400, detail=str(exc))
    media = "image/svg+xml" if fmt == "svg" else "image/png"
    suffix = "_transpiled" if transpiled else ""
    return Response(
        data, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="circuit_{trial_id}{suffix}.{fmt}"'},
    )


@router.get("/plot/{trial_id}/{kind}.{fmt}")
def export_plot(trial_id: int, kind: str, fmt: str) -> Response:
    if fmt not in {"svg", "png", "csv"}:
        raise HTTPException(status_code=400, detail="fmt must be svg|png|csv")
    if kind not in {"trajectory", "histogram", "landscape", "comparison"}:
        raise HTTPException(status_code=400, detail="kind must be trajectory|histogram|landscape|comparison")
    try:
        data = exports.render_plot(trial_id, kind=kind, fmt=fmt)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, KeyError) else 400, detail=str(exc))
    if fmt == "csv":
        return Response(
            data, media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{kind}_{trial_id}.csv"'},
        )
    media = "image/svg+xml" if fmt == "svg" else "image/png"
    return Response(
        data, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{kind}_{trial_id}.{fmt}"'},
    )


@router.get("/bundle/{trial_id}")
def export_bundle(trial_id: int) -> Response:
    try:
        data = exports.export_bundle(trial_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(
        data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="trial_{trial_id}_bundle.zip"'},
    )
