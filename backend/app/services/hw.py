"""IBM Quantum runtime adapter (Phase 4).

The token lives in `backend/.env` and is loaded by ``app.config``. Nothing in
this module — or any other backend module — ships the token to clients.

We deliberately keep `QiskitRuntimeService` import inside helper functions so
the rest of the backend boots cleanly when no IBM credentials are configured
(e.g. for local CI). Endpoints fall back to 503 with a clear message in that
case; the cached-results path (Task 4.5) keeps working regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from portfolio.formulation import (
    PortfolioProblem,
    build_qubo,
    qubo_cost,
    qubo_to_ising,
)

from ..config import SETTINGS
from .qaoa import build_qaoa_circuit


# ---------------------------------------------------------------------------
# Service handle
# ---------------------------------------------------------------------------


def credentials_configured() -> bool:
    return SETTINGS.has_ibm_credentials


def _runtime_service():
    """Lazy-init QiskitRuntimeService — token never leaves this process."""
    if not credentials_configured():
        raise RuntimeError(
            "IBM Quantum token not configured. Add IBM_QUANTUM_TOKEN to backend/.env."
        )
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService(
        channel=SETTINGS.ibm_quantum_channel,
        token=SETTINGS.ibm_quantum_token,
        instance=SETTINGS.ibm_quantum_instance,
    )


# ---------------------------------------------------------------------------
# 4.1 — backend listing
# ---------------------------------------------------------------------------


@dataclass
class BackendInfo:
    name: str
    qubits: int
    queue_length: int
    status: str
    operational: bool
    simulator: bool


def list_backends() -> list[BackendInfo]:
    svc = _runtime_service()
    out: list[BackendInfo] = []
    for b in svc.backends():
        try:
            status = b.status()
            queue = int(getattr(status, "pending_jobs", 0) or 0)
            status_msg = getattr(status, "status_msg", "unknown")
            operational = bool(getattr(status, "operational", False))
        except Exception:
            queue, status_msg, operational = 0, "unavailable", False
        out.append(BackendInfo(
            name=b.name,
            qubits=int(b.num_qubits),
            queue_length=queue,
            status=str(status_msg),
            operational=operational,
            simulator=bool(getattr(b, "simulator", False)),
        ))
    out.sort(key=lambda x: (not x.operational, x.queue_length, x.name))
    return out


# ---------------------------------------------------------------------------
# 4.2 — circuit assembly + submission
# ---------------------------------------------------------------------------


def assemble_measured_circuit(problem: PortfolioProblem, p: int, mixer: str,
                              init_state: str, theta: list[float] | np.ndarray):
    """Build the QAOA circuit, bind θ, append measurements over the full register."""
    from qiskit import ClassicalRegister, QuantumCircuit

    Q, c = build_qubo(problem)
    ising = qubo_to_ising(Q, c)
    K = problem.K_target if init_state == "dicke" else None
    circuit = build_qaoa_circuit(ising, p=p, mixer=mixer, init_state=init_state, K=K)
    bound = circuit.bind(np.asarray(theta, dtype=float))

    n = bound.num_qubits
    creg = ClassicalRegister(n, name="meas")
    measured = QuantumCircuit(*bound.qregs, creg)
    measured.compose(bound, inplace=True)
    measured.barrier()
    measured.measure(range(n), range(n))
    return measured


def submit_qaoa_job(
    problem: PortfolioProblem,
    p: int,
    mixer: str,
    init_state: str,
    theta: list[float],
    backend_name: str,
    shots: int,
    error_mitigation: Optional[dict[str, Any]] = None,
) -> dict:
    """Transpile + submit via Sampler V2 primitive. Returns {job_id, backend_name}."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    svc = _runtime_service()
    backend = svc.backend(backend_name)

    circ = assemble_measured_circuit(problem, p, mixer, init_state, theta)
    transpiled = transpile(circ, backend=backend, optimization_level=3)

    options: dict[str, Any] = {"default_shots": int(shots)}
    if error_mitigation:
        # Sampler V2 routes mitigation under runtime options.
        run_opts = options.setdefault("dynamical_decoupling", {})
        if error_mitigation.get("dynamical_decoupling"):
            run_opts["enable"] = True
        if error_mitigation.get("readout"):
            options.setdefault("twirling", {})["enable_measure"] = True

    sampler = Sampler(backend, options=options)
    job = sampler.run([transpiled])
    return {"job_id": job.job_id(), "backend_name": backend_name}


# ---------------------------------------------------------------------------
# 4.3 / 4.4 — job polling + result ingestion
# ---------------------------------------------------------------------------


def poll_job(job_id: str) -> dict[str, Any]:
    """Status + queue position + (when DONE) raw result for ingestion."""
    svc = _runtime_service()
    job = svc.job(job_id)
    status = job.status()
    status_str = status.name if hasattr(status, "name") else str(status)

    out: dict[str, Any] = {
        "job_id": job_id,
        "status": status_str,
        "backend": job.backend().name if job.backend() else None,
        "queue_position": None,
        "est_start": None,
    }
    try:
        out["queue_position"] = job.queue_position(refresh=True)
    except Exception:
        pass
    try:
        out["est_start"] = str(job.queue_info().estimated_start_time) if job.queue_info() else None
    except Exception:
        pass

    if status_str.upper() in {"DONE", "COMPLETED"}:
        try:
            res = job.result()
            counts = _extract_counts(res)
            out["counts"] = counts
        except Exception as exc:
            out["status"] = "ERROR"
            out["error"] = repr(exc)
    return out


def _extract_counts(result_obj) -> dict[str, int]:
    """SamplerV2 result → {bitstring: count}, big-endian style as Qiskit prints."""
    pub = result_obj[0] if hasattr(result_obj, "__getitem__") else result_obj
    data = getattr(pub, "data", pub)
    bits = getattr(data, "meas", None) or getattr(data, "c", None) or list(data.values())[0]
    counts = bits.get_counts()
    return {str(k): int(v) for k, v in counts.items()}


def ingest_counts(
    problem: PortfolioProblem,
    counts: dict[str, int],
    p: int,
    mixer: str,
    init_state: str,
    theta: list[float],
    classical_optimum: Optional[float] = None,
) -> dict[str, Any]:
    """Convert raw bitstring counts into the same result schema as /qaoa/run."""
    Q, c = build_qubo(problem)
    n = problem.universe.n_assets
    total = sum(counts.values()) or 1

    energy = 0.0
    items = []
    for bitstr, cnt in counts.items():
        # Qiskit's bitstring is big-endian (qubit n-1 first). Reverse to lsb.
        b = bitstr.replace(" ", "")
        x = np.array([int(b[-(i + 1)]) for i in range(n)], dtype=int)
        cost = float(qubo_cost(Q, c, x))
        prob = cnt / total
        energy += prob * cost
        items.append({
            "bitstring": bitstr,
            "x": x.tolist(),
            "count": int(cnt),
            "probability": prob,
            "cost": cost,
            "K": int(x.sum()),
        })

    items.sort(key=lambda d: -d["probability"])
    top = items[:10]

    # Best-by-cost across the empirical distribution (the same projection as
    # in the simulator path).
    best = min(items, key=lambda d: d["cost"])
    selected_x = np.array(best["x"], dtype=int)

    approx_ratio = None
    if classical_optimum is not None and abs(classical_optimum) > 1e-15:
        # Literal ratio: method_cost / brute_cost. See services/qaoa.py for sign conventions.
        approx_ratio = float(best["cost"]) / float(classical_optimum)

    return {
        "kind": "qaoa_hw",
        "status": "complete",
        "energy": float(energy),
        "cost": float(best["cost"]),
        "selected": [int(i) for i in np.where(selected_x == 1)[0]],
        "selected_names": [problem.universe.asset_names[int(i)] for i in np.where(selected_x == 1)[0]],
        "K": int(selected_x.sum()),
        "top_bitstrings": top,
        "approx_ratio": approx_ratio,
        "classical_optimum": classical_optimum,
        "n_unique_bitstrings": len(items),
        "shots": total,
        "p": p,
        "mixer": mixer,
        "init_state": init_state,
        "theta_star": list(theta),
    }
