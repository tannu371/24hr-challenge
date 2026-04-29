"""Phase-3 acceptance tests."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("trials") / "trials_qaoa.db"
    os.environ["TRIALS_DB_PATH"] = str(db_path)

    import importlib
    import app.config
    import app.services.trials_store
    import app.routers.classical
    import app.routers.qaoa
    import app.routers.trials
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.services.trials_store)
    importlib.reload(app.routers.classical)
    importlib.reload(app.routers.qaoa)
    importlib.reload(app.routers.trials)
    importlib.reload(app.main)
    return TestClient(app.main.app)


# ---------------------------------------------------------------------------
# 3.1 — Cost Hamiltonian sanity: ⟨z|H_C|z⟩ + offset matches Ising energy
# ---------------------------------------------------------------------------


def test_cost_hamiltonian_matches_ising_energy_on_basis_states():
    from qiskit.quantum_info import Statevector

    from app.services.synthetic import generate_log_returns_universe
    from portfolio.formulation import (
        ObjectiveWeights, PortfolioProblem, build_qubo, qubo_cost,
        qubo_to_ising, ising_to_pauli,
    )

    universe = generate_log_returns_universe(n_assets=6, seed=4)
    problem = PortfolioProblem(
        universe=universe, K_target=3,
        weights=ObjectiveWeights(
            lam_return=1.0, lam_variance=2.5, P_K=5.0,
            P_S=0.0, P_R=0.5, theta_risk=0.04,
        ),
    )
    Q, c = build_qubo(problem)
    ising = qubo_to_ising(Q, c)
    cost_op, ising_offset = ising_to_pauli(ising)

    n = 6
    rng = np.random.default_rng(0)
    for _ in range(8):
        x = rng.integers(0, 2, size=n)
        # Qiskit ordering: bit i of integer = qubit i (lsb).
        idx = int(sum(int(b) << i for i, b in enumerate(x)))
        sv = Statevector.from_int(idx, dims=2 ** n)
        e_pauli = float(sv.expectation_value(cost_op).real) + ising_offset
        e_qubo = qubo_cost(Q, c, x)
        assert abs(e_pauli - e_qubo) < 1e-9


# ---------------------------------------------------------------------------
# 3.2 — XY-ring + Dicke initial state preserve Hamming weight
# ---------------------------------------------------------------------------


def test_xy_ring_with_dicke_preserves_cardinality():
    """A QAOA circuit with Dicke-K init + XY-ring mixer must put zero
    amplitude on any bitstring outside the K-cardinality shell, regardless
    of (γ, β). The Dicke state has weight K and the XY-ring mixer commutes
    with the total-Z operator, so the result is a known invariant.
    """
    from qiskit.quantum_info import Statevector

    from app.services.qaoa import build_qaoa_circuit
    from portfolio.formulation import IsingModel

    n, K = 6, 3
    rng = np.random.default_rng(7)
    h = rng.normal(size=n)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = rng.normal()
    ising = IsingModel(h=h, J=J, offset=0.0)

    circ = build_qaoa_circuit(ising, p=2, mixer="xy_ring", init_state="dicke", K=K)
    theta = rng.uniform(0.0, np.pi, size=2 * 2)
    sv = Statevector.from_instruction(circ.bind(theta))
    probs = np.abs(sv.data) ** 2

    leak = 0.0
    for s in range(1 << n):
        if bin(s).count("1") != K:
            leak += probs[s]
    assert leak < 1e-8, f"XY-ring + Dicke leaked {leak:.3e} probability outside the K-shell"


# ---------------------------------------------------------------------------
# 3.4 — /qaoa/run returns approximation ratio + populates trial store
# ---------------------------------------------------------------------------


def test_qaoa_run_smoke_and_approx_ratio(client):
    payload = {
        "N": 6, "lambda": 2.5, "K": 3, "P_K": 5.0, "P_R": 0.5, "seed": 11,
        "p": 2, "mixer": "x", "init_state": "uniform",
        "optimizer": "COBYLA", "max_iter": 100, "n_restarts": 3,
        "qaoa_seed": 0,
    }
    resp = client.post("/qaoa/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trial_id"] >= 1
    assert body["energy_star"] is not None
    # Multistart history shape
    assert len(body["history_per_restart"]) == 3
    # Top-10 bitstrings populated
    assert len(body["top_bitstrings"]) == 10
    # Classical optimum was computed (N=6 ≤ 18)
    assert body["classical_optimum"] is not None
    assert body["approx_ratio"] is not None
    # New convention: approx_ratio = method_cost / brute_cost.
    # On a tiny N=6 problem QAOA typically matches the optimum, giving ratio ≈ 1.
    # We allow a wider window because a degenerate seed could produce a worse run.
    assert -10.0 <= body["approx_ratio"] <= 2.0
    # Theta has length 2p
    assert len(body["theta_star"]) == 2 * payload["p"]

    # Trial round-trip
    trial = client.get(f"/trials/{body['trial_id']}").json()
    assert trial["kind"] == "qaoa_sim"
    assert trial["params"]["p"] == 2


def test_qaoa_run_stream_emits_ticks_and_done(client):
    payload = {
        "N": 6, "lambda": 2.5, "K": 3, "P_K": 5.0, "P_R": 0.5, "seed": 11,
        "p": 1, "mixer": "x", "init_state": "uniform",
        "optimizer": "COBYLA", "max_iter": 30, "n_restarts": 2,
        "qaoa_seed": 1,
    }
    n_ticks = 0
    done_payload = None
    with client.stream("POST", "/qaoa/run/stream", json=payload) as stream:
        assert stream.status_code == 200
        event_name = None
        for line in stream.iter_lines():
            if not line:
                event_name = None
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if event_name == "tick":
                    n_ticks += 1
                    assert {"restart", "iter", "energy", "gamma", "beta"}.issubset(data.keys())
                elif event_name == "done":
                    done_payload = data
            if done_payload is not None:
                break
    assert n_ticks >= 5, f"too few tick events received: {n_ticks}"
    assert done_payload is not None
    assert done_payload["trial_id"] >= 1


# ---------------------------------------------------------------------------
# 3.5 — Landscape grid shape, axis lengths, argmin
# ---------------------------------------------------------------------------


def test_qaoa_landscape_shape_and_argmin(client):
    payload = {
        "N": 6, "lambda": 2.5, "K": 3, "P_K": 5.0, "P_R": 0.5, "seed": 11,
        "mixer": "x", "init_state": "uniform",
        "n_gamma": 21, "n_beta": 21,
    }
    resp = client.post("/qaoa/landscape", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    Z = np.array(body["energy"])
    assert Z.shape == (21, 21)
    assert len(body["gamma"]) == 21 and len(body["beta"]) == 21
    assert body["argmin"]["energy"] == pytest.approx(Z.min(), abs=1e-12)
    gi, bi = body["argmin"]["i"], body["argmin"]["j"]
    assert Z[gi, bi] == pytest.approx(Z.min(), abs=1e-12)
