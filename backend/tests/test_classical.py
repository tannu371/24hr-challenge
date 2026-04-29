"""Phase-2 acceptance tests.

* 2.1 — brute force matches the analytical optimum on N=8 fixed seed.
* 2.2 — SA hits the brute optimum ≥95% of the time on N=12 with 20 restarts.
* 2.3 — Markowitz frontier renders 50 distinct λ points with non-decreasing return.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("trials") / "trials_test.db"
    os.environ["TRIALS_DB_PATH"] = str(db_path)

    import importlib
    import app.config
    import app.services.trials_store
    import app.routers.classical
    import app.routers.trials
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.services.trials_store)
    importlib.reload(app.routers.classical)
    importlib.reload(app.routers.trials)
    importlib.reload(app.main)

    return TestClient(app.main.app)


# ---------------------------------------------------------------------------
# 2.1 — brute matches an independent analytical optimum on N=8 fixed seed
# ---------------------------------------------------------------------------


def test_brute_n8_matches_analytical_optimum(client):
    payload = {"N": 8, "lambda": 2.5, "K": 3, "P_K": 5.0, "P_R": 0.5, "seed": 17}
    resp = client.post("/classical/brute", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Independent re-derivation: build the universe & QUBO from scratch and
    # enumerate every C(8, 3) = 56 cardinality-K bitstring locally.
    from itertools import combinations

    from app.services.synthetic import generate_log_returns_universe
    from portfolio.formulation import (
        ObjectiveWeights, PortfolioProblem, build_qubo, qubo_cost,
    )

    universe = generate_log_returns_universe(n_assets=8, seed=17)
    problem = PortfolioProblem(
        universe=universe,
        K_target=3,
        weights=ObjectiveWeights(
            lam_return=1.0, lam_variance=2.5, P_K=5.0,
            P_S=0.0, P_R=0.5, theta_risk=0.04,
        ),
    )
    Q, c = build_qubo(problem)
    best_local, best_x = np.inf, None
    for combo in combinations(range(8), 3):
        x = np.zeros(8, dtype=int)
        x[list(combo)] = 1
        f = qubo_cost(Q, c, x)
        if f < best_local:
            best_local, best_x = f, x

    assert abs(body["cost"] - float(best_local)) < 1e-9
    assert body["best_x"] == best_x.tolist()
    assert body["K"] == 3
    assert len(body["energy_distribution"]) == 56
    assert body["energy_distribution"][0] == pytest.approx(best_local, abs=1e-9)
    assert body["trial_id"] >= 1


# ---------------------------------------------------------------------------
# 2.2 — SA hits the brute optimum ≥95% of the time on N=12, 20 restarts
# ---------------------------------------------------------------------------


def test_sa_n12_hits_brute_optimum_at_least_95pct(client):
    base = {"N": 12, "lambda": 2.5, "K": 4, "P_K": 5.0, "P_R": 0.5, "seed": 23}

    brute_resp = client.post("/classical/brute", json={**base, "distribution_max_n": 20})
    assert brute_resp.status_code == 200
    brute_cost = brute_resp.json()["cost"]

    sa_resp = client.post(
        "/classical/sa",
        json={
            **base,
            "T0": 1.0, "T_min": 1e-3, "sweeps": 300, "restarts": 20,
            "sa_seed": 0, "move": "swap", "init": "random_K",
        },
    )
    assert sa_resp.status_code == 200, sa_resp.text
    sa = sa_resp.json()
    assert sa["best_x"]
    # The "≥95%" criterion: at least 19 of 20 restarts should reach the
    # brute-force optimum (within numerical tolerance).
    runs = sa["runs"]
    n_hit = sum(1 for r in runs if abs(r["final_cost"] - brute_cost) < 1e-6)
    assert n_hit >= 19, (n_hit, [r["final_cost"] for r in runs], brute_cost)
    # Best-of-restarts must hit it exactly.
    assert abs(sa["cost"] - brute_cost) < 1e-9
    # Trajectory shape: per-sweep, length sweeps + 1.
    assert len(runs[0]["trajectory_per_sweep"]) == 301


# ---------------------------------------------------------------------------
# 2.3 — Markowitz frontier renders 50 distinct λ points
# ---------------------------------------------------------------------------


def test_markowitz_frontier_renders_50_points(client):
    payload = {
        "N": 8, "lambda": 2.5, "K": 3, "P_K": 5.0, "P_R": 0.5, "seed": 5,
        "frontier": True, "frontier_n_lambda": 50,
        "frontier_lambda_min": 0.05, "frontier_lambda_max": 20.0,
    }
    resp = client.post("/classical/markowitz", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    frontier = body["frontier"]
    assert frontier is not None
    assert len(frontier) == 50
    # All λ-values must be unique and monotonically increasing.
    lambdas = [pt["lambda"] for pt in frontier]
    assert all(b > a for a, b in zip(lambdas, lambdas[1:]))
    # Variance must be non-increasing in λ (more risk-aversion ⇒ less variance).
    variances = [pt["variance"] for pt in frontier]
    assert all(b <= a + 1e-8 for a, b in zip(variances, variances[1:])), variances
    # Return must be non-increasing in λ too (along the efficient frontier).
    returns = [pt["return"] for pt in frontier]
    assert all(b <= a + 1e-8 for a, b in zip(returns, returns[1:])), returns
    # Solver returned a sensible binary projection.
    assert body["K"] == 3
    assert body["trial_id"] >= 1
