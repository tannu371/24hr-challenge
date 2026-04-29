"""Verify POST /problem matches an independent hand-rolled computation
for the deterministic synthetic case N=4, seed=42.

We don't trust the endpoint's internal call into portfolio.formulation —
we re-derive μ, Σ, Q, and the QUBO cost on every binary x in {0,1}^4 from
first principles inside this test, then compare.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.synthetic import generate_log_returns_universe


N = 4
SEED = 42
LAM_VAR = 2.5      # lambda_2 — variance weight
LAM_RET = 1.0      # lambda_1 — return weight (API default)
K_TARGET = 2
P_K = 5.0          # cardinality penalty
P_R = 0.5          # risk-threshold penalty (deck term ⑤, diag-variance proxy)
THETA_RISK = 0.04  # θ in deck term ⑤


def _hand_qubo(mu: np.ndarray, sigma: np.ndarray):
    """Build Q and offset for the deck-faithful formulation, independently
    of portfolio.formulation."""
    n = len(mu)
    Q = np.zeros((n, n))
    c = 0.0
    inv_K = 1.0 / K_TARGET
    inv_K2 = inv_K * inv_K

    # ① variance scaled
    Q += LAM_VAR * inv_K2 * sigma
    # ② return scaled
    for i in range(n):
        Q[i, i] += -LAM_RET * inv_K * mu[i]
    # ③ cardinality
    for i in range(n):
        Q[i, i] += P_K * (1.0 - 2.0 * K_TARGET)
        for j in range(i + 1, n):
            Q[i, j] += P_K
            Q[j, i] += P_K
    c += P_K * (K_TARGET ** 2)
    # ⑤ risk-threshold proxy
    a = np.diag(sigma) * inv_K2
    for i in range(n):
        Q[i, i] += P_R * (a[i] * a[i] - 2.0 * THETA_RISK * a[i])
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += P_R * a[i] * a[j]
            Q[j, i] += P_R * a[i] * a[j]
    c += P_R * (THETA_RISK ** 2)
    return Q, c


def _hand_cost(mu, sigma, x: np.ndarray) -> float:
    """Direct deck H(x) for binary x. Sector caps + transaction costs disabled."""
    inv_K = 1.0 / K_TARGET
    inv_K2 = inv_K * inv_K
    cost = LAM_VAR * inv_K2 * float(x @ sigma @ x)
    cost += -LAM_RET * inv_K * float(mu @ x)
    cost += P_K * (int(x.sum()) - K_TARGET) ** 2
    a = np.diag(sigma) * inv_K2
    V_hat = float(a @ x)
    cost += P_R * (V_hat - THETA_RISK) ** 2
    return cost


def test_problem_n4_seed42_matches_hand_computation():
    client = TestClient(app)

    payload = dict(
        N=N, K=K_TARGET, P_K=P_K, P_R=P_R, seed=SEED,
        theta_risk=THETA_RISK, lambda_return=LAM_RET,
    )
    payload["lambda"] = LAM_VAR
    resp = client.post("/problem", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # μ, Σ — must match an independent re-run of the seeded generator exactly.
    universe = generate_log_returns_universe(n_assets=N, seed=SEED)
    np.testing.assert_allclose(body["mu"], universe.mu, rtol=0, atol=1e-12)
    np.testing.assert_allclose(body["Sigma"], universe.sigma, rtol=0, atol=1e-12)

    # Hand-built QUBO must match the API's QUBO matrix and offset.
    Q_hand, c_hand = _hand_qubo(universe.mu, universe.sigma)
    np.testing.assert_allclose(body["qubo_Q"], Q_hand, rtol=0, atol=1e-12)
    assert abs(body["qubo_offset"] - c_hand) < 1e-12

    # Cost equality on every binary x — strongest possible check at N=4.
    Q_api = np.array(body["qubo_Q"])
    off_api = body["qubo_offset"]
    for bits in product((0, 1), repeat=N):
        x = np.array(bits, dtype=int)
        f_hand = _hand_cost(universe.mu, universe.sigma, x)
        f_qubo = float(x @ Q_api @ x + off_api)
        assert abs(f_hand - f_qubo) < 1e-9, (bits, f_hand, f_qubo)


def test_problem_csv_mode_round_trip():
    """Hand-build a known geometric price series → check μ, Σ are sensible."""
    client = TestClient(app)
    rng = np.random.default_rng(0)
    T, N_csv = 250, 4
    daily = rng.normal(0.0005, 0.01, size=(T, N_csv))
    prices = np.exp(np.cumsum(daily, axis=0)) * 100.0
    header = ",".join(f"P{i}" for i in range(N_csv))
    body = "\n".join(",".join(f"{p:.6f}" for p in row) for row in prices)
    csv_text = header + "\n" + body

    resp = client.post(
        "/problem",
        json={
            "N": N_csv, "lambda": 2.0, "K": 2, "P_K": 4.0, "P_R": 0.3,
            "csv_data": csv_text,
        },
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["mode"] == "csv"
    assert out["N"] == N_csv
    assert out["asset_names"] == [f"P{i}" for i in range(N_csv)]
    Sigma = np.array(out["Sigma"])
    assert Sigma.shape == (N_csv, N_csv)
    assert np.linalg.eigvalsh(Sigma).min() > 0.0  # PSD
