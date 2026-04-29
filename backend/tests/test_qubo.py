"""Round-trip QUBO ↔ Ising on a 6-asset case (Task 1.3 spec).

The conversion utilities live in `portfolio.formulation` (pure NumPy) and are
re-used by the backend without modification — testing them here makes the
round-trip an explicit contract of the backend.
"""

from __future__ import annotations

import numpy as np

from portfolio.formulation import qubo_to_ising


def _random_symmetric_qubo(n: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=(n, n))
    Q = 0.5 * (A + A.T)
    Q[np.diag_indices(n)] = rng.normal(0.0, 1.0, size=n)
    offset = float(rng.normal(0.0, 1.0))
    return Q, offset


def _qubo_cost(Q: np.ndarray, c: float, x: np.ndarray) -> float:
    return float(x @ Q @ x + c)


def test_qubo_to_ising_round_trip_six_assets():
    n = 6
    Q, c = _random_symmetric_qubo(n, seed=11)
    ising = qubo_to_ising(Q, c)

    # All 64 binary vectors must produce identical cost in QUBO and Ising form.
    for k in range(2 ** n):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
        z = 1 - 2 * x
        e_qubo = _qubo_cost(Q, c, x)
        e_ising = ising.evaluate_z(z)
        e_ising_x = ising.evaluate_x(x)
        assert abs(e_qubo - e_ising) < 1e-9, (k, e_qubo, e_ising)
        assert abs(e_qubo - e_ising_x) < 1e-9, (k, e_qubo, e_ising_x)


def test_qubo_to_ising_offset_handles_constants_only():
    """A QUBO with Q = 0 must map to h=0, J=0, offset=c."""
    n = 6
    Q = np.zeros((n, n))
    c = 3.14
    ising = qubo_to_ising(Q, c)
    np.testing.assert_array_equal(ising.h, np.zeros(n))
    np.testing.assert_array_equal(ising.J, np.zeros((n, n)))
    assert abs(ising.offset - c) < 1e-12


def test_qubo_to_ising_diagonal_only():
    """Pure linear QUBO: Q diagonal, no quadratic. J must remain zero."""
    n = 6
    rng = np.random.default_rng(3)
    Q = np.diag(rng.normal(size=n))
    c = 0.7
    ising = qubo_to_ising(Q, c)
    np.testing.assert_array_equal(ising.J, np.zeros((n, n)))
    # Verify on a few bitstrings.
    for seed in range(8):
        x = np.random.default_rng(seed).integers(0, 2, size=n)
        z = 1 - 2 * x
        assert abs(_qubo_cost(Q, c, x) - ising.evaluate_z(z)) < 1e-9
