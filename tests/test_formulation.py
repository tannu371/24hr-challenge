"""Verify the algebraic equivalences underpinning the QUBO/Ising mapping.

Run with: `python -m tests.test_formulation`  (or pytest if installed).
"""

from __future__ import annotations

import numpy as np

from portfolio.data import make_universe
from portfolio.formulation import (
    PortfolioProblem, ObjectiveWeights, evaluate, build_qubo, qubo_cost,
    qubo_to_ising,
)


def _problem(seed: int = 0):
    u = make_universe(n_assets=8, n_sectors=3, seed=seed)
    transaction_costs = np.zeros(u.n_assets, dtype=float)
    transaction_costs[[0, 2, 6]] = 0.05
    return PortfolioProblem(
        universe=u, K_target=3,
        weights=ObjectiveWeights(
            lam_return=1.0, lam_variance=2.0, P_K=4.0,
            P_S=0.0, P_R=0.7, theta_risk=0.04,
        ),
        transaction_costs=transaction_costs,
    )


def test_qubo_matches_evaluate():
    p = _problem()
    Q, c = build_qubo(p)
    rng = np.random.default_rng(0)
    for _ in range(256):
        x = rng.integers(0, 2, size=p.universe.n_assets)
        f_eval = evaluate(p, x)["cost"]
        f_qubo = qubo_cost(Q, c, x)
        assert abs(f_eval - f_qubo) < 1e-9, (f_eval, f_qubo)


def test_ising_matches_qubo():
    p = _problem(seed=1)
    Q, c = build_qubo(p)
    ising = qubo_to_ising(Q, c)
    rng = np.random.default_rng(1)
    for _ in range(256):
        x = rng.integers(0, 2, size=p.universe.n_assets)
        f_qubo = qubo_cost(Q, c, x)
        f_ising = ising.evaluate_x(x)
        assert abs(f_qubo - f_ising) < 1e-9, (f_qubo, f_ising)


def test_ising_pauli_offset_matches():
    """The SparsePauliOp expectation on a computational-basis state plus
    the offset must equal the Ising energy."""
    from qiskit.quantum_info import Statevector

    from portfolio.formulation import ising_to_pauli

    p = _problem(seed=2)
    Q, c = build_qubo(p)
    ising = qubo_to_ising(Q, c)
    op, offset = ising_to_pauli(ising)

    n = p.universe.n_assets
    rng = np.random.default_rng(2)
    for _ in range(8):
        x = rng.integers(0, 2, size=n)
        # Build computational-basis Statevector using qiskit's bit ordering:
        # the integer's i-th bit is qubit i (lsb).
        idx = int(sum(int(b) << i for i, b in enumerate(x)))
        sv = Statevector.from_int(idx, dims=2 ** n)
        e = float(sv.expectation_value(op).real) + offset
        target = qubo_cost(Q, c, x)
        assert abs(e - target) < 1e-9, (e, target)


def test_brute_force_optimum_lies_on_K_shell():
    """With a sufficiently large P_K, the global optimum of the *unconstrained*
    QUBO must coincide with the cardinality-K-shell optimum.
    """
    from portfolio.classical import brute_force, brute_force_full

    p = _problem(seed=3)
    bf = brute_force(p)
    bff = brute_force_full(p)
    assert abs(bf.cost - bff.cost) < 1e-9
    assert int(bf.x.sum()) == p.K_target
    assert int(bff.x.sum()) == p.K_target


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            print(f"Running {fn_name} ...", end=" ", flush=True)
            fn()
            print("OK")
    print("All tests passed.")
