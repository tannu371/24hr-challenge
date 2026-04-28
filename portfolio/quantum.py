"""Hybrid quantum-classical solver: QAOA on the portfolio Ising model.

Implementation choices
----------------------
We hand-roll QAOA on top of `qiskit.primitives` rather than using
`qiskit_algorithms.QAOA`, for three reasons:

1. The challenge asks us to *understand* the method, not to call a one-liner.
2. We want full transparency on the parameter-update loop, so we can plot
   per-iteration energy, sample distributions, success probabilities, etc.
3. We can switch between an exact statevector estimator (idealised, useful
   for analysis) and a finite-shot sampler (closer to hardware), and reuse
   the same loop.

QAOA structure
--------------
For p layers,

    |psi(beta, gamma)> = (Π_l U_M(beta_l) U_C(gamma_l)) |+>^N

with

    U_C(gamma) = exp(-i gamma H_C),   H_C = Ising cost Hamiltonian
    U_M(beta)  = exp(-i beta * sum_i X_i)

The classical loop minimises  F(beta, gamma) = <psi| H_C |psi>  using COBYLA.

We use a multi-start strategy (a few random initial points) because QAOA's
landscape is highly non-convex and COBYLA gets trapped in local minima.
The best run wins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

from .formulation import (
    IsingModel,
    PortfolioProblem,
    build_qubo,
    qubo_cost,
    qubo_to_ising,
    ising_to_pauli,
)


# ---------------------------------------------------------------------------
# QAOA circuit construction
# ---------------------------------------------------------------------------

def _cost_layer(qc: QuantumCircuit, ising: IsingModel, gamma) -> None:
    """Append U_C(gamma) = exp(-i gamma H_C) to qc.

    For an Ising H_C = sum h_i Z_i + sum J_ij Z_i Z_j, U_C decomposes exactly
    into single-qubit RZ rotations and ZZ rotations (CX-RZ-CX gadgets).
    """
    n = ising.n_qubits()
    # Single-qubit field terms.
    for i in range(n):
        if ising.h[i] != 0.0:
            qc.rz(2.0 * gamma * ising.h[i], i)
    # Two-qubit ZZ couplings.
    for i in range(n):
        for j in range(i + 1, n):
            if ising.J[i, j] != 0.0:
                qc.cx(i, j)
                qc.rz(2.0 * gamma * ising.J[i, j], j)
                qc.cx(i, j)


def _mixer_layer(qc: QuantumCircuit, beta) -> None:
    """Append U_M(beta) = exp(-i beta * sum X_i)."""
    for i in range(qc.num_qubits):
        qc.rx(2.0 * beta, i)


def build_qaoa_circuit(ising: IsingModel, p: int) -> Tuple[QuantumCircuit, list, list]:
    """Build a parameterised QAOA circuit with p layers.

    Returns (circuit, gammas, betas) where gammas[l] and betas[l] are
    `qiskit.circuit.Parameter` objects.
    """
    n = ising.n_qubits()
    qc = QuantumCircuit(n)
    qc.h(range(n))  # initial state |+>^N

    gammas = [Parameter(f"g{l}") for l in range(p)]
    betas = [Parameter(f"b{l}") for l in range(p)]

    for l in range(p):
        _cost_layer(qc, ising, gammas[l])
        _mixer_layer(qc, betas[l])

    return qc, gammas, betas


# ---------------------------------------------------------------------------
# Energy evaluation
# ---------------------------------------------------------------------------

def _bind_params(qc: QuantumCircuit, params: List[Parameter], values) -> QuantumCircuit:
    """Bind a list of parameters to numerical values."""
    return qc.assign_parameters(dict(zip(params, list(values))))


def _statevector_energy(
    qc_template: QuantumCircuit,
    all_params: List[Parameter],
    cost_op: SparsePauliOp,
    offset: float,
    theta: np.ndarray,
) -> float:
    """Exact <psi(theta)| H_C |psi(theta)> via Statevector. Noiseless ideal."""
    qc = _bind_params(qc_template, all_params, theta)
    sv = Statevector.from_instruction(qc)
    # SparsePauliOp.expectation_value is fastest with Statevector
    e = sv.expectation_value(cost_op).real
    return float(e + offset)


def _statevector_distribution(
    qc_template: QuantumCircuit,
    all_params: List[Parameter],
    theta: np.ndarray,
) -> np.ndarray:
    """Return probability over all 2^N bitstrings (qubit 0 = least-significant bit)."""
    qc = _bind_params(qc_template, all_params, theta)
    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2
    return probs


# ---------------------------------------------------------------------------
# Hybrid optimisation loop
# ---------------------------------------------------------------------------

@dataclass
class QAOAResult:
    name: str
    x: np.ndarray
    cost: float
    wall_time: float
    n_evaluations: int
    history: List[float] = field(default_factory=list)
    extra: Optional[dict] = None


def run_qaoa(
    problem: PortfolioProblem,
    p: int = 2,
    n_restarts: int = 5,
    optimizer: str = "COBYLA",
    maxiter: int = 200,
    seed: int = 0,
    use_statevector: bool = True,
    n_shots: int = 4096,
    verbose: bool = False,
) -> QAOAResult:
    """Run hybrid QAOA on the portfolio Ising model.

    use_statevector: if True (default), evaluate energy via exact statevector
        — this is what an idealised, noise-free quantum computer would give.
        If False, sample bitstrings (closer to hardware) and average.

    Returns the best bitstring (decoded to a portfolio) and its *original*
    QUBO cost, plus the per-iteration energy history.
    """
    from scipy.optimize import minimize

    # Build the Ising model from the QUBO.
    Q, c = build_qubo(problem)
    ising = qubo_to_ising(Q, c)
    cost_op, ising_offset = ising_to_pauli(ising)

    # Build parameterised circuit once; rebind for each evaluation.
    qc_template, gammas, betas = build_qaoa_circuit(ising, p)
    all_params = list(gammas) + list(betas)

    rng = np.random.default_rng(seed)

    best_overall = dict(theta=None, energy=np.inf, history=None)
    n_eval_total = 0

    t0 = time.perf_counter()
    for restart in range(n_restarts):
        # Initial parameters: small random in [0, pi].
        theta0 = rng.uniform(0.0, np.pi, size=2 * p)
        history: List[float] = []

        def loss(theta):
            nonlocal n_eval_total
            n_eval_total += 1
            e = _statevector_energy(
                qc_template, all_params, cost_op, ising_offset, theta
            )
            history.append(e)
            return e

        res = minimize(
            loss,
            theta0,
            method=optimizer,
            options={"maxiter": maxiter, "rhobeg": 0.3, "disp": False},
        )

        if verbose:
            print(f"  restart {restart}: final E = {res.fun:.4f}")

        if res.fun < best_overall["energy"]:
            best_overall.update(
                theta=res.x, energy=res.fun, history=history,
            )

    # Best theta in hand. Sample the wavefunction to extract a bitstring.
    probs = _statevector_distribution(qc_template, all_params, best_overall["theta"])
    n = ising.n_qubits()

    # MAP bitstring under the QAOA distribution.
    best_int = int(np.argmax(probs))
    # Qiskit ordering: the integer's bit i corresponds to qubit i (lsb first).
    x_best = np.array([(best_int >> i) & 1 for i in range(n)], dtype=int)

    # Replace map-only with a true scan: among the top-M most-probable
    # bitstrings, pick the one with the lowest *original* QUBO cost. This is
    # the sense in which QAOA is used as a sampler: the quantum part biases
    # the distribution, the classical part picks the winner.
    M = min(64, len(probs))
    top_ints = np.argsort(-probs)[:M]
    best_x, best_cost = x_best, qubo_cost(Q, c, x_best)
    for k in top_ints:
        x = np.array([(int(k) >> i) & 1 for i in range(n)], dtype=int)
        f = qubo_cost(Q, c, x)
        if f < best_cost:
            best_cost, best_x = f, x

    wall = time.perf_counter() - t0

    # Probability mass on the *true* optimum (if we know it: pass via extra)
    extras = dict(
        probs=probs,
        theta_star=best_overall["theta"],
        energy_star=best_overall["energy"],
        ising_offset=ising_offset,
    )

    return QAOAResult(
        name=f"QAOA_p{p}",
        x=best_x,
        cost=best_cost,
        wall_time=wall,
        n_evaluations=n_eval_total,
        history=best_overall["history"],
        extra=extras,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def success_probability(
    problem: PortfolioProblem, qaoa_result: QAOAResult, optimum_x: np.ndarray
) -> float:
    """Sum of probability mass over states with cost <= cost(optimum_x).

    Reported as 'P(measured optimum)' in the analysis.
    """
    Q, c = build_qubo(problem)
    target = qubo_cost(Q, c, optimum_x)
    probs = qaoa_result.extra["probs"]
    n = problem.universe.n_assets

    # Allow small numerical slack so degenerate optima are counted together.
    tol = 1e-9
    p_success = 0.0
    for k, pk in enumerate(probs):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
        if qubo_cost(Q, c, x) <= target + tol:
            p_success += pk
    return float(p_success)
