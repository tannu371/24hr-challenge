"""Multi-objective portfolio formulation and QUBO / Ising mapping.

Decision variables
------------------
We treat the *combinatorial* asset-selection layer with binary variables
``x_i in {0, 1}``: 1 iff asset i is in the portfolio. The portfolio weight is
defined as equal-weight on selected assets,

    w_i(x) = x_i / K       with    K := sum_i x_i.

This is a deliberate modelling choice. The hard part of the real problem is
*which* assets to hold; Markowitz on the continuous weights of any *fixed*
asset set has a closed-form solution. Forcing a binary front-end exposes the
combinatorial structure that quantum heuristics target, and keeps the QUBO
quadratic.

Multi-objective scalarisation
-----------------------------
We minimise

    F(x) = - lam_R * mu^T x                              (return)
           + lam_V * x^T Sigma x                          (variance)
           + lam_D * sum_{(i,j) in P_rho} x_i x_j         (diversification)
           + lam_T * sum_i (x_i - x_prev_i)^2             (transaction cost)
           + lam_K * (sum_i x_i - K_target)^2             (cardinality)

P_rho is the set of asset pairs with correlation > rho_threshold. Adding the
indicator x_i x_j therefore *penalises* holding correlated assets together --
this is a quadratic, hardware-friendly diversification term. (See README for
why we did *not* use a sector-based "max M_s per sector" inequality: that
needs slack qubits which inflate problem size; the pairwise penalty captures
the same intent and keeps the QUBO compact.)

The risk-threshold constraint is enforced softly by tuning ``lam_V`` upward,
or hard via a check after sampling. A truly hard sigma_max constraint is
non-quadratic (it's a max(.,.)) and would require slack variables; we discuss
that explicitly in the research note.

QUBO -> Ising
-------------
Substitute x_i = (1 - z_i) / 2 with z_i in {-1, +1}. A QUBO with matrix Q
(symmetric, with linear part on the diagonal) maps to an Ising Hamiltonian

    H(z) = const + sum_i h_i z_i + sum_{i<j} J_{ij} z_i z_j

which is what QAOA needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .data import AssetUniverse


# ---------------------------------------------------------------------------
# Scalarisation weights and problem definition
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveWeights:
    """Multi-objective scalarisation weights.

    The default values were chosen so that on the default 12-asset universe
    every term contributes a similar order of magnitude to the cost; see
    `experiments/run_main.py` for the calibration.
    """

    lam_return: float = 1.0
    lam_variance: float = 2.5
    lam_diversification: float = 0.5
    lam_transaction: float = 0.2
    lam_cardinality: float = 5.0       # penalty multiplier for cardinality
    rho_threshold: float = 0.5         # corr above which a pair is penalised


@dataclass
class PortfolioProblem:
    universe: AssetUniverse
    K_target: int
    weights: ObjectiveWeights
    sigma_max: Optional[float] = None  # hard risk threshold (post-hoc check)
    x_prev: Optional[np.ndarray] = None  # previous portfolio (for tx cost)

    def __post_init__(self) -> None:
        n = self.universe.n_assets
        if self.x_prev is None:
            self.x_prev = np.zeros(n, dtype=int)
        if self.x_prev.shape != (n,):
            raise ValueError("x_prev shape mismatch")
        if not (1 <= self.K_target <= n):
            raise ValueError("K_target out of range")


# ---------------------------------------------------------------------------
# Direct (un-relaxed) objective evaluation
# ---------------------------------------------------------------------------

def evaluate(problem: PortfolioProblem, x: np.ndarray) -> dict:
    """Evaluate the multi-objective cost AND the unscaled financial metrics.

    Returns a dict of components for transparent analysis.
    """
    x = np.asarray(x, dtype=int)
    u, w = problem.universe, problem.weights

    K = int(x.sum())
    # Financial metrics computed at equal-weight w = x / K (if K > 0).
    if K > 0:
        weights_vec = x / K
        port_return = float(u.mu @ weights_vec)
        port_variance = float(weights_vec @ u.sigma @ weights_vec)
        port_vol = float(np.sqrt(max(port_variance, 0.0)))
    else:
        port_return = 0.0
        port_variance = 0.0
        port_vol = 0.0

    # Diversification: count of correlated pairs both held.
    corr = u.correlation()
    iu, ju = np.triu_indices(u.n_assets, k=1)
    pair_mask = corr[iu, ju] > w.rho_threshold
    n_corr_pairs = int((x[iu[pair_mask]] * x[ju[pair_mask]]).sum())

    # Transaction cost: number of changes between x_prev and x.
    n_changes = int(np.sum((x - problem.x_prev) ** 2))

    # Cardinality slack.
    card_slack = (K - problem.K_target) ** 2

    cost = (
        - w.lam_return * float(u.mu @ x)
        + w.lam_variance * float(x @ u.sigma @ x)
        + w.lam_diversification * n_corr_pairs
        + w.lam_transaction * n_changes
        + w.lam_cardinality * card_slack
    )

    feasible_card = (K == problem.K_target)
    feasible_risk = (problem.sigma_max is None) or (port_vol <= problem.sigma_max)

    return dict(
        K=K,
        cost=cost,
        port_return=port_return,
        port_variance=port_variance,
        port_vol=port_vol,
        n_corr_pairs=n_corr_pairs,
        n_changes=n_changes,
        card_slack=card_slack,
        feasible_card=bool(feasible_card),
        feasible_risk=bool(feasible_risk),
    )


# ---------------------------------------------------------------------------
# QUBO matrix construction
# ---------------------------------------------------------------------------

def build_qubo(problem: PortfolioProblem) -> Tuple[np.ndarray, float]:
    """Build symmetric Q (with linear terms on the diagonal) and constant c
    such that F(x) = x^T Q x + c for x in {0,1}^N, exactly equal to evaluate.

    Note we put linear terms on the diagonal: for x_i in {0,1},
        x_i = x_i^2  =>  L_i x_i  appears as  Q_{ii} = L_i.
    """
    u, w = problem.universe, problem.weights
    n = u.n_assets

    Q = np.zeros((n, n))
    c = 0.0  # constant offset

    # 1) Return:  -lam_R mu^T x
    for i in range(n):
        Q[i, i] += -w.lam_return * u.mu[i]

    # 2) Variance:  +lam_V * x^T Sigma x  (Sigma symmetric).
    Q += w.lam_variance * u.sigma

    # 3) Diversification: pairwise penalty for highly-correlated holdings.
    corr = u.correlation()
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > w.rho_threshold:
                # x_i x_j contributes once; symmetric Q matrix splits it 1/2 + 1/2.
                Q[i, j] += w.lam_diversification / 2.0
                Q[j, i] += w.lam_diversification / 2.0

    # 4) Transaction cost: lam_T * sum (x_i - p_i)^2
    #    = lam_T * sum (x_i - 2 p_i x_i + p_i^2)   (since x_i^2 = x_i for binary)
    #    = lam_T * sum ((1 - 2 p_i) x_i + p_i^2)
    p = problem.x_prev.astype(float)
    for i in range(n):
        Q[i, i] += w.lam_transaction * (1.0 - 2.0 * p[i])
    c += w.lam_transaction * float(np.sum(p ** 2))

    # 5) Cardinality penalty: lam_K * (sum x_i - K_t)^2
    #    = lam_K * (sum x_i)^2 - 2 lam_K K_t sum x_i + lam_K K_t^2
    #    (sum x_i)^2 = sum_i x_i + 2 sum_{i<j} x_i x_j   (binary)
    K_t = problem.K_target
    for i in range(n):
        Q[i, i] += w.lam_cardinality * (1.0 - 2.0 * K_t)
        for j in range(i + 1, n):
            Q[i, j] += w.lam_cardinality
            Q[j, i] += w.lam_cardinality
    c += w.lam_cardinality * (K_t ** 2)

    return Q, c


def qubo_cost(Q: np.ndarray, c: float, x: np.ndarray) -> float:
    return float(x @ Q @ x + c)


# ---------------------------------------------------------------------------
# Ising mapping
# ---------------------------------------------------------------------------

@dataclass
class IsingModel:
    h: np.ndarray         # (N,) field
    J: np.ndarray         # (N, N) upper-triangular couplings (J[i,j] for i<j)
    offset: float         # constant such that H(z) = z^T J z + h^T z + offset
                          # equals QUBO cost on x = (1 - z)/2.

    def n_qubits(self) -> int:
        return len(self.h)

    def evaluate_z(self, z: np.ndarray) -> float:
        z = np.asarray(z)
        return float(z @ self.J @ z + self.h @ z + self.offset)

    def evaluate_x(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=int)
        z = 1 - 2 * x
        return self.evaluate_z(z)


def qubo_to_ising(Q: np.ndarray, c: float) -> IsingModel:
    """Map x_i = (1 - z_i)/2 algebraically.

    For x = (1 - z)/2:
      x_i x_j = (1 - z_i)(1 - z_j)/4 = 1/4 (1 - z_i - z_j + z_i z_j)
      x_i      = (1 - z_i)/2

    For symmetric Q with diagonal carrying the linear part,
      x^T Q x  =  sum_i Q_ii x_i  +  sum_{i!=j} Q_ij x_i x_j
              =  sum_i Q_ii x_i  +  2 sum_{i<j} Q_ij x_i x_j   (Q symmetric)
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = float(c)

    # Linear (diagonal) terms.
    for i in range(n):
        Lii = Q[i, i]
        # Lii x_i = Lii (1 - z_i)/2 = Lii/2 - Lii/2 z_i
        offset += 0.5 * Lii
        h[i]   += -0.5 * Lii

    # Quadratic terms.
    for i in range(n):
        for j in range(i + 1, n):
            Qij = Q[i, j] + Q[j, i]   # since loop only over i<j we sum both halves
            # Qij x_i x_j = Qij/4 (1 - z_i - z_j + z_i z_j)
            offset += Qij * 0.25
            h[i]   += -Qij * 0.25
            h[j]   += -Qij * 0.25
            J[i, j] += Qij * 0.25

    return IsingModel(h=h, J=J, offset=offset)


def ising_to_pauli(ising: IsingModel):
    """Translate to a `qiskit.quantum_info.SparsePauliOp` cost Hamiltonian.

    Uses Z_i, Z_i Z_j strings.  We DROP the constant offset because it only
    shifts the energy by a global constant -- QAOA's optimisation is invariant
    to it. We return the offset separately so callers can reconstruct the
    physically-meaningful cost.
    """
    from qiskit.quantum_info import SparsePauliOp  # local import keeps tests fast

    n = ising.n_qubits()
    paulis, coeffs = [], []
    for i in range(n):
        if ising.h[i] != 0.0:
            s = ["I"] * n
            s[n - 1 - i] = "Z"          # qiskit qubit ordering: qubit 0 = rightmost
            paulis.append("".join(s))
            coeffs.append(ising.h[i])
    for i in range(n):
        for j in range(i + 1, n):
            if ising.J[i, j] != 0.0:
                s = ["I"] * n
                s[n - 1 - i] = "Z"
                s[n - 1 - j] = "Z"
                paulis.append("".join(s))
                coeffs.append(ising.J[i, j])

    if not paulis:
        # Degenerate case: identity only
        op = SparsePauliOp.from_list([("I" * n, 0.0)])
    else:
        op = SparsePauliOp.from_list(list(zip(paulis, coeffs)))
    return op, ising.offset
