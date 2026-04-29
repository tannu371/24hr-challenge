"""Multi-objective portfolio formulation and QUBO / Ising mapping.

Deck-faithful version
---------------------
This module implements the formulation in slides 4–8 of the project deck.

Decision variables (slide 6, Reformulation A)
    x_i ∈ {0,1},   w_i = x_i / K,   Σ x_i = K (target cardinality)

Multi-objective scalarised cost (slide 7, QUBO-PORT)

    H(x) = (λ₂ / K²) · xᵀ Σ x                       ① variance / risk
         − (λ₁ / K) · μᵀ x                          ② expected return
         + P_K · (Σ x_i − K)²                       ③ cardinality penalty
         + Σ_s P_S · (Σ_{i∈s} x_i − L_s)²           ④ per-sector cap penalties
         + P_R · (Σ_i Σ_ii x_i / K² − θ)²           ⑤ risk-threshold cap
         + Σ_i c_i · x_i                            ⑥ linear transaction cost

Notes on faithfulness to the deck
    • Term ⑤ in the deck is written as P_R · max(0, xᵀΣx/K² − θ)². The
      bilinear form xᵀΣx is degree-2 in x, so squaring max(0,·) of it is
      degree-4 — not QUBO-compatible, even with linear-LHS slack qubits
      (Lucas 2014 §2.4). The deck's footnote on this is mathematically
      misleading. We instead use a *diagonal-variance* proxy
          V̂(x) = (1/K²) · Σ_i Σ_ii x_i
      which is linear in x, so (V̂(x) − θ)² is a proper QUBO term and
      captures the same intent (penalise portfolios whose self-variance
      contribution exceeds θ). The off-diagonal covariance is still
      driven down by term ① with sufficient λ₂.
    • The risk-threshold penalty is symmetric (penalises both above and
      below θ); the deck's max(·) makes it one-sided. Practical
      consequence: tune θ slightly above the desired cap.
    • Sector caps default to "off" (P_S = 0); transaction costs default
      to zeros. Set them via ``ObjectiveWeights`` and the optional
      ``sector_caps`` / ``transaction_costs`` on ``PortfolioProblem``.

QUBO ↔ Ising
    Substitute x_i = (1 − z_i)/2 with z_i ∈ {−1,+1}; the QUBO matrix Q
    (symmetric, with linear part on the diagonal) maps to
        H(z) = const + Σ_i h_i z_i + Σ_{i<j} J_{ij} z_i z_j
    which is what QAOA needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .data import AssetUniverse


# ---------------------------------------------------------------------------
# Scalarisation weights and problem definition
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveWeights:
    """Multi-objective scalarisation weights — deck-faithful naming.

    ``lam_return`` is λ₁ in the deck, ``lam_variance`` is λ₂. The
    cardinality penalty is ``P_K``, sector-cap penalty is ``P_S``,
    risk-threshold penalty is ``P_R``, and ``theta_risk`` is θ — the
    target maximum diagonal-variance proxy (in units of Σ).
    """

    lam_return: float = 1.0          # λ₁
    lam_variance: float = 2.5        # λ₂
    P_K: float = 5.0                 # cardinality penalty
    P_S: float = 0.0                 # sector cap penalty (off by default)
    P_R: float = 0.5                 # risk-threshold penalty
    theta_risk: float = 0.04         # ≈ (20% annual vol)² for typical equity Σ


@dataclass
class PortfolioProblem:
    universe: AssetUniverse
    K_target: int
    weights: ObjectiveWeights
    sector_caps: Dict[int, int] = field(default_factory=dict)
    transaction_costs: Optional[np.ndarray] = None
    sigma_max: Optional[float] = None  # post-hoc hard check

    def __post_init__(self) -> None:
        n = self.universe.n_assets
        if self.transaction_costs is None:
            self.transaction_costs = np.zeros(n, dtype=float)
        if not (1 <= self.K_target <= n):
            raise ValueError("K_target out of range")
        if self.transaction_costs.shape != (n,):
            raise ValueError("transaction_costs shape mismatch")


# ---------------------------------------------------------------------------
# Direct objective evaluation (ground truth — what build_qubo must reproduce)
# ---------------------------------------------------------------------------

def evaluate(problem: PortfolioProblem, x: np.ndarray) -> dict:
    """Evaluate the deck's H(x) plus the unscaled financial metrics."""
    x = np.asarray(x, dtype=int)
    u, w = problem.universe, problem.weights
    K = problem.K_target

    K_held = int(x.sum())
    # Financial metrics evaluated at equal weight w = x / K_held (if any).
    if K_held > 0:
        weights_vec = x / K_held
        port_return = float(u.mu @ weights_vec)
        port_variance = float(weights_vec @ u.sigma @ weights_vec)
        port_vol = float(np.sqrt(max(port_variance, 0.0)))
    else:
        port_return = port_variance = port_vol = 0.0

    inv_K = 1.0 / K
    inv_K2 = inv_K * inv_K

    cost = 0.0
    # ① variance, scaled
    cost += w.lam_variance * inv_K2 * float(x @ u.sigma @ x)
    # ② return, scaled
    cost += -w.lam_return * inv_K * float(u.mu @ x)
    # ③ cardinality
    cost += w.P_K * (K_held - K) ** 2
    # ④ sector caps — only sectors with an explicit cap contribute
    if w.P_S > 0.0 and problem.sector_caps:
        sectors = u.sectors
        for s, L_s in problem.sector_caps.items():
            mask = sectors == s
            cnt = int(x[mask].sum())
            cost += w.P_S * (cnt - L_s) ** 2
    # ⑤ risk-threshold (diagonal-variance proxy)
    diag_sigma = np.diag(u.sigma)
    V_hat = inv_K2 * float(diag_sigma @ x)
    cost += w.P_R * (V_hat - w.theta_risk) ** 2
    # ⑥ linear transaction costs
    cost += float(problem.transaction_costs @ x)

    feasible_card = (K_held == K)
    feasible_risk = (problem.sigma_max is None) or (port_vol <= problem.sigma_max)

    return dict(
        K=K_held,
        cost=cost,
        port_return=port_return,
        port_variance=port_variance,
        port_vol=port_vol,
        diag_variance_proxy=V_hat,
        feasible_card=bool(feasible_card),
        feasible_risk=bool(feasible_risk),
    )


# ---------------------------------------------------------------------------
# QUBO matrix construction
# ---------------------------------------------------------------------------

def build_qubo(problem: PortfolioProblem) -> Tuple[np.ndarray, float]:
    """Symmetric Q (linear terms on the diagonal) + constant c such that
    F(x) = xᵀ Q x + c equals evaluate(problem, x)["cost"] for all binary x.
    """
    u, w = problem.universe, problem.weights
    n = u.n_assets
    K = problem.K_target
    inv_K = 1.0 / K
    inv_K2 = inv_K * inv_K

    Q = np.zeros((n, n))
    c = 0.0

    # ① variance: (λ₂/K²) xᵀΣx — Σ symmetric
    Q += w.lam_variance * inv_K2 * u.sigma

    # ② return: −(λ₁/K) μᵀx → diagonal
    for i in range(n):
        Q[i, i] += -w.lam_return * inv_K * u.mu[i]

    # ③ cardinality penalty: P_K · (Σx − K)²
    #   = P_K Σx_i + 2 P_K Σ_{i<j} x_i x_j − 2 P_K K Σx_i + P_K K²  (x² = x)
    #   = P_K (1 − 2K) Σx_i + 2 P_K Σ_{i<j} x_i x_j + P_K K²
    for i in range(n):
        Q[i, i] += w.P_K * (1.0 - 2.0 * K)
        for j in range(i + 1, n):
            Q[i, j] += w.P_K
            Q[j, i] += w.P_K
    c += w.P_K * (K * K)

    # ④ sector caps — for each sector s with explicit cap L_s
    if w.P_S > 0.0 and problem.sector_caps:
        sectors = u.sectors
        for s, L_s in problem.sector_caps.items():
            members = np.where(sectors == s)[0]
            for i in members:
                Q[i, i] += w.P_S * (1.0 - 2.0 * L_s)
                for j in members:
                    if j > i:
                        Q[i, j] += w.P_S
                        Q[j, i] += w.P_S
            c += w.P_S * (L_s * L_s)

    # ⑤ risk-threshold proxy: P_R (Σ_i a_i x_i − θ)² with a_i = Σ_ii / K²
    a = np.diag(u.sigma) * inv_K2
    theta = w.theta_risk
    # Linear: (a_i² − 2 θ a_i) on diagonal
    for i in range(n):
        Q[i, i] += w.P_R * (a[i] * a[i] - 2.0 * theta * a[i])
    # Cross: 2 a_i a_j x_i x_j → split symmetric
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += w.P_R * a[i] * a[j]
            Q[j, i] += w.P_R * a[i] * a[j]
    c += w.P_R * (theta * theta)

    # ⑥ linear transaction costs: Σ c_i x_i → diagonal
    tc = np.asarray(problem.transaction_costs, dtype=float)
    for i in range(n):
        Q[i, i] += tc[i]

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
    offset: float         # const such that H(z) = zᵀ J z + hᵀ z + offset
                          # equals the QUBO cost on x = (1 − z)/2.

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
    """Substitute x_i = (1 − z_i)/2 algebraically.

    For x = (1 − z)/2:
        x_i x_j = (1 − z_i)(1 − z_j)/4 = ¼(1 − z_i − z_j + z_i z_j)
        x_i      = (1 − z_i)/2

    For symmetric Q with diagonal carrying the linear part,
        xᵀ Q x = Σ_i Q_ii x_i + Σ_{i≠j} Q_ij x_i x_j
                = Σ_i Q_ii x_i + 2 Σ_{i<j} Q_ij x_i x_j   (Q symmetric)
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = float(c)

    # Linear (diagonal) terms: Lii x_i = Lii (1 − z_i)/2.
    for i in range(n):
        Lii = Q[i, i]
        offset += 0.5 * Lii
        h[i] += -0.5 * Lii

    # Quadratic terms.
    for i in range(n):
        for j in range(i + 1, n):
            Qij = Q[i, j] + Q[j, i]
            offset += Qij * 0.25
            h[i] += -Qij * 0.25
            h[j] += -Qij * 0.25
            J[i, j] += Qij * 0.25

    return IsingModel(h=h, J=J, offset=offset)


def ising_to_pauli(ising: IsingModel):
    """Translate to a `qiskit.quantum_info.SparsePauliOp` cost Hamiltonian.

    Uses Z_i, Z_i Z_j strings.  We DROP the constant offset (it only shifts
    energy by a global constant — QAOA optimisation is invariant to it).
    Returns the offset separately so callers can reconstruct the
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
        op = SparsePauliOp.from_list([("I" * n, 0.0)])
    else:
        op = SparsePauliOp.from_list(list(zip(paulis, coeffs)))
    return op, ising.offset
