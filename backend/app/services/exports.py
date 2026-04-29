"""Export utilities (Phase 5).

Given a trial_id, this module can:

* rebuild the original PortfolioProblem from the trial's params and the
  trial's stored asset_names / mu / sigma. We re-run the synthetic generator
  with the original seed to recover μ and Σ exactly.
* re-construct the parameter-bound QAOA circuit (when the trial has θ*).
* emit QASM3, a self-contained Qiskit .py script, a PennyLane .py script.
* render the circuit (pre- and post-transpilation) as SVG + PNG.
* render the four standard analysis plots as SVG + PNG + CSV.

We deliberately serialise μ and Σ inline in the standalone scripts rather than
re-running the seeded generator inside them — that way the script reproduces
the *exact* trial regardless of dependency drift in `app.services.synthetic`.
"""

from __future__ import annotations

import csv
import io
import json
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from portfolio.classical import brute_force, simulated_annealing
from portfolio.formulation import (
    ObjectiveWeights,
    PortfolioProblem,
    build_qubo,
    qubo_cost,
    qubo_to_ising,
)

from .problem_builder import build_portfolio_problem
from .qaoa import build_qaoa_circuit, statevector_probabilities, landscape_p1
from .trials_store import TrialsStore


# ---------------------------------------------------------------------------
# Trial → rebuilt artifacts
# ---------------------------------------------------------------------------


@dataclass
class RebuiltTrial:
    trial: dict
    problem: PortfolioProblem
    p: int
    mixer: str
    init_state: str
    theta: list[float]
    Q: np.ndarray
    qubo_offset: float


def load_trial(trial_id: int) -> dict:
    record = TrialsStore().get(trial_id)
    if record is None:
        raise KeyError(f"trial {trial_id} not found")
    return record


def rebuild_qaoa_trial(trial_id: int) -> RebuiltTrial:
    trial = load_trial(trial_id)
    if trial["kind"] not in {"qaoa_sim", "qaoa_hw"}:
        raise ValueError(
            f"trial {trial_id} is {trial['kind']!r} — only qaoa_sim/qaoa_hw can be exported as circuits"
        )

    params = trial["params"]
    results = trial["results"]
    theta = results.get("theta_star")
    p = int(results.get("p") or params.get("p"))
    mixer = results.get("mixer") or params.get("mixer", "x")
    init_state = results.get("init_state") or params.get("init_state", "uniform")
    if theta is None or p is None:
        raise ValueError(f"trial {trial_id} missing theta_star or p")

    problem, _mode = build_portfolio_problem(params)
    Q, c = build_qubo(problem)
    return RebuiltTrial(
        trial=trial, problem=problem, p=p, mixer=mixer,
        init_state=init_state, theta=list(theta), Q=Q, qubo_offset=float(c),
    )


def rebuild_circuit(rt: RebuiltTrial, bind_theta: bool = True):
    ising = qubo_to_ising(rt.Q, rt.qubo_offset)
    K = rt.problem.K_target if rt.init_state == "dicke" else None
    circuit = build_qaoa_circuit(ising, p=rt.p, mixer=rt.mixer, init_state=rt.init_state, K=K)
    qc = circuit.bind(np.asarray(rt.theta)) if bind_theta else circuit.qc
    return qc, circuit


# ---------------------------------------------------------------------------
# 5.1 — QASM 3
# ---------------------------------------------------------------------------


def export_qasm3(trial_id: int) -> str:
    rt = rebuild_qaoa_trial(trial_id)
    qc, _ = rebuild_circuit(rt, bind_theta=True)
    # Append measurements so the QASM is runnable end-to-end.
    measured = qc.copy()
    measured.measure_all()

    from qiskit.qasm3 import dumps  # qiskit ≥ 1.0 ships qasm3
    return dumps(measured)


# ---------------------------------------------------------------------------
# 5.2 — Standalone Qiskit .py
# ---------------------------------------------------------------------------


def export_qiskit_script(trial_id: int) -> str:
    rt = rebuild_qaoa_trial(trial_id)
    universe = rt.problem.universe
    mu = universe.mu.tolist()
    sigma = universe.sigma.tolist()
    asset_names = list(universe.asset_names)
    weights = rt.problem.weights
    K = rt.problem.K_target

    return _qiskit_script_template(
        trial_id=trial_id,
        kind=rt.trial["kind"],
        N=universe.n_assets,
        K=K,
        asset_names=asset_names,
        mu=mu,
        sigma=sigma,
        sectors=universe.sectors.tolist(),
        lam_return=weights.lam_return,
        lam_variance=weights.lam_variance,
        P_K=weights.P_K,
        P_S=weights.P_S,
        P_R=weights.P_R,
        theta_risk=weights.theta_risk,
        sector_caps=dict(rt.problem.sector_caps),
        transaction_costs=rt.problem.transaction_costs.tolist()
            if rt.problem.transaction_costs is not None else [0.0] * universe.n_assets,
        p=rt.p,
        mixer=rt.mixer,
        init_state=rt.init_state,
        theta=rt.theta,
    )


def _qiskit_script_template(**kw: Any) -> str:
    return textwrap.dedent(f'''\
        """Standalone Qiskit reproduction of trial {kw["trial_id"]} ({kw["kind"]}).

        Auto-generated by /export/qiskit/{kw["trial_id"]}. Run with:

            pip install "qiskit>=1.0" qiskit-aer numpy scipy
            python qaoa_trial_{kw["trial_id"]}.py

        Reproduces the deck-faithful multi-objective QUBO (slides 4–8):

            H(x) = (lambda_2 / K^2) x^T Sigma x
                 - (lambda_1 / K) mu^T x
                 + P_K (sum x_i - K)^2
                 + sum_s P_S (sum_{{i in s}} x_i - L_s)^2
                 + P_R (sum_i Sigma_ii x_i / K^2 - theta)^2     (diag proxy)
                 + sum_i c_i x_i

        The mean vector mu and covariance Sigma are baked in so this script
        reproduces the exact trial regardless of upstream code drift.
        """

        from __future__ import annotations
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit.quantum_info import SparsePauliOp, Statevector

        # ---------------- Problem (baked in) ----------------
        N = {kw["N"]}
        K = {kw["K"]}
        ASSET_NAMES = {kw["asset_names"]!r}
        SECTORS = {kw["sectors"]!r}
        mu    = np.array({kw["mu"]!r})
        Sigma = np.array({kw["sigma"]!r})

        # Multi-objective scalarisation weights (deck slide 7)
        LAM_RETURN   = {kw["lam_return"]}      # lambda_1
        LAM_VARIANCE = {kw["lam_variance"]}    # lambda_2
        P_K          = {kw["P_K"]}             # cardinality penalty
        P_S          = {kw["P_S"]}             # sector cap penalty
        P_R          = {kw["P_R"]}             # risk-threshold penalty
        THETA_RISK   = {kw["theta_risk"]}      # theta in deck term (5)
        SECTOR_CAPS  = {kw["sector_caps"]!r}   # {{sector_idx: L_s}}
        TRANSACTION_COSTS = np.array({kw["transaction_costs"]!r})

        # ---------------- QUBO ----------------
        def build_qubo():
            Q = np.zeros((N, N))
            c = 0.0
            inv_K  = 1.0 / K
            inv_K2 = inv_K * inv_K

            # (1) variance: (lambda_2 / K^2) x^T Sigma x
            Q += LAM_VARIANCE * inv_K2 * Sigma
            # (2) return: -(lambda_1 / K) mu^T x
            for i in range(N):
                Q[i, i] += -LAM_RETURN * inv_K * mu[i]
            # (3) cardinality: P_K (sum x - K)^2
            for i in range(N):
                Q[i, i] += P_K * (1.0 - 2.0 * K)
                for j in range(i + 1, N):
                    Q[i, j] += P_K
                    Q[j, i] += P_K
            c += P_K * (K * K)
            # (4) sector caps
            if P_S > 0.0 and SECTOR_CAPS:
                sectors_arr = np.array(SECTORS)
                for s, L_s in SECTOR_CAPS.items():
                    s = int(s); L_s = int(L_s)
                    members = np.where(sectors_arr == s)[0]
                    for i in members:
                        Q[i, i] += P_S * (1.0 - 2.0 * L_s)
                        for j in members:
                            if j > i:
                                Q[i, j] += P_S
                                Q[j, i] += P_S
                    c += P_S * (L_s * L_s)
            # (5) risk-threshold proxy: P_R (sum_i Sigma_ii x_i / K^2 - theta)^2
            a = np.diag(Sigma) * inv_K2
            for i in range(N):
                Q[i, i] += P_R * (a[i] * a[i] - 2.0 * THETA_RISK * a[i])
            for i in range(N):
                for j in range(i + 1, N):
                    Q[i, j] += P_R * a[i] * a[j]
                    Q[j, i] += P_R * a[i] * a[j]
            c += P_R * (THETA_RISK * THETA_RISK)
            # (6) linear transaction costs
            for i in range(N):
                Q[i, i] += TRANSACTION_COSTS[i]
            return Q, c

        def qubo_to_ising(Q, c):
            n = Q.shape[0]
            h = np.zeros(n); J = np.zeros((n, n)); offset = float(c)
            for i in range(n):
                Lii = Q[i, i]; offset += 0.5 * Lii; h[i] += -0.5 * Lii
            for i in range(n):
                for j in range(i + 1, n):
                    Qij = Q[i, j] + Q[j, i]
                    offset += Qij * 0.25
                    h[i] += -Qij * 0.25; h[j] += -Qij * 0.25
                    J[i, j] += Qij * 0.25
            return h, J, offset

        def ising_pauli(h, J):
            n = len(h); paulis, coeffs = [], []
            for i in range(n):
                if h[i] != 0.0:
                    s = ["I"] * n; s[n - 1 - i] = "Z"
                    paulis.append("".join(s)); coeffs.append(h[i])
            for i in range(n):
                for j in range(i + 1, n):
                    if J[i, j] != 0.0:
                        s = ["I"] * n; s[n - 1 - i] = "Z"; s[n - 1 - j] = "Z"
                        paulis.append("".join(s)); coeffs.append(J[i, j])
            if not paulis:
                return SparsePauliOp.from_list([("I" * n, 0.0)])
            return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

        # ---------------- QAOA ansatz ----------------
        P = {kw["p"]}
        MIXER = {kw["mixer"]!r}
        INIT_STATE = {kw["init_state"]!r}
        THETA = np.array({kw["theta"]!r})    # gammas first, then betas

        def cost_layer(qc, h, J, gamma):
            n = qc.num_qubits
            for i in range(n):
                if h[i] != 0.0: qc.rz(2.0 * gamma * h[i], i)
            for i in range(n):
                for j in range(i + 1, n):
                    if J[i, j] != 0.0:
                        qc.cx(i, j); qc.rz(2.0 * gamma * J[i, j], j); qc.cx(i, j)

        def apply_dicke(qc, k):
            from math import comb
            n = qc.num_qubits; norm = 1.0 / np.sqrt(comb(n, k))
            amps = np.zeros(2 ** n, dtype=complex)
            for s in range(2 ** n):
                if bin(s).count("1") == k: amps[s] = norm
            qc.initialize(amps, list(range(n)))

        def build_qaoa_circuit():
            qc = QuantumCircuit(N)
            if INIT_STATE == "uniform":
                qc.h(range(N))
            else:
                apply_dicke(qc, K)
            gammas = [Parameter(f"g{{l}}") for l in range(P)]
            betas  = [Parameter(f"b{{l}}") for l in range(P)]
            h, J, _ = qubo_to_ising(*build_qubo())
            for l in range(P):
                cost_layer(qc, h, J, gammas[l])
                if MIXER == "x":
                    for i in range(N): qc.rx(2.0 * betas[l], i)
                else:  # xy_ring
                    for i in range(N):
                        j = (i + 1) % N
                        qc.rxx(betas[l], i, j); qc.ryy(betas[l], i, j)
            return qc, list(gammas) + list(betas)

        def main():
            Q, c = build_qubo()
            h, J, ising_offset = qubo_to_ising(Q, c)
            cost_op = ising_pauli(h, J)
            qc, params = build_qaoa_circuit()
            bound = qc.assign_parameters(dict(zip(params, THETA)))
            sv = Statevector.from_instruction(bound)
            energy = float(sv.expectation_value(cost_op).real) + ising_offset
            probs = np.abs(sv.data) ** 2
            top = np.argsort(-probs)[:10]
            print(f"<H_C> = {{energy:.6f}}")
            for k in top:
                x = np.array([(int(k) >> i) & 1 for i in range(N)])
                cost = float(x @ Q @ x + c)
                bitstr = "".join(str(int(b)) for b in reversed(x))
                print(f"  {{bitstr}}  p={{probs[k]:.4f}}  cost={{cost:.4f}}  K={{int(x.sum())}}")

        if __name__ == "__main__":
            main()
        ''')


# ---------------------------------------------------------------------------
# 5.3 — Standalone PennyLane .py
# ---------------------------------------------------------------------------


def export_pennylane_script(trial_id: int) -> str:
    rt = rebuild_qaoa_trial(trial_id)
    universe = rt.problem.universe
    mu = universe.mu.tolist()
    sigma = universe.sigma.tolist()
    weights = rt.problem.weights
    K = rt.problem.K_target

    return _pennylane_script_template(
        trial_id=trial_id,
        kind=rt.trial["kind"],
        N=universe.n_assets,
        K=K,
        mu=mu, sigma=sigma,
        sectors=universe.sectors.tolist(),
        lam_return=weights.lam_return,
        lam_variance=weights.lam_variance,
        P_K=weights.P_K,
        P_S=weights.P_S,
        P_R=weights.P_R,
        theta_risk=weights.theta_risk,
        sector_caps=dict(rt.problem.sector_caps),
        transaction_costs=rt.problem.transaction_costs.tolist()
            if rt.problem.transaction_costs is not None else [0.0] * universe.n_assets,
        p=rt.p, mixer=rt.mixer, init_state=rt.init_state,
        theta=rt.theta,
    )


def _pennylane_script_template(**kw: Any) -> str:
    return textwrap.dedent(f'''\
        """Standalone PennyLane reproduction of trial {kw["trial_id"]} ({kw["kind"]}).

        Auto-generated by /export/pennylane/{kw["trial_id"]}. Run with:

            pip install pennylane pennylane-lightning numpy
            python qaoa_trial_{kw["trial_id"]}_pennylane.py

        Implements the QAOA cost / mixer Hamiltonians as PennyLane qml.Hamiltonian
        objects and uses qml.qaoa.cost_layer / mixer_layer to apply U_C(γ) and
        U_M(β). Theta is bound to the optimised values from the original trial.
        """

        from __future__ import annotations
        import numpy as np
        import pennylane as qml
        from pennylane import qaoa as pl_qaoa
        from math import comb

        N = {kw["N"]}
        K = {kw["K"]}
        SECTORS = {kw["sectors"]!r}
        mu    = np.array({kw["mu"]!r})
        Sigma = np.array({kw["sigma"]!r})

        LAM_RETURN   = {kw["lam_return"]}
        LAM_VARIANCE = {kw["lam_variance"]}
        P_K          = {kw["P_K"]}
        P_S          = {kw["P_S"]}
        P_R          = {kw["P_R"]}
        THETA_RISK   = {kw["theta_risk"]}
        SECTOR_CAPS  = {kw["sector_caps"]!r}
        TRANSACTION_COSTS = np.array({kw["transaction_costs"]!r})

        P = {kw["p"]}
        MIXER = {kw["mixer"]!r}
        INIT_STATE = {kw["init_state"]!r}
        THETA = np.array({kw["theta"]!r})

        def build_qubo():
            Q = np.zeros((N, N)); c = 0.0
            inv_K  = 1.0 / K
            inv_K2 = inv_K * inv_K
            Q += LAM_VARIANCE * inv_K2 * Sigma
            for i in range(N): Q[i, i] += -LAM_RETURN * inv_K * mu[i]
            for i in range(N):
                Q[i, i] += P_K * (1.0 - 2.0 * K)
                for j in range(i + 1, N):
                    Q[i, j] += P_K; Q[j, i] += P_K
            c += P_K * (K * K)
            if P_S > 0.0 and SECTOR_CAPS:
                sectors_arr = np.array(SECTORS)
                for s, L_s in SECTOR_CAPS.items():
                    s = int(s); L_s = int(L_s)
                    members = np.where(sectors_arr == s)[0]
                    for i in members:
                        Q[i, i] += P_S * (1.0 - 2.0 * L_s)
                        for j in members:
                            if j > i:
                                Q[i, j] += P_S; Q[j, i] += P_S
                    c += P_S * (L_s * L_s)
            a = np.diag(Sigma) * inv_K2
            for i in range(N):
                Q[i, i] += P_R * (a[i] * a[i] - 2.0 * THETA_RISK * a[i])
            for i in range(N):
                for j in range(i + 1, N):
                    Q[i, j] += P_R * a[i] * a[j]
                    Q[j, i] += P_R * a[i] * a[j]
            c += P_R * (THETA_RISK * THETA_RISK)
            for i in range(N):
                Q[i, i] += TRANSACTION_COSTS[i]
            return Q, c

        def qubo_to_ising(Q, c):
            n = Q.shape[0]; h = np.zeros(n); J = np.zeros((n, n)); offset = float(c)
            for i in range(n):
                Lii = Q[i, i]; offset += 0.5 * Lii; h[i] += -0.5 * Lii
            for i in range(n):
                for j in range(i + 1, n):
                    Qij = Q[i, j] + Q[j, i]
                    offset += Qij * 0.25
                    h[i] += -Qij * 0.25; h[j] += -Qij * 0.25
                    J[i, j] += Qij * 0.25
            return h, J, offset

        def cost_hamiltonian():
            Q, c = build_qubo()
            h, J, offset = qubo_to_ising(Q, c)
            coeffs, ops = [], []
            for i in range(N):
                if h[i] != 0.0:
                    coeffs.append(float(h[i])); ops.append(qml.PauliZ(i))
            for i in range(N):
                for j in range(i + 1, N):
                    if J[i, j] != 0.0:
                        coeffs.append(float(J[i, j])); ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
            if not coeffs:
                coeffs = [0.0]; ops = [qml.Identity(0)]
            return qml.Hamiltonian(coeffs, ops), offset, Q, c

        def x_mixer():
            return qml.Hamiltonian([1.0] * N, [qml.PauliX(i) for i in range(N)])

        def xy_ring_mixer():
            coeffs, ops = [], []
            for i in range(N):
                j = (i + 1) % N
                coeffs += [0.5, 0.5]
                ops += [qml.PauliX(i) @ qml.PauliX(j), qml.PauliY(i) @ qml.PauliY(j)]
            return qml.Hamiltonian(coeffs, ops)

        def dicke_amplitudes(n, k):
            amps = np.zeros(2 ** n, dtype=complex)
            norm = 1.0 / np.sqrt(comb(n, k))
            for s in range(2 ** n):
                if bin(s).count("1") == k: amps[s] = norm
            return amps

        H_C, OFFSET, Q_MAT, C_CONST = cost_hamiltonian()
        H_M = xy_ring_mixer() if MIXER == "xy_ring" else x_mixer()

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(theta):
            if INIT_STATE == "uniform":
                for i in range(N): qml.Hadamard(wires=i)
            else:
                qml.StatePrep(dicke_amplitudes(N, K), wires=range(N))
            for l in range(P):
                pl_qaoa.cost_layer(theta[l], H_C)
                pl_qaoa.mixer_layer(theta[P + l], H_M)
            return qml.expval(H_C)

        @qml.qnode(dev)
        def circuit_probs(theta):
            if INIT_STATE == "uniform":
                for i in range(N): qml.Hadamard(wires=i)
            else:
                qml.StatePrep(dicke_amplitudes(N, K), wires=range(N))
            for l in range(P):
                pl_qaoa.cost_layer(theta[l], H_C)
                pl_qaoa.mixer_layer(theta[P + l], H_M)
            return qml.probs(wires=range(N))

        if __name__ == "__main__":
            energy = float(circuit(THETA)) + OFFSET
            probs  = np.array(circuit_probs(THETA))
            print(f"<H_C> = {{energy:.6f}}")
            for k in np.argsort(-probs)[:10]:
                x = np.array([(int(k) >> i) & 1 for i in range(N)])
                cost = float(x @ Q_MAT @ x + C_CONST)
                bitstr = "".join(str(int(b)) for b in reversed(x))
                print(f"  {{bitstr}}  p={{probs[k]:.4f}}  cost={{cost:.4f}}  K={{int(x.sum())}}")
        ''')


# ---------------------------------------------------------------------------
# 5.4 — Circuit diagrams (SVG + PNG)
# ---------------------------------------------------------------------------


def render_circuit(trial_id: int, fmt: str = "svg", transpiled: bool = False) -> bytes:
    rt = rebuild_qaoa_trial(trial_id)
    qc, _ = rebuild_circuit(rt, bind_theta=True)
    if transpiled:
        from qiskit import transpile
        qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3)
    return _draw_circuit(qc, fmt)


def _draw_circuit(qc, fmt: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg", force=True)
    fig = qc.draw(output="mpl", fold=-1)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 5.5 — Plots
# ---------------------------------------------------------------------------


def render_plot(trial_id: int, kind: str, fmt: str) -> bytes | str:
    """fmt ∈ {svg, png, csv}.  kind ∈ {trajectory, histogram, landscape, comparison}."""
    if kind == "trajectory":
        return _plot_trajectory(trial_id, fmt)
    if kind == "histogram":
        return _plot_histogram(trial_id, fmt)
    if kind == "landscape":
        return _plot_landscape(trial_id, fmt)
    if kind == "comparison":
        return _plot_comparison(trial_id, fmt)
    raise ValueError(f"unknown plot kind: {kind!r}")


def _plot_trajectory(trial_id: int, fmt: str):
    trial = load_trial(trial_id)
    histories: list[list[float]] = (
        trial["results"].get("history_per_restart")
        or trial["results"].get("runs")  # SA stores list of dicts
        or []
    )
    if isinstance(histories, list) and histories and isinstance(histories[0], dict):
        histories = [r.get("trajectory_per_sweep", []) for r in histories]

    if not histories or not any(histories):
        raise ValueError(
            f"trial {trial_id} ({trial['kind']}) has no per-iteration trajectory "
            f"(only QAOA and SA produce one)"
        )

    if fmt == "csv":
        return _csv_trajectories(histories)
    return _matplot_lines(
        title=f"Trial {trial_id} — energy trajectory",
        x_label="iteration", y_label="energy / cost",
        traces=histories, fmt=fmt,
    )


def _plot_histogram(trial_id: int, fmt: str):
    trial = load_trial(trial_id)
    results = trial["results"]
    top = results.get("top_bitstrings") or []

    # QAOA path — top-K most-probable bitstrings under the quantum distribution.
    if top:
        labels = [d["bitstring"] for d in top]
        probs = [d.get("probability", d.get("count", 0)) for d in top]
        if fmt == "csv":
            return _csv_rows([("bitstring", "probability_or_count", "cost", "K")]
                             + [(d["bitstring"], d.get("probability", d.get("count")),
                                  d.get("cost"), d.get("K")) for d in top])
        return _matplot_bar(
            title=f"Trial {trial_id} — top-{len(top)} bitstring distribution",
            x_label="bitstring", y_label="probability",
            labels=labels, values=probs, fmt=fmt,
        )

    # Brute-force fallback — energy distribution over the C(N,K) feasible shell.
    energies = results.get("energy_distribution")
    if isinstance(energies, list) and energies:
        if fmt == "csv":
            return _csv_rows([("rank", "cost"), *((i, e) for i, e in enumerate(energies))])
        # Histogram of costs (real distribution shape, not bar-by-bitstring).
        return _matplot_hist(
            title=f"Trial {trial_id} — feasible-shell cost distribution ({len(energies)} portfolios)",
            x_label="QUBO cost", y_label="count",
            values=energies, fmt=fmt,
        )

    raise ValueError(f"trial {trial_id} has no histogram-able data (no top_bitstrings or energy_distribution)")


def _plot_landscape(trial_id: int, fmt: str):
    rt = rebuild_qaoa_trial(trial_id)
    grid = landscape_p1(rt.problem, mixer=rt.mixer, init_state=rt.init_state,
                        n_gamma=41, n_beta=41)
    Z = np.array(grid["energy"])
    gammas = np.array(grid["gamma"])
    betas = np.array(grid["beta"])
    if fmt == "csv":
        rows = [("gamma", "beta", "energy")]
        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                rows.append((float(g), float(b), float(Z[i, j])))
        return _csv_rows(rows)
    return _matplot_heatmap(
        title=f"Trial {trial_id} — ⟨H_C⟩ landscape (p=1)",
        x_label="β", y_label="γ",
        x=betas, y=gammas, Z=Z, fmt=fmt,
    )


def _plot_comparison(trial_id: int, fmt: str):
    """Compare *this* trial's cost against fresh brute + SA on the same problem.

    Works for every trial kind — we rebuild the PortfolioProblem from the
    stored params, then run the two reference solvers on it. The trial's own
    cost (whatever solver produced it) is plotted alongside as a third bar.
    """
    trial = load_trial(trial_id)
    problem, _mode = build_portfolio_problem(trial["params"])

    bf = brute_force(problem)
    sa = simulated_annealing(problem, n_steps=2000, T0=1.0, T1=1e-3, seed=0)

    own_cost = trial["results"].get("cost")
    if own_cost is None:
        own_cost = trial["results"].get("energy_star", float("nan"))

    labels = ["brute_force", "simulated_annealing", f"this trial ({trial['kind']})"]
    values = [float(bf.cost), float(sa.cost), float(own_cost)]
    if fmt == "csv":
        return _csv_rows([("solver", "cost"), *zip(labels, values)])
    return _matplot_bar(
        title=f"Trial {trial_id} — solver comparison",
        x_label="solver", y_label="QUBO cost (lower is better)",
        labels=labels, values=values, fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _matplot_lines(title, x_label, y_label, traces, fmt):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, h in enumerate(traces):
        ax.plot(h, label=f"restart {i}")
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    if traces:
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, fmt)


def _matplot_hist(title, x_label, y_label, values, fmt, bins: int = 32):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color="#1f8a8a", edgecolor="white", linewidth=0.5)
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    fig.tight_layout()
    return _save(fig, fmt)


def _matplot_bar(title, x_label, y_label, labels, values, fmt):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    fig.tight_layout()
    return _save(fig, fmt)


def _matplot_heatmap(title, x_label, y_label, x, y, Z, fmt):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
    )
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    fig.colorbar(im, ax=ax, label="⟨H_C⟩")
    fig.tight_layout()
    return _save(fig, fmt)


def _save(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def _csv_trajectories(traces: list[list[float]]) -> str:
    sio = io.StringIO()
    w = csv.writer(sio)
    n = max((len(h) for h in traces), default=0)
    w.writerow(["iter"] + [f"restart_{i}" for i in range(len(traces))])
    for k in range(n):
        row = [k]
        for h in traces:
            row.append(h[k] if k < len(h) else "")
        w.writerow(row)
    return sio.getvalue()


def _csv_rows(rows) -> str:
    sio = io.StringIO()
    w = csv.writer(sio)
    for row in rows:
        w.writerow(row)
    return sio.getvalue()


# ---------------------------------------------------------------------------
# 5.6 — Bundle (zip)
# ---------------------------------------------------------------------------


def export_bundle(trial_id: int) -> bytes:
    import zipfile

    trial = load_trial(trial_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"trial_{trial_id}.json", json.dumps(trial, indent=2, default=_json_default))

        if trial["kind"] in {"qaoa_sim", "qaoa_hw"}:
            try:
                zf.writestr(f"qaoa_trial_{trial_id}.qasm", export_qasm3(trial_id))
            except Exception:
                pass
            zf.writestr(f"qaoa_trial_{trial_id}.py", export_qiskit_script(trial_id))
            zf.writestr(f"qaoa_trial_{trial_id}_pennylane.py", export_pennylane_script(trial_id))
            for tag, transpiled in (("pre", False), ("post", True)):
                try:
                    zf.writestr(f"circuit_{tag}.svg", render_circuit(trial_id, "svg", transpiled))
                    zf.writestr(f"circuit_{tag}.png", render_circuit(trial_id, "png", transpiled))
                except Exception:
                    pass

        for plot_kind in ("trajectory", "histogram", "landscape", "comparison"):
            for fmt in ("svg", "png", "csv"):
                try:
                    out = render_plot(trial_id, plot_kind, fmt)
                    name = f"plots/{plot_kind}.{fmt}"
                    zf.writestr(name, out if isinstance(out, (bytes, bytearray)) else out.encode())
                except Exception:
                    continue

    return buf.getvalue()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
