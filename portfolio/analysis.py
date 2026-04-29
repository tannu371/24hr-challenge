"""Comparative analysis utilities and plotting.

The functions here glue the classical baselines and the QAOA results into
a single comparison table and produce the figures used in the research note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .formulation import PortfolioProblem, evaluate, build_qubo, qubo_cost


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    name: str
    cost: float
    K: int
    port_return: float
    port_vol: float
    diag_variance_proxy: float        # V̂(x) — deck term ⑤ scaled diag-var
    feasible_card: bool
    feasible_risk: bool
    wall_time: float
    n_evaluations: int


def summarise(problem: PortfolioProblem, results: List) -> List[ComparisonRow]:
    rows = []
    for r in results:
        m = evaluate(problem, r.x)
        rows.append(
            ComparisonRow(
                name=r.name,
                cost=r.cost,
                K=m["K"],
                port_return=m["port_return"],
                port_vol=m["port_vol"],
                diag_variance_proxy=m["diag_variance_proxy"],
                feasible_card=m["feasible_card"],
                feasible_risk=m["feasible_risk"],
                wall_time=r.wall_time,
                n_evaluations=r.n_evaluations,
            )
        )
    return rows


def format_table(rows: List[ComparisonRow]) -> str:
    """Plain-text comparison table."""
    header = (
        f"{'method':<25s} {'cost':>9s} {'K':>3s} {'ret':>7s} {'vol':>7s} "
        f"{'V̂':>7s} {'feasK':>5s} {'feasR':>5s} "
        f"{'time(s)':>8s} {'#eval':>7s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.name:<25s} {r.cost:9.4f} {r.K:3d} {r.port_return:7.3f} "
            f"{r.port_vol:7.3f} {r.diag_variance_proxy:7.4f} "
            f"{str(r.feasible_card):>5s} {str(r.feasible_risk):>5s} "
            f"{r.wall_time:8.3f} {r.n_evaluations:7d}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(
    histories: dict, optimum: Optional[float] = None, savepath: str = None,
    title: str = "Convergence",
):
    """Plot per-iteration cost trajectories for any solvers that expose history."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, h in histories.items():
        if h is None:
            continue
        ax.plot(np.arange(1, len(h) + 1), h, label=name, linewidth=1.6)
    if optimum is not None:
        ax.axhline(optimum, color="k", linestyle="--", linewidth=1.0,
                   label=f"global optimum = {optimum:.3f}")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("cost")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    return fig


def plot_qaoa_distribution(
    problem: PortfolioProblem, qaoa_result, optimum_x: np.ndarray,
    top: int = 32, savepath: str = None,
):
    """Bar plot of the top-K most probable bitstrings under the QAOA distribution,
    coloured by whether they are feasible / optimal."""
    import matplotlib.pyplot as plt

    n = problem.universe.n_assets
    Q, c = build_qubo(problem)
    optimum_int = int(sum(int(b) << i for i, b in enumerate(optimum_x)))
    optimum_cost = qubo_cost(Q, c, optimum_x)

    probs = qaoa_result.extra["probs"]
    top_idx = np.argsort(-probs)[:top]

    bitstrings, ps, costs, is_opt = [], [], [], []
    for k in top_idx:
        x = np.array([(int(k) >> i) & 1 for i in range(n)], dtype=int)
        bitstrings.append("".join(str(b) for b in x[::-1]))
        ps.append(probs[int(k)])
        costs.append(qubo_cost(Q, c, x))
        is_opt.append(abs(qubo_cost(Q, c, x) - optimum_cost) < 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#d62728" if o else "#1f77b4" for o in is_opt]
    ax.bar(range(len(ps)), ps, color=colors)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(bitstrings, rotation=90, fontsize=7, family="monospace")
    ax.set_ylabel("P(bitstring)")
    ax.set_title(f"QAOA output distribution (top {top}); red = optimal")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    return fig


def plot_scaling(
    sizes, time_classical, time_qaoa, qaoa_label="QAOA p=2", savepath: str = None,
):
    """Wall-time vs. problem size for classical vs. QAOA."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sizes, time_classical, "o-", label="brute force (K-shell)")
    ax.plot(sizes, time_qaoa, "s-", label=qaoa_label)
    ax.set_xlabel("number of assets N")
    ax.set_ylabel("wall time (s)")
    ax.set_yscale("log")
    ax.set_title("Scalability: wall time vs problem size")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    return fig


def plot_efficient_frontier(
    problem: PortfolioProblem, results: dict, savepath: str = None,
):
    """Plot every solver's portfolio in (vol, return) space against the
    K-shell brute-force enumeration of all feasible portfolios."""
    import matplotlib.pyplot as plt
    import itertools as it

    u = problem.universe
    n, K = u.n_assets, problem.K_target

    # Cloud of all cardinality-K portfolios (feasible).
    rets, vols = [], []
    for combo in it.combinations(range(n), K):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        w = x / K
        rets.append(float(u.mu @ w))
        vols.append(float(np.sqrt(w @ u.sigma @ w)))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(vols, rets, s=10, alpha=0.25, color="gray", label="all C(N,K)")

    markers = it.cycle(["o", "s", "^", "v", "D", "P", "X", "*"])
    for name, r in results.items():
        m = evaluate(problem, r.x)
        ax.scatter(m["port_vol"], m["port_return"], s=110,
                   marker=next(markers), edgecolor="k", label=name)
    ax.set_xlabel("portfolio volatility")
    ax.set_ylabel("expected return")
    ax.set_title(f"Risk-return scatter (N={n}, K={K})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    return fig
