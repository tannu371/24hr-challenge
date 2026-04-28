"""Hybrid quantum-classical portfolio optimization framework.

Modules
-------
data         Synthetic asset universe generation with controllable correlations.
formulation  Mean-variance / multi-objective model and QUBO + Ising mapping.
classical    Exhaustive enumeration, greedy heuristic, simulated annealing,
             continuous (relaxed) convex baseline.
quantum      Hand-rolled QAOA built on Qiskit primitives, plus hooks for the
             qiskit-algorithms QAOA reference.
analysis     Comparison utilities and metrics.
"""

__all__ = ["data", "formulation", "classical", "quantum", "analysis"]
