"""Seed /artifacts with stand-in hardware snapshots for the demo.

Run from /backend with venv active:
    python -m scripts.seed_hw_cache

This builds N=8 and N=12 problems, runs QAOA on the Aer statevector to find
optimal θ, samples 4096 shots from the *exact* QAOA distribution, then perturbs
the empirical counts with a small amount of bit-flip noise to mimic device
readout error. The resulting blob has the same shape as a real /hw/job result
ingestion, so the frontend doesn't have to special-case the stand-ins.

Each snapshot is tagged ``meta.stand_in: true`` and ``meta.backend:
"aer_statevector_with_noise_stand_in"`` so judges can see at a glance that
these are synthetic placeholders.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_BACKEND = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.problem_builder import build_portfolio_problem, canonical_params
from app.services.qaoa import run_qaoa_optimisation, statevector_probabilities, build_qaoa_circuit
from app.services.hw import ingest_counts
from app.services.hw_cache import save_cached
from portfolio.classical import brute_force
from portfolio.formulation import build_qubo, qubo_to_ising


READOUT_BIT_FLIP_P = 0.015  # mimic ~1.5% per-qubit readout error on a current-gen device


def _make_stand_in(N: int, K: int, seed: int, p: int, mixer: str, init_state: str,
                   shots: int, name: str) -> dict:
    payload = {
        "N": N, "lambda": 2.5, "K": K, "P_K": 5.0, "P_R": 0.5, "seed": seed,
    }
    problem, mode = build_portfolio_problem(payload)

    bf = brute_force(problem)
    classical_opt = float(bf.cost)

    qres = run_qaoa_optimisation(
        problem=problem, p=p, mixer=mixer, init_state=init_state,
        optimizer="COBYLA", max_iter=300, n_restarts=8, seed=seed,
        classical_optimum=classical_opt,
    )

    Q, c = build_qubo(problem)
    ising = qubo_to_ising(Q, c)
    K_arg = problem.K_target if init_state == "dicke" else None
    circuit = build_qaoa_circuit(ising, p=p, mixer=mixer, init_state=init_state, K=K_arg)
    probs = statevector_probabilities(circuit, qres.theta_star)

    rng = np.random.default_rng(seed + 991)
    n = problem.universe.n_assets
    samples = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    bit_flip_mask = rng.random(size=(shots, n)) < READOUT_BIT_FLIP_P
    bits = np.array([[(int(s) >> i) & 1 for i in range(n)] for s in samples])
    bits = np.where(bit_flip_mask, 1 - bits, bits)

    counts: dict[str, int] = {}
    for row in bits:
        bitstr = "".join(str(int(b)) for b in reversed(row))  # big-endian as Qiskit prints
        counts[bitstr] = counts.get(bitstr, 0) + 1

    results = ingest_counts(
        problem=problem, counts=counts, p=p, mixer=mixer, init_state=init_state,
        theta=[float(t) for t in qres.theta_star],
        classical_optimum=classical_opt,
    )

    meta = {
        "stand_in": True,
        "backend": "aer_statevector_with_noise_stand_in",
        "shots": shots,
        "readout_bit_flip_p": READOUT_BIT_FLIP_P,
        "seed": seed,
        "qaoa_p": p,
        "mixer": mixer,
        "init_state": init_state,
        "note": "Synthetic stand-in for demo. Replace with a real hardware run via /hw/submit.",
    }
    params = canonical_params(payload, mode, problem) | {
        "p": p, "mixer": mixer, "init_state": init_state,
    }
    save_cached(name=name, params=params, results=results, meta=meta)
    print(f"  wrote hw_{name}.json — cost={results['cost']:.4f} approx_ratio={results['approx_ratio']:.3f}")
    return {"name": name, "results": results}


def main():
    print("Seeding /artifacts with stand-in hardware snapshots …")
    _make_stand_in(N=8, K=3, seed=42, p=2, mixer="x", init_state="uniform",
                   shots=4096, name="n8_seed42_p2_x")
    _make_stand_in(N=12, K=4, seed=7, p=3, mixer="x", init_state="uniform",
                   shots=4096, name="n12_seed7_p3_x")
    _make_stand_in(N=8, K=3, seed=42, p=2, mixer="xy_ring", init_state="dicke",
                   shots=4096, name="n8_seed42_p2_xy_dicke")
    print("Done. Replace these with real hardware runs as they come in.")


if __name__ == "__main__":
    main()
