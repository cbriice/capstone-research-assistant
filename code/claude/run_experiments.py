"""Reproduce the MWPM-vs-optimal headroom tables (d=3, rounds=3, exact, no sampling).

Run from code/:  .venv/bin/python claude/run_experiments.py
Each line takes ~15 s and ~1 GB RAM.
"""
import numpy as np

from exact_decoding import (
    exact_joint,
    mwpm_ler,
    optimal_ler,
    per_qubit_noise,
    surface_circuit,
    with_crosstalk,
)


def report(label, true_circ, nominal_circ):
    P, n = exact_joint(true_circ.detector_error_model(decompose_errors=False))
    opt = optimal_ler(P)
    oracle = mwpm_ler(P, n, true_circ.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True))
    nominal = mwpm_ler(P, n, nominal_circ.detector_error_model(decompose_errors=True))
    print(
        f"{label:<28} optimal={opt:.3e}  MWPM-oracle={oracle:.3e} ({oracle / opt:.2f}x)"
        f"  MWPM-nominal={nominal:.3e} ({nominal / opt:.2f}x)",
        flush=True,
    )


if __name__ == "__main__":
    print("== 1. uniform noise: headroom vs p ==")
    for p in (1e-3, 2e-3, 3e-3, 5e-3):
        c = surface_circuit(p)
        report(f"uniform p={p:.0e}", c, c)

    p0 = 1e-3
    base = surface_circuit(p0)

    print("== 2. per-qubit log-normal rate variation (mean rate fixed) ==")
    for sigma in (0.5, 1.0):
        for seed in (0, 1):
            mult = np.random.default_rng(seed).lognormal(0, sigma, size=base.num_qubits)
            mult /= mult.mean()
            report(f"sigma={sigma} seed={seed}", per_qubit_noise(base, mult), base)

    print("== 3. toy nearest-neighbour crosstalk ==")
    for p_ct in (5e-4, 1e-3, 2e-3):
        report(f"crosstalk p_ct={p_ct:.0e}", with_crosstalk(base, p_ct), base)
