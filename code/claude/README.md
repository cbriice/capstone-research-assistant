# code/claude/

Scripts written by Claude (research assistant), kept separate from team-authored code.

- `exact_decoding.py`: computes, with no sampling, the optimal decoder's logical error rate and PyMatching's exact error rate for small circuits (d=3, rounds ≤ 3). It also has noise-variant helpers: per-qubit rate scaling and a toy crosstalk model.
- `run_experiments.py`: reproduces the headroom tables. Run it from `code/` with `.venv/bin/python claude/run_experiments.py`. It takes about 3 min and needs ~1 GB of RAM.

The method: XOR-convolve each DEM error mechanism into the joint distribution P(s, L) over all 2^25 outcomes. The optimal decoder's LER is Σ_s min(P(s,0), P(s,1)).

**Terms in the output:**
- "MWPM-oracle" means PyMatching built from the true (noisy) DEM.
- "MWPM-nominal" means PyMatching built from the uniform-noise DEM.

## Results (2026-10-08)

| noise (p=1e-3 unless stated) | MWPM ÷ optimal |
|---|---|
| uniform, p=1e-3 / 5e-3 | 1.26× / 1.19× |
| per-qubit log-normal rates, σ=0.5 to 1.0 | 1.24–1.30× (no meaningful change) |
| + toy crosstalk, p_ct=5e-4 to 2e-3 | 1.03–1.60× (erratic; depends on the invented model) |

Validation: the exact MWPM LER at p=5e-3 is 1.718%, which agrees with 1.715% ± 0.009% from 2M sampled shots.
