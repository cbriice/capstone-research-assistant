# code/claude/

Scripts written by Claude (research assistant), kept separate from team-authored code.

- `exact_decoding.py`: computes, with no sampling, the optimal decoder's logical error rate and PyMatching's exact error rate for small circuits (d=3, rounds ≤ 3). It also has noise-variant helpers: per-qubit rate scaling and a toy crosstalk model.
- `run_experiments.py`: reproduces the headroom tables. Run it from `code/` with `.venv/bin/python claude/run_experiments.py`. It takes about 3 min and needs ~1 GB of RAM.
- `kernel_dims.py`: GF(2) kernel dimensions of the true H′, the graph H_G and the decomposition map D (H′ = H_G D). See `research_log/mwpm_gap_derivation.md` §3.
- `worst_syndrome.py`: breaks down the syndrome where MWPM loses most to optimal: the exact P(s, L), the true vs graph costs of the leading explanations, and the DEM lines feeding each matched edge. See §4.
- `mwpm_runtime.py`: PyMatching time per shot vs number of defects on random circuit noise (d=3, 11), plus an adversarial contiguous defect block (~k² even, ~k³ odd). See `research_log/decoder_complexity.md` §3. It takes about 3 min.
- `build_schedule.py`: generates `deliverables/QEC_Project_Schedule.xlsx` (WBS, Milestones, Gantt, Notes) from the task list hard-coded at the top. It needs `openpyxl`. Edit the task list, then run `python build_schedule.py deliverables/QEC_Project_Schedule.xlsx`. The Gantt bars are plain cell fills, so they render in Numbers as well as Excel.

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
