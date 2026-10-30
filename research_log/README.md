# Research log

What has been learned and settled across sessions, so each session can pick up where the last one stopped. Read this first, then the entry relevant to the question at hand.

## Entries

| entry | topic | dates |
|---|---|---|
| [mwpm_headroom.md](mwpm_headroom.md) | How far MWPM is from the optimal decoder at d=3 (exact numbers), why the old baseline was a coin flip, the per-round ε baseline over 3–100 rounds, and how to frame "beat MWPM" | 2026-10-08 → 2026-10-28 |
| [decoder_complexity.md](decoder_complexity.md) | Why matching needs graphlike edges (P vs NP-complete/#P-hard), BP's cost and failure mode, p → weights in general (static vs dynamic, ML decoders), MWPM runtime typical vs worst case, boundary vs bulk | 2026-10-20 → 2026-10-26 |
| [cryo_hardware_constraints.md](cryo_hardware_constraints.md) | How a ≤4 K FPGA/cryo placement bounds model size: stage cooling budgets, energy per MAC, power/latency/memory constraints → MACs per round, superconducting-architecture landscape (cryo-CMOS, cryo-FPGA, SFQ) | 2026-10-29 → |
| [mwpm_gap_derivation.md](mwpm_gap_derivation.md) | How PyMatching builds its graph, what decomposing hyperedges throws away, and an algebraic proof of why MWPM picks the wrong class | 2026-10-08 → 2026-10-20 |

## Settled conclusions

1. **MWPM can be beaten at d=3.**
   - With uniform noise, MWPM's LER is 1.19–1.26× the optimal decoder's.
   - So an exact decoder could cut the LER by 16–21%.
2. **Per-qubit rate variation doesn't widen the gap at d=3.**
   - MWPM built from the true DEM stays at 1.24–1.30×.
   - The gap comes from correlations, not from uneven error rates.
3. **The gap is correlations between the pieces of decomposed hyperedges.**
   - MWPM optimizes over the right set of corrections but with the wrong probability on it.
   - It uses the product of the edge marginals instead of their joint distribution, and it takes the max instead of summing.
   - In the code-capacity model the result is that MWPM's prediction depends on s_x alone, while the optimal one uses s_z through Y errors.
   - Summed over all syndromes at r=3, 92–96% of the gap is correlation and 4–8% is degeneracy. A hypergraph most-likely-error decoder is within 2% of optimal.
4. **Compare decoders on ε per round, not on per-shot LER.**
   - Scope: d=3, 20–30 rounds base, 100+ realistic.
   - MWPM's fitted ε is flat from r=3 to r=100. At p=1e-3 it is 2.40e-4.
   - The pipeline reproduces the Stim tutorial's PyMatching numbers (threshold ≈1% per round, d=5 ε at p=1e-3 within 3%). See `mwpm_headroom.md`.
   - The per-shot LER tends to ½ for every decoder, so a coin flip at long r says nothing about MWPM.
5. **Graphlike is the line between tractable and NP-complete.**
   - Min-weight decoding with column weight ≤ 2 reduces to perfect matching, which is in P. With weight-3 columns it is NP-complete (3DM), and exact ML is #P-hard.
   - BP is cheap but inexact on the surface code: its loops and degeneracy are the problem.
   - MWPM's runtime is near-linear in defects on random noise and O(k³) on adversarial syndromes. Real-time budgets care about the tail.
6. **Real data is where ML decoders show large wins** (on Sycamore: PyMatching 4.0% vs AlphaQubit 2.9%).
   - Effects like leakage are correlations over time that no DEM and no matching can represent.

## Open questions and next steps

- [ ] Sharpen the capstone goal with the professor. Proposed wording:
  - relative LER at fixed (d, p, r), with error bars;
  - plus the fraction of the MWPM-to-optimal gap closed at d=3;
  - plus Sycamore data as a stretch goal.
- [x] Split the d=3 gap over all syndromes (2026-10-20): 92–96% correlation, 4–8% degeneracy. See `mwpm_gap_derivation.md` §7.
- [ ] Decide whether to give the model input features that carry the s_z → s_x coupling, e.g. a co-located X/Z detector pairing, or whether to let it learn that.
- [ ] Sycamore data exploration (Zenodo 6804040): pretrain on SI1000 or a DEM fitted with p_ij, then finetune.
- [ ] Add correlated matching (two-pass reweighting) as a stronger baseline.
- [ ] Measure the MWPM-to-optimal ε headroom at r=25–100. The joint table stops at r≤3, but exact per-shot ML is cheap at d=3 for any r (`code/claude/exact_sweep.py`, `decoder_complexity.md` §4). Tesseract is only needed for d≥5.
- [ ] Ask Dr. Kim:
  - which FPGA board we can use (decides INT4 vs INT8, and FINN vs hls4ml);
  - whether latency means per-round streaming cost or reaction time after the last round.

## Conventions

- Notation follows `mwpm_gap_derivation.md`:
  - H′: true signatures
  - H_G: graph (edge) signatures
  - D: decomposition map
  - Q = D_*P: true distribution pushed forward to graph space
  - P_G: product of Q's marginals
- Claude's code is in `code/claude/`. The team's code is in `code/`.
