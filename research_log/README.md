# Research log

What has been learned and settled across sessions, so each session can pick up where the last one stopped. Read this first, then the entry relevant to the question at hand.

## Entries

| entry | topic | dates |
|---|---|---|
| [mwpm_headroom.md](mwpm_headroom.md) | How far MWPM is from the optimal decoder at d=3 (exact numbers), why the old baseline was a coin flip, and how to frame "beat MWPM" | 2026-10-08 |
| [mwpm_gap_derivation.md](mwpm_gap_derivation.md) | How PyMatching builds its graph, what decomposing hyperedges throws away, and an algebraic proof of why MWPM picks the wrong class | 2026-10-08 → 2026-10-13 |

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
4. **Real data is where ML decoders show large wins** (on Sycamore: PyMatching 4.0% vs AlphaQubit 2.9%).
   - Effects like leakage are correlations over time that no DEM and no matching can represent.

## Open questions and next steps

- [ ] Sharpen the capstone goal with the professor. Proposed wording:
  - relative LER at fixed (d, p, r), with error bars;
  - plus the fraction of the MWPM-to-optimal gap closed at d=3;
  - plus Sycamore data as a stretch goal.
- [ ] Measure how much of the d=3 gap is the correlation term and how much is degeneracy, summed over all syndromes. So far this has only been done for the single worst syndrome.
- [ ] Decide whether to give the model input features that carry the s_z → s_x coupling, e.g. a co-located X/Z detector pairing, or whether to let it learn that.
- [ ] Sycamore data exploration (Zenodo 6804040): pretrain on SI1000 or a DEM fitted with p_ij, then finetune.
- [ ] Add correlated matching (two-pass reweighting) as a stronger baseline.

## Conventions

- Notation follows `mwpm_gap_derivation.md`:
  - H′: true signatures
  - H_G: graph (edge) signatures
  - D: decomposition map
  - Q = D_*P: true distribution pushed forward to graph space
  - P_G: product of Q's marginals
- Claude's code is in `code/claude/`. The team's code is in `code/`.
