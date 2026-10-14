# MWPM headroom at d=3 (2026-10-08)

## The coin-flip baseline was not a bug

The original `code/mwpm.ipynb` ran 25 rounds at p up to 5e-2. That is above threshold, and errors compound over rounds:

  LER(r) = ½ (1 − (1 − 2ε)^r),  where ε is the logical error rate per round

So the LER approaching 0.5 is expected. The y-axis is also mislabeled: it shows LER per shot, not per round. The notebook has since been re-run with realistic numbers.

The 1.78% figure is from `code/data_pipe.ipynb`:
- d=3, rounds=3, p=0.005 on all four Stim channels;
- the four channels are after_clifford, before_round_data, before_measure and after_reset;
- test LER 1.784% ± 0.059%.

## Per-round ε baseline over long runs (2026-10-14)

**Scope.** Decided with Dr. Kim: d=3, base experiments of 20–30 rounds, and a realistic target of 100+ rounds. At those lengths the per-shot LER tends to ½ for every decoder, so decoders are compared on ε.

**Method.** The code is the last five cells of `code/mwpm.ipynb`.
- Fit ln(1 − 2·LER) = ln A + r·ln(1 − 2ε) by weighted least squares over r ∈ {3, 5, 10, 25, 50, 100}, with binomial weights.
- A absorbs the state-preparation and readout boundaries.
- Setup: d=3 rotated memory-Z, uniform p on all four channels, PyMatching.
- Sampling: 1000 logical failures or 2M shots per point, whichever comes first.

| p | ε per round (MWPM) | A | LER at r=25 | LER at r=100 |
|---|---|---|---|---|
| 1e-3 | 2.40e-4 ± 0.04e-4 | 1.000 | 0.62% | 2.3% |
| 2e-3 | 9.26e-4 ± 0.08e-4 | 1.000 | 2.3% | 8.5% |
| 3e-3 | 2.02e-3 ± 0.01e-3 | 0.999 | 4.8% | 16.7% |
| 5e-3 | 5.34e-3 ± 0.03e-3 | 0.997 | 12.0% | 32.8% |

- **ε is flat in r.** The single-r estimate ½(1 − (1 − 2·LER)^(1/r)) matches the fit from r ≈ 10 on. At p=5e-3, r=3 reads about 8% high because of the boundary rounds. MWPM does not degrade on longer runs. The rise in per-shot LER is compounding that every decoder suffers.
- **ε ∝ p², as expected at d=3,** since (d+1)/2 = 2.
- **Cross-check:** at p=5e-3, r=3 the LER is 1.71%, against the exact 1.718% below.
- **Win condition:** a lower fitted ε at the same (d, p), with non-overlapping error bars, evaluated on the same r list using the same `fit_epsilon`.

## Exact optimal decoder (no sampling)

Code: `code/claude/exact_decoding.py`. Reproduce the tables with `code/claude/run_experiments.py`.

**Step 1.** Start with the full joint distribution P(s, L), represented as a vector over 2^(n+1) outcomes. Fold in each DEM mechanism j (probability p_j, signature mask_j):

  P ← (1 − p_j) P + p_j P[idx ⊕ mask_j]

This is how degeneracy is handled. You never list error patterns. Every combination of faults that produces the same (s, L) adds into the same entry automatically.

**Step 2.** Optimal LER = Σ_s min(P(s,0), P(s,1)).

**Step 3.** Exact MWPM LER: decode every syndrome with P(s) > 1e-14, then sum P(s, L ≠ MWPM(s)).

**Limit:** this needs 2^(n+1) entries, so it's only feasible for n + 1 ≤ ~30 bits. That means d=3 with rounds ≤ 3 (24 detectors). The run takes about 15 s and 1 GB.

**Validation:** at p=5e-3 the exact MWPM LER is 1.718%, against 1.715% ± 0.009% from 2M sampled shots.

## Results (d=3, r=3)

| noise | optimal LER | MWPM LER | MWPM ÷ optimal |
|---|---|---|---|
| uniform p=1e-3 | 6.101e-4 | 7.699e-4 | 1.26× |
| uniform p=2e-3 | | | 1.24× |
| uniform p=3e-3 | | | 1.22× |
| uniform p=5e-3 | 1.444e-2 | 1.718e-2 | 1.19× |
| per-qubit log-normal rates, σ=0.5–1.0, p=1e-3 | | | 1.24–1.30× |
| toy nearest-neighbour ZZ+XX crosstalk, p_ct=5e-4 to 2e-3 | | | 1.03–1.60× (erratic) |

- **Per-qubit rates:** MWPM is built from the true DEM, so rate variation by itself doesn't widen the gap. At d=3, this direction is a dead end.
- **Crosstalk:** this is an invented model, and the results depend on it. Don't cite them as a result.
- **Single round:** at r=1, p=1e-3, optimal is 2.862e-4 and MWPM is 2.956e-4 (1.03×). Most of the gap comes from multiple rounds, because correlations across rounds build up.

## Framing "beat MWPM"

Report the result in two parts:
- the relative LER against MWPM at fixed (d, p, r), with error bars;
- the fraction of the MWPM-to-optimal gap closed, (LER_MWPM − LER_model) / (LER_MWPM − LER_opt).

At d=3 the maximum possible improvement is a 16–21% reduction in LER.

**Real-data reference** (Sycamore d=3, from AlphaQubit 1, in `project_docs_context/decoder_ML.md`):

| decoder | LER |
|---|---|
| PyMatching | 4.015% |
| MWPM-BP | 3.117% |
| tensor network | 3.028% |
| AlphaQubit | 2.901% |

That gap is much larger than the gap in simulation. The recipe there is to pretrain on simulated data (SI1000, or a DEM fitted with p_ij; see `DEM_prior.md` Path B), then finetune on real data.
