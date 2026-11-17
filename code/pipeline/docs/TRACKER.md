# Pipeline tracker

Production pipeline, built one step at a time; each decision below was made deliberately. It replaces the archived one-shot PoC in `code/claude/pipeline/`.

## How to run

```bash
cd code
.venv/bin/python pipeline/generate_data.py              # uses pipeline/config.json, writes pipeline/data/<YYYY-MM-DD_HHMMSS>/
```

- Each run writes a new versioned folder `data/<YYYY-MM-DD_HHMMSS>/` (local time). Runs never overwrite each other, and each folder's `manifest.json` records the config that produced it.
- `data/` is gitignored. A run is regenerable from its seed in about 4 s and takes about 11 MB (compressed npz; ~8× larger once loaded, since each bit is a byte).
- Identical output on rerun was verified (same Stim version and machine). Stim only guarantees identical shots for the same seed on the same version and machine.

## v1.0 — current

### Decisions

| # | Decision | Value | Why |
|---|---|---|---|
| 1 | Noise model | uniform circuit-level depolarizing: all four Stim generator channels at p | Simplest stochastic baseline. Matches the existing MWPM ε baseline (`research_log/README.md` settled #4). SI1000 deferred (v1.4). |
| 2 | p | 1e-3, fixed | Fixed p for v1 |
| 3 | Code | rotated surface code, d = 3, memory-Z only | MVP. X basis is v1.1 |
| 4 | Sampling | from the decomposed DEM (`decompose_errors=True`), the same DEM PyMatching uses | Checked at r=10 with 2M shots: circuit and DEM samplers agree within noise (≤1.7σ) |
| 5 | Round counts | every r in [3, 25], all splits (2026-11-17; was [3, 30] + [31, 100]) | Fixed-r training (the PoC) doesn't test generalization across lengths. Cut to ≤25 because 25–100 rounds is impractical for the tiny transformer; matches AQ1 pretraining lengths ({1, 3, …, 25}, `alphaqubit_1.pdf` §A.6.1). Longer r moves to the length-generalization test (1.4) |
| 6 | Train/val mix | ~~80% short / 20% long~~ removed with the buckets (2026-11-17) | Single range, so equal weight per r |
| 7 | Allocation over r | stratified: fixed equal count per r (train ≈39.1k, val ≈4.35k, test ≈8.7k per r) | Each r is its own circuit anyway; no sampling noise in the mix |
| 8 | Train / val size | 1e6 total; val = 10% (1e5), train = 9e5 | Val has its own seed, which is equivalent to a random held-out split and avoids splitting arrays of different lengths |
| 9 | Test | 2e5 shots, **even over every r in 3..25** (≈8.7k per r) | Per-r error bars |
| 10 | Empty shots | **train only**: keep each empty shot with probability 1/3 (`train_empty_keep_fraction`). Val and test keep the natural distribution | The drop depends on the syndrome only, so P(label \| syndrome) is unchanged. Dropping from val/test would bias the reported LER |
| 11 | Seeds | one RNG stream per (seed, split, r) | Splits independent; any (split, r) file can be regenerated on its own |
| 12 | Ground-truth errors | not stored | Seeded generation lets us regenerate the exact shots with `return_errors=True` when v2 needs per-window labels |
| 13 | Config | `config.json`; unknown or missing keys are errors | Kept from the PoC: typos fail loudly |

### Data layout

```
data/<YYYY-MM-DD_HHMMSS>/     one folder per run
├── manifest.json        config, Stim/NumPy versions, per (split, r): shots, num_detectors, label_rate,
│                        empty_fraction, empty_fraction_before_drop
├── circuits/r{r:03d}.stim, .dem    the circuit and decomposed DEM for each round count
└── {train,val,test}/r{r:03d}.npz   dets [n, D] uint8, obs [n, 1] uint8, rounds, num_detectors
```

`dets` and `obs` are raw 0/1 (not bit-packed), ready to feed the model:

```python
z = np.load("data/2026-11-11_204554/test/r100.npz")
dets, obs = z["dets"], z["obs"]   # uint8 0/1, [n, D] and [n, 1]
```

Detectors are in Stim's order. Their (x, y, t) are in the run's `circuits/r{r:03d}.stim` via `get_detector_coordinates()`.

### Generated: run `2026-11-11_204554` (Stim 1.16.0)

| split | shots | empty | label = 1 |
|---|---|---|---|
| train | 900,000 | 19.7% (after drop) | 15.10% |
| val | 100,000 | 37.0% | 12.47% |
| test | 200,000 | 16.3% | 21.23% |

Empty fraction in train, before → after the drop: r=3: 84.2% → 64.0%, r=10: 58.0% → 31.5%, r=30: 20.0% → 7.7%, r=100: 0.5% → 0.2%.

### Open in v1.0

- [ ] Empty-shot policy. A uniform keep of 1/3 still leaves r=3 at 64% empty. The alternative is a target empty fraction per r. Decide after seeing training behavior.
- [ ] MWPM baseline on the test split: LER per r (or binned), plus an ε-per-round fit.
- [ ] Loader and tokenization. The loader takes a run folder; decide whether it defaults to the newest one. The time encoding must not depend on total length: not t / rounds as in the PoC, but absolute t or relative (RoPE).
- [ ] Model v1: transformer over the full variable-length history (padding + mask), then training and evaluation.

## v1.x — extensions inside the v1 architecture

- **1.1 X basis**: generate memory-X too, and add a basis flag (or rotate coordinates). Stim's CX ordering breaks the X/Z symmetry slightly, so X can't be assumed identical to Z.
- **1.2 Noise range**: train over a range of p, with p (or a noise prior) as a feature. AQ2 is robust across a 16× range of noise strength (`project_docs_context/decoder_ML.md:109`).
- **1.3 SI1000**: rates per operation from Gidney et al. (2021) Table 2, on a CZ-native circuit, checked against a reference implementation before use.
- **1.4 Length generalization test**: train ≤25 rounds, test on 100–1000+ (AQ1 held from ≤25 to 10⁵; AQ2 from ≤168 to 10⁶).
  - not exactly this. train <= 25 rounds for v1; for long horizon we aren't predicting entire blocks of data at once, we would be maintaining a rolling context window and streaming data into the model, updating a pauli frame to keep track of errors. this is proper long horizon architecture instead of just dumping 10^5 shots into transformer. 3-25 is just for our baseline

## v2 — rolling context window (architectural change)

A streaming decoder that takes rounds as they arrive. Attention runs over a window that is capped in length, and old rounds are evicted, with no mandated recurrence.

- Prior art: sliding-window QEC decoding (Dennis et al. 2002; Skoric et al. 2023; Tan et al. 2023), and LLM KV-cache eviction (StreamingLLM, H2O) and sliding-window attention.
- Open design questions:
  - what is committed or carried forward when rounds are evicted (a frame update, or a summary token, which amounts to recurrence);
  - where per-window labels come from (regenerate with `return_errors=True`, decision 12).
- RoPE in time keeps cached keys valid as the window slides.
