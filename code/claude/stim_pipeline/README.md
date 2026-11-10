# Stim surface-code data and a PyMatching baseline

This first stage generates labeled data and checks an MWPM decoder. There is no
neural network yet. Your original `test_stim_data.py` is preserved unchanged as
a minimal ten-shot demonstration; the reproducible pipeline starts at
`run_experiment.py`.

**Yes: Stim generates the logical labels itself.** A circuit detector sampler
called with `separate_observables=True` returns both detector events `X` and
observable flips `y`, aligned shot by shot. PyMatching only predicts those flips.

## Run in the VS Code terminal on macOS

Open the repository in VS Code, then run these commands from the repository root:

```bash
cd code/claude/stim_pipeline

# Only create this environment if it does not already exist.
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install -r requirements.txt

# Meaningful checks, including a noiseless experiment and no split overlap.
python -m pytest -q

# Default experiment; use a NEW output directory each time.
python run_experiment.py --output-dir data
```

The environment is local and is not committed. In VS Code, run **Python: Select
Interpreter** from the Command Palette and select
`code/stim_pipeline/.venv/bin/python`. Run the commands above from
`code/claude/stim_pipeline/`. Python 3.9 or newer is required;
some Python/platform combinations may build Stim from source.

If `data/` already exists, the runner deliberately refuses to overwrite it:

```bash
python run_experiment.py --output-dir data_run2
```

`requirements.txt` specifies compatible package ranges. `requirements-lock.txt`
records the exact packages in the verified Python 3.9 environment. For a matching
Python/platform environment, install that file instead to reproduce versions:

```bash
python -m pip install -r requirements-lock.txt
```

## Files and parameters

| File | Purpose |
| --- | --- |
| `config.py` | Central defaults, seeds, split fractions, parameter validation |
| `data_pipeline.py` | Circuit generation, Stim sampling, splitting, NPZ saving/loading |
| `baseline.py` | Circuit-to-MWPM graph, predictions, metrics, saved-data evaluation |
| `run_experiment.py` | Small command-line runner tying these steps together |
| `tests/test_pipeline.py` | Provenance, dimensions, reproducibility, split and decoder checks |
| `test_stim_data.py` | Original example, unchanged |

Defaults are distance 3, rounds 3, physical error rate 0.001, 10,000 shots,
Stim seed 42, and split seed 43. Splits are 70% training, 15% validation, and
15% testing. Training and validation counts round down; test gets the remainder.

Edit `ExperimentConfig` or override common settings from the terminal:

```bash
python run_experiment.py --distance 5 --rounds 5 \
    --physical-error-rate 0.003 --shots 100000 \
    --seed 123 --split-seed 456 --output-dir data_d5
```

Here, `physical_error_rate` is applied separately to Clifford depolarization,
measurement flips, and reset flips. It is not the total error probability for
an entire circuit. Additional before-round data depolarization is left at zero.

## Where the labels and predictions come from

The essential data-generation code is:

```python
circuit = build_circuit(config)
sampler = circuit.compile_detector_sampler(seed=config.seed)
X, y = sampler.sample(shots=config.shots, separate_observables=True)
```

- `X[i, j]` is whether detector `j` fired in shot `i`; it is a detection event
  relative to the reference, not a raw measurement bit or a physical error list.
- `y[i, k]` is whether logical observable `k` flipped relative to the noiseless
  reference in that same shot. This circuit has one logical Z observable.
- A label of 1 does **not** by itself mean the decoder failed. The decoder can
  correctly predict that flip. Failure means its prediction differs from `y`.

Default shapes are `X.shape == (10000, 24)` and `y.shape == (10000, 1)`.
The arrays stay Boolean for storage and decoding; console examples show 0/1.
The runner prints training examples, including one with a flipped observable
when available. At low noise many examples legitimately contain only zeros.

The circuit is also used to create the decoder:

```python
dem = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(dem)
matching.ensure_num_fault_ids(circuit.num_observables)
predicted_obs = matching.decode_batch(X)
```

`ensure_num_fault_ids` retains the logical output column even in the zero-noise
sanity check. No labels are passed to the decoder. In the runner, `X` for decoding
comes directly from the saved validation file; no new shots are sampled.

The screenshot's DEM discussion describes another possible sampling route.
**This project samples the circuit**, while the DEM supplies the matching graph.
The graph is built from known circuit noise, without fitting validation or test
labels. Standard PyMatching MWPM is used with its default uncorrelated matching;
decomposing errors does not make MWPM an exact maximum-likelihood decoder for
all circuit correlations.

Observable arrays have shape `(shots, num_observables)`. The comparison helper
accepts a `(shots,)` single-observable vector by explicitly adding its column
axis, then checks that shapes match. It never relies on NumPy broadcasting:

```python
failed_shots = np.any(predicted_obs != y, axis=1)
logical_error_rate = np.count_nonzero(failed_shots) / len(y)
```

For multiple observables, any wrong bit makes that shot a failure. This rate is
per **complete memory experiment**, not per round. The report includes failure
count, sample count, error rate, and accuracy. With low noise and only 1,500
validation shots, estimates will be noisy; zero observed failures is not proof
of zero error probability. Increase shots for a precise research estimate.

## Saved data and an untouched test partition

```text
data/
    train.npz
    validation.npz
    test.npz
    circuit.stim
    detector_error_model.dem
    metadata.json
    mwpm_validation_predictions.npz
    mwpm_validation_metrics.json
```

Each dataset NPZ contains `X`, `y`, `sample_ids`, and `metadata` (a JSON string).
Metadata includes distance, rounds, noise settings, number of samples, circuit
type, seeds, split membership, package versions, and the circuit's SHA-256 hash.
Prediction files contain `predicted_obs` and aligned `sample_ids`, separately
from the ground-truth labels.

A seeded permutation partitions the original shot IDs once. No shot is copied
into two splits. Independent shots may naturally have identical bit patterns,
especially all-zero syndromes; these are not leakage and are not deduplicated.
Do not merge separately generated runs with the same sampling seed and settings
into a larger dataset: those runs can contain the exact same simulated shots.

The default runner saves the test partition but never displays its examples,
evaluates it, or uses its labels to choose anything. Train the future ML decoder
on `train.npz` and tune choices on `validation.npz`. Fit any learned preprocessing
only on training data. Keep the fixed test file for the final comparison.

Load the training data without pickle:

```python
from pathlib import Path
from data_pipeline import load_dataset

X_train, y_train, shot_ids, metadata = load_dataset(Path("data/train.npz"))
print(X_train.shape, y_train.shape)
# Convert to the ML framework's preferred dtype only when training.
```

Reevaluate the validation baseline without generating data:

```bash
python baseline.py --data-dir data --split validation
```

When the ML model and all research choices are frozen, evaluate MWPM on the
reserved test data (this writes separate test predictions/metrics):

```bash
python baseline.py --data-dir data --split test
```

Use **that same `test.npz`**, in its saved row order, for ML predictions and
`logical_metrics(y_test, ml_predictions)`. Never use MWPM predictions as ML
training labels. Avoid repeated test-driven tuning. Repeated baseline commands
replace their prediction/metric files but never modify the dataset NPZ files.

## Reproducibility and verification

Stim receives an explicit seed and samples all shots in one call. The independent
NumPy split seed determines membership without looking at labels. Metadata and
the saved circuit capture the experimental setup. Exact seeded Stim samples can
change with Stim version, machine SIMD support, or sampling call sizes; a seed
alone is not a portable dataset identifier. Reuse the saved NPZ files for exact
future comparisons, and keep their circuit and metadata together.

Tests check direct agreement with Stim's sampler, a deliberately injected error
with known label, batched versus single-shot MWPM decoding, zero-noise behavior,
multi-observable failure counting, lossless saving, reproducible/disjoint splits,
circuit mismatch detection, and protection against overwriting datasets. Tests
exercise final test evaluation only on disposable temporary datasets.
`pytest.ini` limits collection to `tests/`, so importing the original demonstration
is not part of the test suite.

The initial local verification passed **15 tests** with Stim 1.16.0, PyMatching
2.4.0, and NumPy 2.0.2. The initial local default experiment generated:
7,000 training, 1,500 validation, and 1,500 test shots. MWPM observed 0 failures
in the 1,500 validation shots; that local test partition remains unevaluated.
Generated datasets are not committed. Run the pipeline to create your own
`data/` directory; use a new output directory for subsequent experiments. The test run emitted
third-party Matplotlib/Pyparsing deprecation warnings, with no test failures.

API references:

- [Stim Python API: detector sampling and observable outputs](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.CompiledDetectorSampler.sample)
- [Stim Python API: sampler seed and reproducibility](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit.compile_detector_sampler)
- [PyMatching: detector-error-model construction](https://pymatching.readthedocs.io/en/stable/api.html#pymatching.Matching.from_detector_error_model)
- [PyMatching: batch decoding](https://pymatching.readthedocs.io/en/stable/api.html#pymatching.Matching.decode_batch)
