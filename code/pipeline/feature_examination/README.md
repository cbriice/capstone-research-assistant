# feature_examination

Trains a small transformer decoder for the distance-3 rotated surface code (memory-Z, simulated with Stim,
uniform circuit-level noise p = 0.001) and finds out which token features and which training order work best.
Every model is scored on the same shots as two baselines:

- **always-0**: always guess "no logical error". A model that is not better than this has learned nothing.
- **MWPM**: minimum-weight perfect matching (PyMatching), the standard decoder. The target is to stay within
  2x MWPM's error per round.

## Files

| file | what it does |
|---|---|
| `config.py` | every setting, in `CONFIG`. `load_config()` checks the values and stops with a message if one is wrong |
| `generate_data.py` | samples the train / val / test detection events with Stim and scores always-0 and MWPM on them |
| `stim_preprocessing.py` | detection events -> tokens `[event, x, y, t, is_z, is_final, ...]`, one token per detector |
| `training_data.py` | loads a data folder and builds the batches on the GPU (or CPU), epoch by epoch |
| `train.py` | trains one model |
| `evaluate.py` | tests one trained model on the test split |
| `experiments.py` | the full study: every feature combination, then every training order, 3 seeds each (57 runs) |
| `show_results.py` | tables, `results.csv` and 4 graphs of an `experiments.py` study |
| `explore_results.py` | more graphs of a study: training loss, every seed, every round count, two studies compared |
| `plots.py` | the shared look of the graphs |
| `tests.py` | quick checks of the code and of your config (under a minute) |
| `requirements.txt` | the package versions this was tested with |

## 1. Install

Tested with Python 3.13.

```
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of torch first from https://pytorch.org/get-started/locally/
and then run the line above. Check that torch sees the GPU:

```
python -c "import torch; print(torch.cuda.is_available())"
```

`True` means training uses the GPU automatically. `False` means it runs on the CPU, which is much slower
(one d_model 16 run: about 7.5 min on an RTX 4060 Laptop GPU, about 75 min on a 4-core CPU).

## 2. Run the full study

Run the scripts from inside this folder. They also work from anywhere, because every path is taken relative to
the scripts themselves.

```
cd feature_examination
python tests.py                                   # under a minute: checks the code and config.py
python generate_data.py                           # under a minute: writes data/<date_time>/
python experiments.py                             # 57 training runs: writes experiments/<data folder name>/
python show_results.py <data folder name>         # e.g. python show_results.py 2026-11-17_183941
python explore_results.py <data folder name>      # more graphs, in experiments/<data folder name>/explore/
```

What `experiments.py` does:

1. **Feature study.** It trains every combination of the 4 feature groups in `FEATURE_GROUPS` (position = x, y;
   time = t; is_z; is_final), which is 16 combinations, each with 3 seeds. All use the base order: round counts
   mixed in every epoch, with the curriculum. The event bit is always in the token.
2. **Order study.** The combination with the lowest mean val LER is trained again in the 3 other orders
   (unmixed + curriculum, mixed without curriculum, unmixed without curriculum), 3 seeds each.
3. Every model is tested on the test split (r = 3 to 50). Nothing is chosen on test. At the end it runs
   `show_results.py`.

That is (16 + 3) x 3 = 57 runs. On an RTX 4060 Laptop GPU this took about 7.5 h at d_model 16 and 14.4 h at
d_model 32.

**Stopping and resuming.** You can stop it at any time (Ctrl+C, or close the laptop: sleep only pauses it).
Run `python experiments.py` again and it skips every finished run with the same settings and continues a stopped
run from its last finished epoch. Only the epoch that was in progress is lost.

**Seeds.** All seeds of a setting use the same data. The seed only changes the starting weights, the dropout
masks and the batch order. Some seeds get stuck near always-0, so each setting is trained 3 times and the graphs
use the middle seed.

## 3. Train and test one model

```
python train.py                                   # settings from config.py; tests.py runs first
python train.py --skip-tests                      # start without tests.py
python train.py --resume train_runs/<run>         # continue a stopped run from its last finished epoch
python evaluate.py [<run>]                        # test it (default: the newest folder in train_runs/)
python evaluate.py <run> --again                  # test it again
```

The test split is meant to be used once, at the end, so `evaluate.py` refuses to test a run twice unless you
add `--again`.

## 4. Change parameters

Everything is in `CONFIG` in `config.py`. The study itself is set at the top of `experiments.py`. The values below
are the ones the code ships with.

### Data: run `generate_data.py` again after changing these

| setting | shipped | meaning |
|---|---|---|
| `seed` | 1234 | data seed: one random stream per (seed, split, r) |
| `circuit.distance` | 3 | code distance (odd, at least 3). The study only ran at 3 |
| `circuit.basis` | `"z"` | memory experiment basis, `"z"` or `"x"` |
| `circuit.p` | 0.001 | uniform circuit-level depolarizing noise |
| `rounds.short_bucket` | [3, 15] | round counts trained and validated on... |
| `rounds.long_bucket` | [16, 30] | ...together with these: r = 3 to 30 |
| `rounds.long_fraction` | 0.2 | share of the training shots in the long bucket |
| `rounds.test_max` | 50 | test uses r = 3 to 50; above 30 the model has never seen that round count |
| `dataset.train_shots` | 900,000 | training shots in total |
| `dataset.train_empty_keep_fraction` | 1/3 | share of empty shots (nothing fired) kept in train; val and test keep all |
| `dataset.eval_mwpm_errors` | 150 | val and test get enough shots per r for about this many MWPM errors |
| `dataset.eval_min_shots`, `eval_max_shots` | 2,000, 300,000 | limits on those sizes |
| `dataset.pilot_shots` | 100,000 | shots per r of the pilot sample that measures MWPM's error rate first |

`train.py` and `experiments.py` use the newest folder in `data/` unless `training.run` names one.

### Tokens

| setting | shipped | meaning |
|---|---|---|
| `tokens.features` | `["x", "y", "t", "is_z", "is_final"]` | what each token carries after the event bit. Any subset of `x, y, t, is_z, is_final, is_first, sincos`, in that order |
| `tokens.sincos_periods` | [8, 64] | periods of the sin/cos time features (only used with `"sincos"`) |

x, y = the detector's position; t = round index / r; is_z = Z stabilizer; is_final = the last time slice (made
from the final data-qubit measurement); is_first = the first time slice. `experiments.py` sets `tokens.features`
itself for every combination.

### Model

| setting | shipped | meaning |
|---|---|---|
| `training.encoder.d_model` | 16 | width of every token's vector |
| `training.encoder.nhead` | 1 | attention heads (d_model must be divisible by nhead) |
| `training.encoder.dim_feedforward` | 64 | hidden size of each layer's MLP |
| `training.encoder.dropout` | 0.05 | dropout |
| `training.encoder.activation` | `"gelu"` | `"gelu"` or `"relu"` |
| `training.num_layers` | 2 | transformer layers |
| `training.pooling` | `"mean"` | `"mean"` of the token outputs, or a learned `"cls"` token |

The shipped model has 6,721 parameters. The larger model of the second study used `d_model` 32, `nhead` 4 and
`dim_feedforward` 128 (25,729 parameters) and nothing else changed.

### Training

| setting | shipped | meaning |
|---|---|---|
| `training.run` | `None` | name of the `data/` folder to train on; `None` = the newest |
| `training.seed` | 0 | starting weights, dropout and batch order |
| `training.num_epochs` | 10 | epochs; the epoch with the lowest val LER is kept |
| `training.lr` | 2e-3 | peak learning rate (AdamW) |
| `training.lr_scheduler` | `"linear"` | `"linear"` or `"cosine"` decay to lr / 1000 |
| `training.warmup_ratio` | 0.1 | linear warmup over this share of all steps |
| `training.weight_decay` | 0.01 | AdamW weight decay |
| `training.grad_clip` | 1.0 | largest gradient norm in one update |
| `training.batch_size` | 256 | most shots in one batch |
| `training.max_tokens_per_batch` | 65,536 | cap on shots x detectors in one batch; lower it if the GPU runs out of memory |
| `training.curriculum` | [5, 10, 20] | longest r in epochs 1, 2 and 3; later epochs use every r. `[]` = off |
| `training.mix_rounds` | `True` | `True`: every epoch mixes batches of all r. `False`: one r at a time |
| `training.eval_every` | 100 | steps between quick val checks in the log |
| `training.eval_rounds` | [3, 10, 20, 30] | round counts of the quick checks |
| `training.eval_check_shots` | 2,000 | val shots per r in a quick check (every val shot is used after each epoch) |
| `evaluation.target_x_mwpm` | 2.0 | the target: error per round at most this many times MWPM's |

`experiments.py` sets `training.seed`, `mix_rounds` and `curriculum` itself for every run.

### The study: top of `experiments.py`

| setting | shipped | meaning |
|---|---|---|
| `FEATURE_GROUPS` | position, time, is_z, is_final | groups switched on and off together. Adding `"is_first": ["is_first"]` or `"sincos": ["sincos"]` doubles the number of runs |
| `SEEDS` | [0, 1, 2] | every setting is trained once per seed |
| `ORDERS` | 4 orders | name: (mix_rounds, curriculum on). The first one is the base order of the feature study |
| `TRAINING_CHANGES` | `{}` | changes to `CONFIG["training"]` for every run of the study, e.g. `{"num_epochs": 20}` |

`TRAINING_CHANGES` replaces whole entries, so to change the model there give the whole encoder:
`{"encoder": {"d_model": 32, "nhead": 4, "dim_feedforward": 128, "dropout": 0.05, "activation": "gelu",
"batch_first": True, "norm_first": True}}`.

### Before starting a study with new settings

`experiments.py` writes into `experiments/<data folder name>/`. A run whose settings changed is trained again
and its `result.json` is overwritten. To keep the old results, rename that folder first, for example to
`experiments/d_model_16`, and then start the new study.

`show_results.py` and `explore_results.py` without a folder name pick the last folder in alphabetical order
(not the newest), so always pass the folder name.

## 5. Outputs

```
data/<date_time>/                      about 13 MB
  circuits/rNNN.stim, rNNN.dem         circuit and detector error model of every round count r
  train/ val/ test/ rNNN.npz           detection events [shots, detectors] and labels [shots, 1], uint8 0/1
  manifest.json                        config, package versions, and per split and r: shots, always-0 and MWPM errors
train_runs/<date_time>/                about 1 MB (inside experiments/ for a study)
  train.log                            learning rate, training loss, quick checks, full val after every epoch
  history.json                         every logged number: loss every 100 steps, quick checks, val per epoch and r
  last.pt, best.pt, model.pt           last finished epoch (for --resume), best epoch, kept model + its settings
  test_results.json/.csv, *.png        written by evaluate.py
experiments/<data folder name>/
  setup.json                           the study's settings and the always-0 / MWPM errors of val and test
  <combination>__<order>__seed<k>/     one run: result.json and its train_runs-style folder
  results.csv, features.png, feature_effects.png, rounds.png, order.png       from show_results.py
  explore/                                                                    from explore_results.py
```

`explore_results.py <data folder name> --compare <other folder name>` also draws two studies next to each other
(for example d_model 32 against d_model 16): val of every combination, error per round of each study's best
setting, and training loss.

## 6. Reading the numbers

- **LER** (logical error rate): share of shots decoded wrong. Lower is better; accuracy = 1 - LER.
- **Val LER** and the LER in `show_results.py` and `explore_results.py` are means over the round counts, so
  every r counts the same. `evaluate.py`'s summary line pools all shots instead, which gives lower numbers.
  Don't compare the two directly.
- **Val** covers r = 3 to 30, the round counts the model was trained on. **Test** covers r = 3 to 50; r = 31 to
  50 were never seen in training and show how well the model generalizes to longer experiments.
- **Error per round** = (1 - (1 - 2 LER)^(1/r)) / 2. LER grows with r for every decoder, so round counts are
  compared by this number. The target is at most 2x MWPM's error per round at every r.
- **Middle seed**: the seed with the middle val LER of the 3. One stuck seed cannot pull it up.
- **Stuck**: val LER above half of always-0's. Usually a seed that sat on the always-0 plateau and never learned
  to use the token features.
- Val and test are sized for about 150 MWPM errors per r, so a single r's ratio to MWPM moves by about 8% (one
  standard deviation) from sampling noise alone. All models share the same test shots, so they all show the same
  zigzag. Read the trend over several r, not one point.

## 7. What to push

`data/` can be made again with `generate_data.py` in under a minute. Stim only promises the very same shots for
the same Stim version on the same kind of CPU, so push it (about 13 MB) if everyone must use exactly the same
shots. Each training folder is about 1 MB, so a whole study is about 60 MB. Leave out `__pycache__/`.
