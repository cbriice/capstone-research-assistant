"""Pipeline settings. Everything lives in CONFIG below, and every script reads it through load_config(), which checks
the values and returns a copy, so a script that changes its copy does not change anyone else's.
The values follow the team's model requirements list of 2026-11-17 (section numbers in the comments)."""

import copy
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_DIR / "data"                # generate_data.py writes one data/<YYYY-MM-DD_HHMMSS>/ folder per run
TRAIN_DIR = PIPELINE_DIR / "train_runs"         # train.py writes one train_runs/<YYYY-MM-DD_HHMMSS>/ folder per run
EXPERIMENTS_DIR = PIPELINE_DIR / "experiments"  # experiments.py writes one experiments/<data run>/ folder

# every feature a token can carry after the event, in token order. "sincos" adds 2 numbers per period, the others 1
TOKEN_FEATURES = ["x", "y", "t", "is_z", "is_final", "is_first", "sincos"]

# ==================================== CONFIG ===============================================
CONFIG: dict = {
  "seed": 1234,                    # data generation: one RNG stream per (seed, split, r)
  "circuit": {                     # 1. scope
    "distance": 3,
    "basis": "z",                  # z | x
    "p": 0.001                     # uniform circuit-level depolarizing noise on all four Stim noise channels (no SI1000)
  },
  "rounds": {                      # 1. and 2.
    "short_bucket": [3, 15],       # train and val use every r of both buckets: 3-30
    "long_bucket": [16, 30],
    "long_fraction": 0.3,          # share of the training shots in the long bucket
    "test_max": 50,                # test uses r 3-50; 31-50 are never trained on (the generalization test)
  },
  "dataset": {                     # 2. data
    "train_shots": 900_000,
    "train_empty_keep_fraction": 1 / 3,  # shots where nothing fired kept in train; val and test keep every shot
    "eval_mwpm_errors": 150,       # val and test: shots per round count for about this many MWPM errors. 150, not 100, so
                                   # that the noise of the pilot estimate still leaves at least 100 in every r
    "eval_min_shots": 2_000,       #   but never fewer than this
    "eval_max_shots": 300_000,     #   or more than this
    "pilot_shots": 100_000,        # shots per round count sampled first to measure the MWPM error rate behind those sizes
  },
  "tokens": {                      # 3. input tokens
    "features": ["x", "y", "t", "is_z", "is_final", "is_first", "sincos"],   # after the event (always there); any subset of TOKEN_FEATURES, in that order
    # t = round index / r (relative time: every circuit spans 0..1). is_first = 1 on the first time slice.
    # sincos = sin and cos of 2 pi * round index / period, one pair per period: absolute time, so round 3 of 4 and
    # round 75 of 100 get different values (t / r gives both 0.75). Not used by default; experiments.py can test it.
    "sincos_periods": [8, 64],
  },
  "training": {                    # 4. model and 5. training
    "run": None,                   # name of the data/<run> folder to train on. None = the newest one
    "seed": 0,                     # starting weights, dropout and batch order: same seed, same run
    "encoder": {                   # nn.TransformerEncoderLayer(**encoder); head size is fixed at d_model / nhead
      "d_model": 32,
      "nhead": 4,
      "dim_feedforward": 128,
      "dropout": 0.05,
      "activation": "gelu",        # relu | gelu
      "batch_first": True,         # tokens are (shots, detectors, features). torch should not auto batch our data
      "norm_first": True,          # LayerNorm before attention and before the MLP (pre-LN); train.py adds the final LayerNorm
    },
    "num_layers": 2,               # nn.TransformerEncoder(layer, num_layers)
    "pooling": "mean",             # mean: g = 1/N * sum(e_L) | cls: a learned summary token, g = its output
    "batch_size": 256,             # most shots in one batch
    "max_tokens_per_batch": 65536, # cap on shots x detectors; 256 shots at r=30 are 61,440 tokens, so it only binds above r=32
    "curriculum": [5, 10, 20],     # longest r in epochs 1, 2 and 3; later epochs use every r (up to 30). [] = off
    "mix_rounds": True,            # True: every epoch mixes batches of all r. False: one r at a time, r order shuffled per epoch
    "warmup_ratio": 0.1,           # linear warmup from lr / 1000 over the first 10% of all steps
    "lr_scheduler": "linear",      # linear: down to lr / 1000 at the last step | cosine
    "num_epochs": 10,
    "lr": 2e-3,                    # peak learning rate
    "weight_decay": 0.01,          # L2 regularization term for AdamW applied to each theta directly
    "grad_clip": 1.0,              # largest gradient norm allowed in one update
    "eval_every": 100,             # steps between quick val checks in the log
    "eval_rounds": [3, 10, 20, 30],  # round counts of the quick check
    "eval_check_shots": 2_000,     # val shots per round count in a quick check; every val shot is used after each epoch
  },
  "evaluation": {                  # 6. evaluation
    "target_x_mwpm": 2.0,          # target: per-round error at most this many times MWPM's, at every tested r
  },
}

"""
CONFIG NOTES:
z = w^Tg + b, g = 1/N * sum(LN(e_L)); z is per-basis.
- training trials start with 1 head memory Z for simplicity. in practice we have >=2 heads, one for memory X and one for memory Z.
- consequence of this: with two heads for z and x, d across board should probably increase. need to understand mathematically whether enough
  unique information is stored in each z and x embedding that d_model *= 2 makes sense or if u can be fine with some overlap
- setting d_k higher than d_model/nhead gains no expressive power because rank(Ah) <= d_model. more params, same function class, no extra value.
    - useful when d_k < d/n_h (low-rank sim score while preserving residual stream information; d >= d_k > d/n_h (many heads, d/n_h too small)
    - torch.nn.TransformerEncoder doesnt take d_k as param so for now we just leave it as d/n_h
"""
# ============================================================================================


def train_rounds(cfg: dict) -> list[int]:
    # round counts of the train and val splits: both buckets
    return list(range(cfg["rounds"]["short_bucket"][0], cfg["rounds"]["long_bucket"][1] + 1))


def test_rounds(cfg: dict) -> list[int]:
    # round counts of the test split: the training range and beyond, up to test_max
    return list(range(cfg["rounds"]["short_bucket"][0], cfg["rounds"]["test_max"] + 1))


def _check(ok: bool, message: str) -> None:
    # an explicit check instead of assert, because python -O removes asserts
    if not ok:
        raise ValueError(message)


def load_config() -> dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)    # a copy, so changes made by one script don't leak into CONFIG
    check_config(cfg)
    return cfg


def check_config(cfg: dict) -> None:
    c, r, d, k, t = cfg["circuit"], cfg["rounds"], cfg["dataset"], cfg["tokens"], cfg["training"]
    _check(c["distance"] >= 3 and c["distance"] % 2 == 1, "circuit.distance must be odd and >= 3")
    _check(c["basis"] in ("z", "x"), "circuit.basis must be 'z' or 'x'")
    _check(0 < c["p"] < 0.1, "circuit.p must be in (0, 0.1)")
    for name in ("short_bucket", "long_bucket"):
        lo_hi = r[name]
        _check(len(lo_hi) == 2 and all(isinstance(v, int) for v in lo_hi) and 1 <= lo_hi[0] <= lo_hi[1],
               f"rounds.{name} must be [min, max] integers with 1 <= min <= max")
    _check(r["short_bucket"][1] < r["long_bucket"][0], "rounds buckets must not overlap")
    _check(0 <= r["long_fraction"] <= 1, "rounds.long_fraction must be in [0, 1]")
    _check(isinstance(r["test_max"], int) and r["test_max"] >= r["long_bucket"][1],
           "rounds.test_max must be an integer >= the last round count of long_bucket")
    for name in ("train_shots", "eval_mwpm_errors", "eval_min_shots", "eval_max_shots", "pilot_shots"):
        _check(isinstance(d[name], int) and d[name] > 0, f"dataset.{name} must be a positive integer")
    _check(d["eval_min_shots"] <= d["eval_max_shots"], "dataset.eval_min_shots must be <= eval_max_shots")
    _check(0 < d["train_empty_keep_fraction"] <= 1, "dataset.train_empty_keep_fraction must be in (0, 1]")
    _check(k["features"] == [f for f in TOKEN_FEATURES if f in k["features"]],
           f"tokens.features must be a subset of {TOKEN_FEATURES}, in that order")
    _check(len(k["sincos_periods"]) > 0 and all(p > 0 for p in k["sincos_periods"]),
           "tokens.sincos_periods must be a non-empty list of positive numbers")

    e = t["encoder"]
    _check(e["d_model"] % e["nhead"] == 0, "training.encoder.d_model must be divisible by nhead")
    _check(t["pooling"] in ("mean", "cls"), "training.pooling must be 'mean' or 'cls'")
    _check(isinstance(t["mix_rounds"], bool), "training.mix_rounds must be True or False")
    _check(t["lr_scheduler"] in ("linear", "cosine"), "training.lr_scheduler must be 'linear' or 'cosine'")
    _check(0 <= t["warmup_ratio"] < 1, "training.warmup_ratio must be in [0, 1)")
    _check(t["lr"] > 0 and t["grad_clip"] > 0, "training.lr and grad_clip must be > 0")
    for name in ("num_epochs", "batch_size", "max_tokens_per_batch", "eval_every", "eval_check_shots"):
        _check(isinstance(t[name], int) and t[name] >= 1, f"training.{name} must be an integer >= 1")
    rounds = train_rounds(cfg)
    _check(all(v in rounds for v in t["eval_rounds"] + t["curriculum"]),
           f"training.eval_rounds and curriculum must be round counts in {rounds[0]}..{rounds[-1]}")
    _check(cfg["evaluation"]["target_x_mwpm"] > 0, "evaluation.target_x_mwpm must be > 0")
