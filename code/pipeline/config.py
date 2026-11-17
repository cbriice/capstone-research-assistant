"""Load and validate config.json. Every pipeline script reads its settings through load_config()."""

import json
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "config.json"
DATA_DIR = PIPELINE_DIR / "data"

# ==================================== CONFIG ===============================================
# some params are related. declare them here so that they can be used in computations to preserve ratios
TRAIN_VAL_SHOTS = 1000000

CONFIG: dict = {
  "seed": 1234,
  "circuit": {
    "distance": 3,
    "basis": "z",
    "p": 0.001
  },
  "rounds": [3, 25],              # [min, max], inclusive; equal shots per r in every split (AQ1 pretraining used 1..25)
  "dataset": {
    "train_val_shots": TRAIN_VAL_SHOTS,
    "val_fraction": 0.1,
    "test_shots": int(0.2*TRAIN_VAL_SHOTS),
    "train_empty_keep_fraction": 0.333
  },
  "training": {
    "encoder": {                   # nn.TransformerEncoderLayer(**encoder); head size is fixed at d_model / nhead
      "d_model": 16,
      "nhead": 1,
      "dim_feedforward": 32,
      "dropout": 0.05,
      "activation": "gelu",        # relu | gelu
      "batch_first": True,         # tokens are (shots, detectors, features). torch should not auto batch our data
    },
    "num_layers": 2,               # nn.TransformerEncoder(layer, num_layers)
    "batch_size": 1000,
    "warmup_ratio": 0.1,
    "lr_scheduler": "linear",
    "num_epochs": 5,              
    "lr": 5e-4,
    "weight_decay": 0.01,           # L2 regularization term for AdamW applied to each theta directly 
  }
}

"""
Parameter Optimization Notes: (in progress) 

* Testing embedding dimensions: d ∈ {8, 16, 32} (keep all other params fixed). 

- if  d = 8 performs almost as well as d =16, d = 8 becomes an FPGA candidate. 

- d= 32 improves accuracy significantly, we must determine whether that improvement is worth the hardware expense. 

Selection rule: Keep the smallest embedding dimension that meet our accuracy criteria


* Testing Feed Forward dimensions: dim_feedforward ∈ {16, 32, 64} (keep all other params fixed).
- The FFN increases computation work linearly with width. If dim_feedforward = 16 performs almost as well as dim_feedforward = 32, we can use the smaller value to save computation.
- Reducing FFN width provides more modest savings than reducing the number of tokens 

Example : assuming d = 16

FFN width            FFN MAC                Encoder MAC
16                    51,200                   473,600
32                    102,400                  524,800
64                    204,800                  627,200

Selection rule: Keep the smallest FFN width that meets our accuracy criteria


*Gelu vs ReLU: Gelu is more expensive than ReLU, but it is more expressive. If Gelu performs significantly better than ReLU, we will use Gelu. Otherwise, we will use ReLU to save computation.




"""





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

def load_config() -> dict[str, Any]:
    cfg = CONFIG
    c, r, d = cfg["circuit"], cfg["rounds"], cfg["dataset"]
    # validate config params
    assert c["distance"] >= 3 and c["distance"] % 2 == 1, "circuit.distance must be odd and >= 3"
    assert 0 < c["p"] < 0.1, "circuit.p must be in (0, 0.1)"
    assert len(r) == 2 and all(isinstance(v, int) for v in r) and 1 <= r[0] <= r[1], "rounds must be [min, max] integers with 1 <= min <= max"
    assert 0 < d["val_fraction"] < 1, "dataset.val_fraction must be in (0, 1)"
    assert d["train_val_shots"] > 0 and d["test_shots"] > 0, "shot counts must be positive"
    assert 0 < d["train_empty_keep_fraction"] <= 1, "dataset.train_empty_keep_fraction must be in (0, 1]"

    t = cfg["training"]
    e = t["encoder"]
    assert e["d_model"] % e["nhead"] == 0, "training.encoder.d_model must be divisible by nhead"

    return cfg
