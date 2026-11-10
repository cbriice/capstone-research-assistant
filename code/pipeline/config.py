"""Load and validate config.json. Every pipeline script reads its settings through load_config()."""

import json
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "config.json"
DATA_DIR = PIPELINE_DIR / "data"

# Every key the pipeline accepts. A key in config.json that isn't listed here is an error, so typos fail loudly.
SCHEMA: dict[str, Any] = {
    "seed": int,
    "circuit": {"distance": int, "basis": str, "p": float},
    "rounds": {"short_bucket": list, "long_bucket": list, "long_fraction": float},
    "dataset": {"train_val_shots": int, "val_fraction": float, "test_shots": int, "train_empty_keep_fraction": float},
}

def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)

    # exact key sets: typos and missing keys fail at load time
    assert set(cfg) == set(SCHEMA), f"top-level keys must be {sorted(SCHEMA)}, got {sorted(cfg)}"
    for block, keys in SCHEMA.items():
        if isinstance(keys, dict):
            assert set(cfg[block]) == set(keys), f"{block}: keys must be {sorted(keys)}, got {sorted(cfg[block])}"

    c, r, d = cfg["circuit"], cfg["rounds"], cfg["dataset"]
    # validate config params
    assert c["distance"] >= 3 and c["distance"] % 2 == 1, "circuit.distance must be odd and >= 3"
    assert 0 < c["p"] < 0.1, "circuit.p must be in (0, 0.1)"
    for name in ("short_bucket", "long_bucket"):
        lo_hi = r[name]
        assert len(lo_hi) == 2 and all(isinstance(v, int) for v in lo_hi) and 1 <= lo_hi[0] <= lo_hi[1], f"rounds.{name} must be [min, max] integers with 1 <= min <= max"
    assert r["short_bucket"][1] < r["long_bucket"][0], "rounds buckets must not overlap"
    assert 0 <= r["long_fraction"] <= 1, "rounds.long_fraction must be in [0, 1]"
    assert 0 < d["val_fraction"] < 1, "dataset.val_fraction must be in (0, 1)"
    assert d["train_val_shots"] > 0 and d["test_shots"] > 0, "shot counts must be positive"
    assert 0 < d["train_empty_keep_fraction"] <= 1, "dataset.train_empty_keep_fraction must be in (0, 1]"

    return cfg
