"""Generate the train / val / test detection-event datasets.

Run from anywhere:  python code/pipeline/generate_data.py      (settings: CONFIG in config.py)
Output: a new run folder code/pipeline/data/<YYYY-MM-DD_HHMMSS>/ with
  circuits/rNNN.stim and rNNN.dem     the circuit and detector error model of every round count
  train/ val/ test/ rNNN.npz          dets [shots, detectors] and obs [shots, 1], uint8 0/1
  manifest.json                       the config, package versions and, per split and r: shots, label rate,
                                      empty fraction, and the always-0 and MWPM errors on exactly those shots
Splits:
  train  r 3-30: train_shots in total, 80% spread evenly over r 3-15 and 20% over r 16-30; 2/3 of empty shots dropped
  val    r 3-30, test r 3-50: per round count, enough shots for at least about 100 MWPM errors. The size comes from a
         pilot sample (its own seed) decoded with PyMatching: shots = eval_mwpm_errors (150) / pilot MWPM error rate,
         kept between eval_min_shots and eval_max_shots. The manifest has the MWPM errors each r really got.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pymatching
import stim

from config import DATA_DIR, load_config, test_rounds, train_rounds

SPLITS = ("train", "val", "test")
SPLIT_IDS = {"train": 0, "val": 1, "test": 2, "pilot": 3}   # part of each (split, rounds) seed; never reorder
MAX_CHUNK_BYTES = 1 << 28                       # max detector bytes per sampler call (1 byte per bit), bounds peak memory
PACKAGES = ("stim", "pymatching", "numpy", "torch", "matplotlib")


def make_circuit(distance: int, rounds: int, p: float, basis: str) -> stim.Circuit:
    # Rotated surface-code memory experiment with uniform circuit-level depolarizing noise: all four of
    # Stim's generator noise channels at strength p (the noise model of research_log settled conclusion #4).
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=distance, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def split_evenly(total: int, rounds: list[int]) -> dict[int, int]:
    # Integer shot counts per round count that sum exactly to `total`.
    q, rem = divmod(total, len(rounds))
    return {r: q + (i < rem) for i, r in enumerate(rounds)}


def train_plan(cfg: dict[str, Any]) -> dict[int, int]:
    # Training shots per round count: (1 - long_fraction) of them over the short bucket, the rest over the long one.
    rc = cfg["rounds"]
    short = list(range(rc["short_bucket"][0], rc["short_bucket"][1] + 1))
    long = list(range(rc["long_bucket"][0], rc["long_bucket"][1] + 1))
    n_long = round(cfg["dataset"]["train_shots"] * rc["long_fraction"])
    return split_evenly(cfg["dataset"]["train_shots"] - n_long, short) | split_evenly(n_long, long)


def sample_round(dem: stim.DetectorErrorModel, n_target: int, empty_keep: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    # Sample until n_target shots are kept. Empty shots (no detector fired) are kept with probability
    # empty_keep; the decision depends on the syndrome only, so P(label | syndrome) is unchanged.
    # Returns dets [n, D] and obs [n, 1] as uint8 0/1 (one byte per bit), and the empty fraction before dropping.
    sampler = dem.compile_sampler(seed=int(rng.integers(2**32)))
    max_chunk = MAX_CHUNK_BYTES // dem.num_detectors
    dets_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    kept = sampled = empty_sampled = 0
    while kept < n_target:
        need = n_target - kept
        # With a drop, the kept count per batch is random: oversample 2x (floor 4096) and trim at the end.
        n = min(max_chunk, need if empty_keep == 1 else max(2 * need, 4096))
        dets, obs, _ = sampler.sample(n)                         # bool [n, D], [n, 1]
        empty = ~dets.any(axis=1)                                # elementwise NOT
        keep = ~empty | (rng.random(n) < empty_keep)             # all True when empty_keep == 1
        dets_parts.append(dets[keep])
        obs_parts.append(obs[keep])
        kept += int(keep.sum())
        sampled += n
        empty_sampled += int(empty.sum())
    dets = np.concatenate(dets_parts)[:n_target].view(np.uint8)  # same bytes, 0/1 values
    obs = np.concatenate(obs_parts)[:n_target].view(np.uint8)
    return dets, obs, empty_sampled / sampled


def mwpm_errors(matcher: pymatching.Matching, dets: np.ndarray, obs: np.ndarray) -> int:
    # Shots where PyMatching's prediction of the observable is wrong.
    return int((matcher.decode_batch(dets)[:, 0] != obs[:, 0]).sum())


def eval_size(cfg: dict[str, Any], dem: stim.DetectorErrorModel, matcher: pymatching.Matching,
              r: int) -> tuple[int, float]:
    # Val / test shots for one round count: decode a pilot sample, then size for about eval_mwpm_errors MWPM errors.
    d = cfg["dataset"]
    rng = np.random.default_rng([cfg["seed"], SPLIT_IDS["pilot"], r])   # its own stream: the pilot is not reused
    dets, obs, _ = sample_round(dem, d["pilot_shots"], 1.0, rng)
    rate = mwpm_errors(matcher, dets, obs) / d["pilot_shots"]
    n = math.ceil(d["eval_mwpm_errors"] / max(rate, 1 / d["pilot_shots"]))   # 0 pilot errors: treat as 1
    return min(max(n, d["eval_min_shots"]), d["eval_max_shots"]), rate


def package_versions() -> dict[str, Optional[str]]:
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def main(cfg: Optional[dict] = None, data_dir: Path = DATA_DIR) -> Path:
    # Settings come from config.py unless a cfg is passed (tests.py passes a tiny one). Returns the new run folder.
    cfg = load_config() if cfg is None else cfg

    run_dir = data_dir / datetime.now().strftime("%Y-%m-%d_%H%M%S")   # one folder per run, local time
    for sub in ("circuits", *SPLITS):
        (run_dir / sub).mkdir(parents=True)                       # fails if the folder already exists

    c = cfg["circuit"]
    plans: dict[str, dict[int, int]] = {"train": train_plan(cfg), "val": {}, "test": {}}
    manifest: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": package_versions(),
        "config": cfg,
        "format": "dets [n, num_detectors] and obs [n, 1], uint8 0/1, not bit-packed",
        "splits": {split: {} for split in SPLITS},
        "pilot_mwpm_rate": {},
    }

    t0 = time.perf_counter()
    for r in test_rounds(cfg):                                    # 3-50: every r any split needs
        circuit = make_circuit(c["distance"], r, c["p"], c["basis"])
        dem = circuit.detector_error_model(decompose_errors=True)   # the same DEM PyMatching is built from
        circuit.to_file(run_dir / "circuits" / f"r{r:03d}.stim")
        dem.to_file(run_dir / "circuits" / f"r{r:03d}.dem")
        matcher = pymatching.Matching.from_detector_error_model(dem)

        n_eval, rate = eval_size(cfg, dem, matcher, r)
        manifest["pilot_mwpm_rate"][str(r)] = rate
        plans["test"][r] = n_eval
        if r in train_rounds(cfg):
            plans["val"][r] = n_eval

        for split in SPLITS:
            n = plans[split].get(r, 0)
            if n == 0:
                continue
            empty_keep = cfg["dataset"]["train_empty_keep_fraction"] if split == "train" else 1.0
            rng = np.random.default_rng([cfg["seed"], SPLIT_IDS[split], r])   # one stream per (split, r)
            dets, obs, empty_before = sample_round(dem, n, empty_keep, rng)
            np.savez_compressed(run_dir / split / f"r{r:03d}.npz", dets=dets, obs=obs,
                                rounds=np.int64(r), num_detectors=np.int64(circuit.num_detectors))
            info = {
                "shots": n,
                "num_detectors": circuit.num_detectors,
                "label_rate": float(obs[:, 0].mean()),
                "empty_fraction": float((~dets.any(axis=1)).mean()),
                "empty_fraction_before_drop": empty_before,
                "always0_errors": int(obs[:, 0].sum()),          # always-0 is wrong exactly when the observable flipped
            }
            if split != "train":                                  # the baselines every val / test number is shown next to
                info["mwpm_errors"] = mwpm_errors(matcher, dets, obs)
            manifest["splits"][split][str(r)] = info
        print(f"r={r:3d} | " + " | ".join(f"{s} {plans[s].get(r, 0):>7,}" for s in SPLITS)
              + f" | pilot MWPM {100 * rate:5.2f}% | {time.perf_counter() - t0:6.1f} s", flush=True)

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    for split in SPLITS:
        stats = manifest["splits"][split]
        n_total = sum(s["shots"] for s in stats.values())
        empty = sum(s["shots"] * s["empty_fraction"] for s in stats.values()) / n_total
        label = sum(s["shots"] * s["label_rate"] for s in stats.values()) / n_total
        mwpm = min((s["mwpm_errors"] for s in stats.values()), default=None) if split != "train" else None
        print(f"{split:>5}: {n_total:>10,} shots, r {min(map(int, stats))}-{max(map(int, stats))} | "
              f"empty {100 * empty:5.1f}% | label = 1 {100 * label:5.2f}%"
              + ("" if mwpm is None else f" | fewest MWPM errors in one r: {mwpm}"))
    print(f"done in {time.perf_counter() - t0:.0f} s -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    # no options (settings live in config.py); this only makes --help print the text at the top, and any
    # other argument an error instead of being ignored
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
