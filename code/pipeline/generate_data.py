"""Generate the train / val / test detection-event datasets (v1.0). Decisions and their reasons: TRACKER.md.

Run from anywhere:  python code/pipeline/generate_data.py [--config path/to/config.json]
Output: a new run folder code/pipeline/data/<YYYY-MM-DD_HHMMSS>/ (see TRACKER.md "Data layout").
"""

import argparse
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import stim

from config import DATA_DIR, DEFAULT_CONFIG_PATH, load_config

SPLITS = ("train", "val", "test")
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}   # part of each (split, rounds) seed; never reorder
MAX_CHUNK = 1 << 20                             # max shots per sampler call, bounds peak memory


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


def shots_per_round(cfg: dict[str, Any], split: str) -> dict[int, int]:
    # Stratified allocation: a fixed number of shots per round count (no random draw of r).
    rc, dc = cfg["rounds"], cfg["dataset"]
    short = list(range(rc["short_bucket"][0], rc["short_bucket"][1] + 1))
    long = list(range(rc["long_bucket"][0], rc["long_bucket"][1] + 1))
    if split == "test":                                    # even over every r, for per-r profiling
        return split_evenly(dc["test_shots"], short + long)
    n_val = round(dc["train_val_shots"] * dc["val_fraction"])
    total = n_val if split == "val" else dc["train_val_shots"] - n_val
    n_long = round(total * rc["long_fraction"])            # train and val share the short/long mix
    return split_evenly(total - n_long, short) | split_evenly(n_long, long)


def sample_round(dem: stim.DetectorErrorModel, n_target: int, empty_keep: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    # Sample until n_target shots are kept. Empty shots (no detector fired) are kept with probability
    # empty_keep; the decision depends on the syndrome only, so P(label | syndrome) is unchanged.
    # Returns bit-packed dets [n, ceil(D/8)], bit-packed obs [n, 1], and the empty fraction before dropping.
    sampler = dem.compile_sampler(seed=int(rng.integers(2**32)))
    dets_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    kept = sampled = empty_sampled = 0
    while kept < n_target:
        n = min(MAX_CHUNK, max(2 * (n_target - kept), 4096))
        dets, obs, _ = sampler.sample(n, bit_packed=True)
        empty = ~dets.any(axis=1)
        keep = ~empty | (rng.random(n) < empty_keep) if empty_keep < 1 else np.ones(n, dtype=np.bool_)
        dets_parts.append(dets[keep])
        obs_parts.append(obs[keep])
        kept += int(keep.sum())
        sampled += n
        empty_sampled += int(empty.sum())
    return (np.concatenate(dets_parts)[:n_target], np.concatenate(obs_parts)[:n_target],
            empty_sampled / sampled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    cfg = load_config(parser.parse_args().config)

    run_dir = DATA_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")   # one folder per run, local time
    for sub in ("circuits", *SPLITS):
        (run_dir / sub).mkdir(parents=True)                       # fails if the folder already exists

    c = cfg["circuit"]
    plans = {split: shots_per_round(cfg, split) for split in SPLITS}
    all_rounds = sorted({r for plan in plans.values() for r, n in plan.items() if n > 0})
    manifest: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {pkg: version(pkg) for pkg in ("stim", "numpy")},
        "config": cfg,
        "format": "dets/obs bit-packed uint8, bitorder little; unpack with "
                  "np.unpackbits(a, axis=1, bitorder='little', count=num_detectors)",
        "splits": {split: {} for split in SPLITS},
    }

    t0 = time.perf_counter()
    for r in all_rounds:
        circuit = make_circuit(c["distance"], r, c["p"], c["basis"])
        dem = circuit.detector_error_model(decompose_errors=True)   # the same DEM PyMatching is built from
        circuit.to_file(run_dir / "circuits" / f"r{r:03d}.stim")
        dem.to_file(run_dir / "circuits" / f"r{r:03d}.dem")

        for split in SPLITS:
            n = plans[split].get(r, 0)
            if n == 0:
                continue
            empty_keep = cfg["dataset"]["train_empty_keep_fraction"] if split == "train" else 1.0
            rng = np.random.default_rng([cfg["seed"], SPLIT_IDS[split], r])   # one stream per (split, r)
            dets, obs, empty_before = sample_round(dem, n, empty_keep, rng)
            np.savez_compressed(run_dir / split / f"r{r:03d}.npz", dets=dets, obs=obs,
                                rounds=np.int64(r), num_detectors=np.int64(circuit.num_detectors))
            manifest["splits"][split][str(r)] = {
                "shots": n,
                "num_detectors": circuit.num_detectors,
                "label_rate": float(obs.any(axis=1).mean()),
                "empty_fraction": float((~dets.any(axis=1)).mean()),
                "empty_fraction_before_drop": empty_before,
            }
        print(f"r={r:3d} | " + " | ".join(f"{s} {plans[s].get(r, 0):>7,}" for s in SPLITS)
              + f" | {time.perf_counter() - t0:6.1f} s", flush=True)

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    for split in SPLITS:
        stats = manifest["splits"][split].values()
        n_total = sum(s["shots"] for s in stats)
        empty = sum(s["shots"] * s["empty_fraction"] for s in stats) / n_total
        label = sum(s["shots"] * s["label_rate"] for s in stats) / n_total
        print(f"{split:>5}: {n_total:>10,} shots over {len(stats)} round counts | "
              f"empty {100 * empty:5.1f}% | label = 1 {100 * label:5.2f}%")
    print(f"done in {time.perf_counter() - t0:.0f} s -> {run_dir}")


if __name__ == "__main__":
    main()
