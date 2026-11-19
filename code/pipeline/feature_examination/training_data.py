"""Batches for train.py: a generate_data.py run folder -> token batches on DEVICE, epoch by epoch.

Each split is loaded once as uint8 syndromes (one byte per detector) plus, per round count, the token columns that are
the same in every shot (the features in config tokens.features). Both are moved to DEVICE once. A batch's tokens
[shots, detectors, 1 + columns] are built on DEVICE from those two with one torch.cat, so the float tokens are never
all in memory at once. A batch holds one round count, so all its sequences have the same length: no padding or mask.

Dry run (no model):  python code/pipeline/training_data.py
"""
import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from config import DATA_DIR, load_config
from stim_preprocessing import feature_columns, load_syndromes, read_manifest, static_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_run(run: Optional[str], data_dir: Path = DATA_DIR) -> Path:                            #run = a folder name in data/, None = the newest
    runs = sorted(p for p in Path(data_dir).glob("*") if (p / "manifest.json").exists())        #only finished runs have a manifest
    if run is not None:
        runs = [p for p in runs if p.name == run]
    if not runs:
        raise FileNotFoundError(f"no generate_data.py run {run or ''} in {data_dir}")
    return runs[-1]                                                                             #names are dates, so the last one is the newest


def split_rounds(run_dir: Path, split: str) -> list[int]:
    return sorted(int(r) for r in read_manifest(str(run_dir))["splits"][split])                 #round counts the split has, from the manifest


def load_split(run_dir: Path, split: str, tokens_cfg: dict, device: str = DEVICE) -> dict[int, dict]:
    basis = read_manifest(str(run_dir))["config"]["circuit"]["basis"]
    part = {}
    for r in split_rounds(run_dir, split):
        dets, labels, circuit = load_syndromes(run_dir, split, r)
        static = static_features(circuit, basis, tokens_cfg["features"], tokens_cfg["sincos_periods"])
        part[r] = {"dets": torch.from_numpy(dets).to(device),                                   #uint8 0/1 [shots, detectors]
                   "labels": torch.from_numpy(labels).float().to(device),                       #float [shots]: BCEWithLogitsLoss wants floats
                   "static": torch.from_numpy(static).to(device)}                               #float [detectors, columns]: same every shot
    return part


def token_names(tokens_cfg: dict) -> tuple[str, ...]:
    return ("event", *feature_columns(tokens_cfg["features"], tokens_cfg["sincos_periods"]))    #e.g. event, x, y, t, is_z, is_final


def prepare_data(cfg: dict, data_dir: Path = DATA_DIR, device: str = DEVICE) -> dict:
    run_dir = find_run(cfg["training"]["run"], data_dir)
    k = cfg["tokens"]
    return {"run_dir": run_dir, "features": token_names(k),
            "train": load_split(run_dir, "train", k, device),
            "val": load_split(run_dir, "val", k, device)}                                       #test is loaded once, at the end (evaluate.py)


def first_shots(part: dict[int, dict], rounds: list[int], n: int) -> dict[int, dict]:
    return {r: {"dets": part[r]["dets"][:n], "labels": part[r]["labels"][:n],                   #the quick val check during training:
                "static": part[r]["static"]} for r in rounds}                                   #the first n shots of a few round counts


def make_batch(part: dict, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dets = part["dets"][idx].float().unsqueeze(-1)                                              #[B, detectors, 1]: the event column
    static = part["static"].expand(len(idx), -1, -1)                                            #[B, detectors, columns]: a view, no copy until the cat
    return torch.cat([dets, static], dim=-1), part["labels"][idx]                               #tokens [B, detectors, 1 + columns], labels [B]


def shots_per_batch(num_detectors: int, max_shots: int, max_tokens: int) -> int:
    return max(1, min(max_shots, max_tokens // num_detectors))                                  #long r get fewer shots, so tokens per batch stay capped


def longest_rounds(t: dict, epoch: int) -> Optional[int]:
    curriculum = t["curriculum"]                                                                #e.g. [5, 10, 20]: epoch index 0 -> r <= 5
    return curriculum[epoch] if epoch < len(curriculum) else None                               #None = every r


def plan_epoch(part: dict[int, dict], cfg: dict, epoch: int, train: bool) -> list[tuple[int, np.ndarray]]:
    t = cfg["training"]
    rng = np.random.default_rng([t["seed"], epoch])                                             #same seed and epoch -> same batches
    limit = longest_rounds(t, epoch) if train else None                                         #longest r this epoch (None = all)
    max_tokens = t["max_tokens_per_batch"] * (1 if train else 4)                                #no backward pass in eval: 4x fits
    max_shots = t["batch_size"] * (1 if train else 16)
    batches = []
    for r, p in part.items():
        if limit is not None and r > limit:
            continue
        n = len(p["labels"])
        order = rng.permutation(n) if train else np.arange(n)                                   #train: new shot order every epoch
        size = shots_per_batch(p["dets"].shape[1], max_shots, max_tokens)
        batches += [(r, order[i:i + size]) for i in range(0, n, size)]
    if train and t["mix_rounds"]:                                                               #mixed: batches of every r shuffled together
        batches = [batches[i] for i in rng.permutation(len(batches))]
    elif train:                                                                                 #unmixed: one r at a time
        rounds = rng.permutation(sorted({r for r, _ in batches}))                               #r in a new random order each epoch
        rank = {int(r): i for i, r in enumerate(rounds)}                                        #position of each r in that order
        batches.sort(key=lambda b: rank[b[0]])                                                  #stable sort: each r's shuffled batches stay together
    return batches


def epoch_batches(part: dict[int, dict], cfg: dict, epoch: int, train: bool) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
    for r, idx in plan_epoch(part, cfg, epoch, train):
        x, y = make_batch(part[r], torch.from_numpy(idx).to(part[r]["dets"].device))
        yield x, y, r


def main() -> None:                                                                             #dry run: build every batch of every epoch, check it
    cfg = load_config()
    data = prepare_data(cfg)
    print(f"run {data['run_dir'].name} | device {DEVICE} | token {', '.join(data['features'])}")
    for split in ("train", "val"):
        part = data[split]
        print(f"{split:>5}: {sum(len(p['labels']) for p in part.values()):,} shots, r {min(part)}-{max(part)}, "
              f"{sum(p['dets'].numel() for p in part.values()) / 1e6:.0f} MB of syndromes")
    for epoch in range(cfg["training"]["num_epochs"]):
        plan = plan_epoch(data["train"], cfg, epoch, train=True)
        seen = {r: np.zeros(len(p["labels"]), dtype=int) for r, p in data["train"].items()}
        for r, idx in plan:
            seen[r][idx] += 1
        used = [r for r in seen if seen[r].any()]
        if not all((seen[r] == 1).all() for r in used):
            raise RuntimeError("a shot was used twice or skipped")
        biggest = max(len(idx) * data["train"][r]["dets"].shape[1] for r, idx in plan)
        x, y, r = next(epoch_batches(data["train"], cfg, epoch, train=True))
        print(f"epoch {epoch + 1:2d}: {len(plan):5d} batches | r {min(used)}-{max(used)} | max tokens {biggest:,} | "
              f"first batch x {tuple(x.shape)} y {tuple(y.shape)} r={r}")


if __name__ == "__main__":
    # no options (settings live in config.py); this only makes --help print the text at the top, and any
    # other argument an error instead of being ignored
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
