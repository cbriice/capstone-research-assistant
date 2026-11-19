"""Compare token features, then round order and curriculum, on the same data (requirements section 9).

Run from anywhere:  python code/pipeline/experiments.py [--skip-tests]   (settings: the block below + CONFIG in config.py)
  1. Feature study: one run per combination of FEATURE_GROUPS (16 with the 4 default groups) and seed, all with the
     base training: rounds mixed, curriculum on. The event is always in the token.
  2. Order study: the combination with the lowest mean val LER over the seeds is trained in the 3 other orders:
     unmixed with the curriculum, mixed without it, unmixed without it (mixed with it is reused from step 1).
  3. Every model is tested on the whole test split (r 3-50) next to always-0 and MWPM. Nothing is chosen on test.
  4. show_results.py prints the tables and saves results.csv and the graphs.
tests.py runs first. A finished run is skipped when the script is started again with the same settings, and a run
that was stopped mid-way continues from its last finished epoch, so the script can be stopped and started any time.
Output: code/pipeline/experiments/<data run>/<run name>/ with the training folder and result.json.
"""
import argparse
import copy
import itertools
import json
import time
from pathlib import Path
from typing import Optional

import torch

import evaluate
import show_results
import train
from config import DATA_DIR, EXPERIMENTS_DIR, TOKEN_FEATURES, check_config, load_config
from training_data import find_run, token_names

# ======================================== SETTINGS ===========================================
FEATURE_GROUPS = {"position": ["x", "y"], "time": ["t"], "is_z": ["is_z"], "is_final": ["is_final"]}
# optional groups, each doubles the number of runs: "is_first": ["is_first"], "sincos": ["sincos"]
SEEDS = [0, 1, 2]                                                                               #every setting is trained once per seed
ORDERS = {"mixed+curriculum": (True, True),                                                     #name: (mix_rounds, curriculum on)
          "unmixed+curriculum": (False, True),                                                  #the first one is the base training
          "mixed": (True, False),
          "unmixed": (False, False)}
TRAINING_CHANGES: dict = {}                                                                     #applied to CONFIG["training"] in every run. {} = config.py
# =============================================================================================
BASE_ORDER = next(iter(ORDERS))


def combinations() -> list[tuple[str, list[str]]]:
    combos = []
    for on in itertools.product([True, False], repeat=len(FEATURE_GROUPS)):                     #all on first, event only last
        groups = [g for g, keep in zip(FEATURE_GROUPS, on) if keep]
        features = [f for g in groups for f in FEATURE_GROUPS[g]]
        combos.append(("+".join(groups) or "event_only", features))
    return combos


def run_name(combo: str, order: str, seed: int) -> str:
    return f"{combo}__{order}__seed{seed}"                                                      #e.g. position+time__mixed+curriculum__seed0


def run_config(base: dict, features: list[str], order: str, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["tokens"]["features"] = [f for f in TOKEN_FEATURES if f in features]                    #in token order
    mix, curriculum = ORDERS[order]
    cfg["training"].update(mix_rounds=mix, seed=seed)
    if not curriculum:
        cfg["training"]["curriculum"] = []
    check_config(cfg)
    return cfg


def unfinished(out: Path, cfg: dict) -> Optional[Path]:                                         #a training folder of this run that stopped early
    for folder in sorted(out.glob("*/last.pt"), reverse=True):
        if not (folder.parent / "model.pt").exists():
            if torch.load(folder, map_location="cpu", weights_only=False)["config"] == cfg:
                return folder.parent
    return None


def run(name: str, combo: str, order: str, cfg: dict, run_dir: Path, exp_dir: Path, data_dir: Path) -> dict:
    out = exp_dir / name
    path = out / "result.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["config"] == cfg:                                                             #same settings: reuse it
            print(f"{name}: already done | val LER {100 * result['val_ler']:.3f}% | test LER {100 * result['test_ler']:.3f}%")
            return result
        print(f"{name}: settings changed since it was trained, training again")
    print(f"\n{'=' * 30} {name} {'=' * 30}", flush=True)
    t0 = time.perf_counter()
    resume = unfinished(out, cfg)
    if resume is not None:
        print(f"continuing {resume.name} from its last finished epoch")
    history = train.run_training(cfg, data_dir, out_dir=out, resume=resume)
    folder = Path(history["folder"])
    test, _ = evaluate.test_errors(folder / "model.pt", data_dir)                               #the test split: reported, never used to choose
    base = train.baselines(run_dir, "test")
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    result = {"name": name, "combo": combo, "order": order, "config": cfg, "data_run": run_dir.name,
              "run_folder": folder.name, "features": list(token_names(cfg["tokens"])), "seed": cfg["training"]["seed"],
              "params": sum(v.numel() for v in saved["model"].values()), "minutes": (time.perf_counter() - t0) / 60,
              "val_ler": history["best"]["val_ler"], "best_epoch": history["best"]["epoch"],
              "test_ler": sum(test[r] / base[r]["shots"] for r in test) / len(test),            #mean over r, like val
              "test_errors": {str(r): n for r, n in test.items()},
              "epochs": [{"epoch": e["epoch"], "step": e["step"], "val_ler": e["val_ler"]} for e in history["epochs"]],
              "history": {k: history[k] for k in ("step", "lr", "train_loss")}}
    path.write_text(json.dumps(result, indent=1))
    print(f"{name}: done in {result['minutes']:.1f} min | best epoch {result['best_epoch']} | val LER "
          f"{100 * result['val_ler']:.3f}% | test LER {100 * result['test_ler']:.3f}%", flush=True)
    return result


def best_combination(results: list[dict]) -> str:
    by_combo: dict = {}
    for r in results:
        by_combo.setdefault(r["combo"], []).append(r["val_ler"])                                #one entry per seed
    return min(by_combo, key=lambda c: sum(by_combo[c]) / len(by_combo[c]))                     #lowest mean val LER (test is not used to choose)


def main(cfg: Optional[dict] = None, data_dir: Path = DATA_DIR, exp_root: Path = EXPERIMENTS_DIR) -> Path:
    base = load_config() if cfg is None else copy.deepcopy(cfg)
    base["training"].update(TRAINING_CHANGES)
    run_dir = find_run(base["training"]["run"], data_dir)
    base["training"]["run"] = run_dir.name                                                      #every run uses this data folder, even if a newer one appears
    exp_dir = exp_root / run_dir.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    combos = combinations()
    setup = {"feature_groups": FEATURE_GROUPS, "combinations": [c for c, _ in combos], "seeds": SEEDS,
             "orders": list(ORDERS), "base_order": BASE_ORDER,
             "baselines": {split: {str(r): b for r, b in train.baselines(run_dir, split).items()}
                           for split in ("val", "test")}}                                       #always-0 and MWPM per r, for show_results
    (exp_dir / "setup.json").write_text(json.dumps(setup, indent=1))
    total = (len(combos) + len(ORDERS) - 1) * len(SEEDS)
    print(f"data {run_dir.name} | {total} runs ({len(combos)} feature combinations and {len(ORDERS) - 1} more orders, "
          f"x {len(SEEDS)} seeds) | results in {exp_dir}")

    results, n = [], 0                                                                          #1. every feature combination, base order
    for combo, features in combos:
        for seed in SEEDS:
            n += 1
            print(f"--- run {n} of {total}")
            results.append(run(run_name(combo, BASE_ORDER, seed), combo, BASE_ORDER,
                               run_config(base, features, BASE_ORDER, seed), run_dir, exp_dir, data_dir))

    best = best_combination(results)                                                            #2. the best combination in the other orders
    features = dict(combos)[best]
    print(f"\nbest feature combination on val: {best} ({', '.join(['event', *features])})")
    for order in list(ORDERS)[1:]:
        for seed in SEEDS:
            n += 1
            print(f"--- run {n} of {total}")
            run(run_name(best, order, seed), best, order, run_config(base, features, order, seed), run_dir, exp_dir, data_dir)

    show_results.main(exp_dir)                                                                  #3. tables, results.csv, graphs
    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-tests", action="store_true", help="do not run tests.py first")
    args = parser.parse_args()
    if not args.skip_tests:
        import tests
        tests.run_all()
    main()
