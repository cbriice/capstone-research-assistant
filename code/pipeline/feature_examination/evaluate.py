"""Test a trained model on the test split: once, at the end, after every choice was made on val.

Run from anywhere:  python code/pipeline/evaluate.py [run folder]           (default: the newest folder in train_runs/)
                    python code/pipeline/evaluate.py <run folder> --again   test again, although it was tested before
For every test round count (3-50; 31-50 were never trained on) it prints and saves:
  LER    logical error rate = share of shots decoded wrong, with the error count and the standard error sqrt(p(1-p)/n)
  eps    error per round = (1 - (1 - 2 LER)^(1/r)) / 2, so round counts can be compared
  next to always-0 and MWPM on the same shots, and the two targets of the requirements:
  below always-0, and eps at most target_x_mwpm (2) times MWPM's eps.
Saved in the run folder: test_results.json, test_results.csv, test_ler.png, test_per_round.png
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import torch

import plots
from config import DATA_DIR, TRAIN_DIR
from train import Decoder, baselines, evaluate, per_round_error
from training_data import DEVICE, find_run, load_split


def newest_run(train_dir: Path = TRAIN_DIR) -> Path:
    runs = sorted(p for p in Path(train_dir).glob("*") if (p / "model.pt").exists())            #only finished runs have a model.pt
    if not runs:
        raise FileNotFoundError(f"no finished training run in {train_dir}")
    return runs[-1]


def data_folder(saved: dict, data_dir: Path = DATA_DIR) -> Path:
    path = Path(saved["data_run"])                                                              #where the data was when the model trained
    return path if (path / "manifest.json").exists() else find_run(path.name, data_dir)         #else the same run name in data/


def test_errors(model_path: Path, data_dir: Path = DATA_DIR) -> tuple[dict[int, int], Path]:
    saved = torch.load(model_path, map_location=DEVICE, weights_only=False)
    cfg = saved["config"]
    model = Decoder(len(saved["features"]), cfg["training"]).to(DEVICE)
    model.load_state_dict(saved["model"])
    run_dir = data_folder(saved, data_dir)
    part = load_split(run_dir, "test", saved["tokens"])                                         #the same token recipe as in training
    ev = evaluate(model, part, cfg, sorted(part))
    return {r: e["errors"] for r, e in ev.items()}, run_dir                                     #test errors per round count


def se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def result_rows(errors: dict[int, int], base: dict[int, dict], target: float) -> list[dict]:
    rows = []
    for r in sorted(errors):
        n, b = base[r]["shots"], base[r]
        ler, a0, mw = errors[r] / n, b["always0"] / n, b["mwpm"] / n
        eps, eps_mw = per_round_error(ler, r), per_round_error(mw, r)
        rows.append({"r": r, "shots": n, "errors": errors[r], "ler": ler, "se": se(ler, n),
                     "always0": a0, "mwpm": mw, "mwpm_errors": b["mwpm"], "eps": eps, "eps_mwpm": eps_mw,
                     "x_mwpm": eps / eps_mw if eps_mw > 0 else math.inf,
                     "below_always0": errors[r] < b["always0"], "within_target": eps <= target * eps_mw})
    return rows


def summary(rows: list[dict]) -> dict:
    n = sum(x["shots"] for x in rows)
    errors = sum(x["errors"] for x in rows)
    return {"round_counts": len(rows), "shots": n, "errors": errors, "ler": errors / n,
            "always0": sum(x["always0"] * x["shots"] for x in rows) / n,
            "mwpm": sum(x["mwpm"] * x["shots"] for x in rows) / n,
            "mean_x_mwpm": sum(x["x_mwpm"] for x in rows) / len(rows),
            "below_always0": sum(x["below_always0"] for x in rows), "within_target": sum(x["within_target"] for x in rows)}


def print_report(rows: list[dict], cut: int, target: float, name: str) -> dict:
    print(f"\nTEST RESULTS | {name} | r {rows[0]['r']}-{rows[-1]['r']} (trained on r <= {cut})")
    head = (f"{'r':>3}{'shots':>9}{'errors':>8}{'LER +- SE':>19}{'always-0':>10}{'MWPM':>9}{'eps':>10}"
            f"{'eps MWPM':>10}{'xMWPM':>7}  targets")
    print(head + "\n" + "-" * len(head))
    for x in rows:
        if x["r"] == cut + 1:
            print(f"{'':>3}  ---- round counts below were never trained on ----")
        marks = ("ok " if x["below_always0"] else "NO ") + ("ok" if x["within_target"] else "NO")
        print(f"{x['r']:>3}{x['shots']:>9,}{x['errors']:>8}{plots.pct(x['ler'], 3):>10} +- {plots.pct(x['se'], 3):<6}"
              f"{plots.pct(x['always0'], 2):>10}{plots.pct(x['mwpm'], 2):>9}{plots.pct(x['eps'], 3):>10}"
              f"{plots.pct(x['eps_mwpm'], 3):>10}{x['x_mwpm']:>6.1f}x  {marks}")
    print(f"targets: below always-0 | eps <= {target:g} x MWPM's eps")
    out = {}
    for label, part in ((f"r <= {cut} (trained)", [x for x in rows if x["r"] <= cut]),
                        (f"r > {cut} (never trained)", [x for x in rows if x["r"] > cut])):
        if not part:
            continue
        s = summary(part)
        out[label] = s
        print(f"{label:<24} LER {plots.pct(s['ler'])} | always-0 {plots.pct(s['always0'])} | MWPM {plots.pct(s['mwpm'])} | "
              f"mean eps {s['mean_x_mwpm']:.1f}x MWPM | below always-0 in {s['below_always0']} of {s['round_counts']} r | "
              f"within {target:g}x MWPM in {s['within_target']} of {s['round_counts']} r")
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(rows[0])
        w.writerow(keys)
        for x in rows:
            w.writerow([f"{x[k]:.6g}" if isinstance(x[k], float) else x[k] for k in keys])


def plot_results(rows: list[dict], cut: int, target: float, folder: Path, name: str) -> None:
    by = lambda k: {x["r"]: x[k] for x in rows}
    plots.plot_by_rounds(
        [{"name": "always-0", "values": by("always0"), "color": plots.INK_2, "style": "--"},
         {"name": "MWPM", "values": by("mwpm"), "color": plots.INK, "style": ":"},
         {"name": "model", "values": by("ler"), "color": plots.BLUE, "style": "-", "se": by("se")}],
        cut, folder / "test_ler.png", "Test logical error rate against the number of rounds",
        f"{name} | every test shot, band = +- 1 standard error, log scale", "test LER")
    plots.plot_by_rounds(
        [{"name": f"{target:g} x MWPM (target)", "values": {r: target * v for r, v in by("eps_mwpm").items()},
          "color": plots.GREEN, "style": "--"},
         {"name": "MWPM", "values": by("eps_mwpm"), "color": plots.INK, "style": ":"},
         {"name": "model", "values": by("eps"), "color": plots.BLUE, "style": "-"}],
        cut, folder / "test_per_round.png", "Error per round against the number of rounds",
        f"{name} | eps = (1 - (1 - 2 LER)^(1/r)) / 2: the model meets the target where it is under the green line",
        "error per round")


def main(run_folder: Optional[Path] = None, again: bool = False, data_dir: Path = DATA_DIR) -> dict:
    folder = newest_run() if run_folder is None else Path(run_folder)
    if not (folder / "model.pt").exists() and (TRAIN_DIR / folder.name / "model.pt").exists():
        folder = TRAIN_DIR / folder.name                                                        #a bare run name: look in train_runs/
    out_path = folder / "test_results.json"
    if out_path.exists() and not again:                                                         #the test split is for the end, not for choosing
        raise SystemExit(f"{folder.name} was already tested ({out_path}). Use --again to test it again.")
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    cfg = saved["config"]
    cut, target = cfg["rounds"]["long_bucket"][1], cfg["evaluation"]["target_x_mwpm"]
    errors, run_dir = test_errors(folder / "model.pt", data_dir)
    rows = result_rows(errors, baselines(run_dir, "test"), target)
    name = f"run {folder.name}, epoch {saved['epoch']}"
    totals = print_report(rows, cut, target, name)
    result = {"run": folder.name, "data_run": run_dir.name, "epoch": saved["epoch"], "val_ler": saved["val_ler"],
              "features": list(saved["features"]), "target_x_mwpm": target, "summary": totals, "per_round": rows}
    out_path.write_text(json.dumps(result, indent=1))
    write_csv(rows, folder / "test_results.csv")
    plot_results(rows, cut, target, folder, name)
    print(f"saved test_results.json, test_results.csv, test_ler.png and test_per_round.png in {folder}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="a train_runs/<run> folder (default: the newest)")
    parser.add_argument("--again", action="store_true", help="test again even if test_results.json exists")
    args = parser.parse_args()
    main(args.run, args.again)
