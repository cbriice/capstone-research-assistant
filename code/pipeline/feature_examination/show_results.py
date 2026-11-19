"""Tables and graphs for the results of experiments.py.

Run from anywhere:  python code/pipeline/show_results.py [data run]     (default: the newest folder in experiments/)
Prints three tables and saves these next to the results:
  results.csv          one row per setting (feature combination and order), seeds pooled
  features.png         test LER of every feature combination, best first, next to always-0 and MWPM
  feature_effects.png  mean test LER of the combinations without and with each feature group
  rounds.png           test LER against r for the best combination in every order, always-0 and MWPM
  order.png            every order: val LER after each epoch, and the training loss
LER = logical error rate = share of shots the decoder gets wrong (accuracy = 1 - LER). Lower is better.
Val and test LER here are means over the round counts (each r weighs the same), as in train.py.
Settings are ranked by val LER: the test split is reported, never used to choose.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

from matplotlib.ticker import MaxNLocator

import plots
from config import EXPERIMENTS_DIR
from plots import BLUE, GRID, INK, INK_2, SERIES, SURFACE, label, log_axis, new_figure, pct


# ------------------------------------------------------------------------------- numbers
def load(exp_dir: Path) -> tuple[dict, list[dict]]:
    setup = json.loads((exp_dir / "setup.json").read_text())
    for split, base in setup["baselines"].items():
        setup["baselines"][split] = {int(r): b for r, b in base.items()}                        #JSON keys are text
    results = []
    for path in sorted(exp_dir.glob("*/result.json")):
        x = json.loads(path.read_text())
        if x["combo"] in setup["combinations"] and x["order"] in setup["orders"] and x["seed"] in setup["seeds"]:
            results.append(x)                                                                   #only runs of the current settings
    return setup, results


def rate(errors: dict[int, float], base: dict[int, dict], rs: list[int]) -> Optional[float]:    #mean over r of the per-r LER
    return sum(errors[r] / base[r]["shots"] for r in rs) / len(rs) if rs else None


def reference(base: dict[int, dict], key: str, rs: Optional[list[int]] = None) -> Optional[float]:
    rs = sorted(base) if rs is None else rs                                                     #key = "always0" or "mwpm"
    return rate({r: base[r][key] for r in rs}, base, rs)


def summarize(runs: list[dict], base: dict[int, dict]) -> dict:                                 #one setting, pooled over its seeds
    cut = runs[0]["config"]["rounds"]["long_bucket"][1]                                         #last trained r
    k = len(runs)
    errors = {r: sum(run["test_errors"][str(r)] for run in runs) / k for r in base}             #mean errors per seed
    var = sum(errors[r] / base[r]["shots"] * (1 - errors[r] / base[r]["shots"]) / base[r]["shots"]
              for r in base) / len(base) ** 2                                                   #sampling error of the test shots
    vals = [run["val_ler"] for run in runs]
    return {"combo": runs[0]["combo"], "order": runs[0]["order"], "features": runs[0]["features"], "seeds": k,
            "params": runs[0]["params"], "minutes": sum(run["minutes"] for run in runs),
            "val": sum(vals) / k, "val_lo": min(vals), "val_hi": max(vals),
            "test": rate(errors, base, sorted(base)), "se": math.sqrt(var),
            "trained": rate(errors, base, [r for r in base if r <= cut]),
            "unseen": rate(errors, base, [r for r in base if r > cut]),
            "per_round": {r: errors[r] / base[r]["shots"] for r in base}, "cut": cut, "runs": runs}


def summaries(results: list[dict], base: dict[int, dict]) -> dict[tuple[str, str], dict]:
    groups: dict = {}
    for run in results:
        groups.setdefault((run["combo"], run["order"]), []).append(run)
    return {key: summarize(runs, base) for key, runs in groups.items()}


def token_name(features: list[str]) -> str:
    return "event, " + ", ".join(features[1:]) if len(features) > 1 else "event only"


def feature_effects(rows: list[dict], groups: list[str]) -> list[dict]:
    by = {v["combo"]: v for v in rows}
    effects = []
    for g in groups:
        pairs = []
        for combo, v in by.items():                                                             #every combination without g ...
            have = [] if combo == "event_only" else combo.split("+")
            if g in have:
                continue
            with_g = "+".join(x for x in groups if x in have or x == g)                         #... next to the same one with g
            if with_g in by:
                pairs.append((v["test"], by[with_g]["test"]))
        if pairs:
            effects.append({"group": g, "pairs": len(pairs), "without": sum(a for a, _ in pairs) / len(pairs),
                            "with": sum(b for _, b in pairs) / len(pairs), "helped": sum(b < a for a, b in pairs)})
    return effects


# ------------------------------------------------------------------------------- tables
def print_feature_table(rows: list[dict], base: dict[int, dict], order: str, data_run: str) -> None:
    cut = rows[0]["cut"]
    trained, unseen = [r for r in base if r <= cut], [r for r in base if r > cut]
    print(f"\nFEATURE STUDY | {order} | data {data_run} | test r {min(base)}-{max(base)} | {rows[0]['seeds']} seed(s) | "
          f"ranked by val LER")
    head = (f"{'rank':>4}  {'token':<34}{'size':>5}{'val LER (seed range)':>28}{'test LER +- SE':>21}"
            f"{f'r<={cut}':>9}{f'r>{cut}':>9}{'xMWPM':>7}")
    print(head + "\n" + "-" * len(head))
    mwpm = reference(base, "mwpm")
    for i, v in enumerate(rows, start=1):
        val = f"{pct(v['val'])} ({pct(v['val_lo'], 2)}-{pct(v['val_hi'], 2)})"
        print(f"{i:>4}  {token_name(v['features']):<34}{len(v['features']):>5}{val:>28}"
              f"{pct(v['test']):>10} +- {pct(v['se']):<7}{pct(v['trained'], 2):>9}{pct(v['unseen'], 2):>9}"
              f"{v['test'] / mwpm:6.1f}x")
    for name, key in (("always-0", "always0"), ("MWPM", "mwpm")):
        print(f"{'':>4}  {name:<34}{'':>5}{'':>28}{pct(reference(base, key)):>10}{'':>11}"
              f"{pct(reference(base, key, trained), 2):>9}{pct(reference(base, key, unseen), 2):>9}")
    print("size = numbers per token. LER = mean over the round counts. +- SE = standard error of the test LER from "
          "the number of test shots;\nxMWPM = test LER / MWPM's test LER. Two settings less than about 2 SE apart are "
          "not clearly different.")


def print_effects(effects: list[dict]) -> None:
    print("\nEFFECT OF EACH FEATURE GROUP | mean test LER of the combinations without it and of the same ones with it")
    head = f"{'feature group':<18}{'without':>10}{'with':>10}{'change':>11}{'helped in':>14}"
    print(head + "\n" + "-" * len(head))
    for e in effects:
        print(f"{e['group']:<18}{pct(e['without'], 2):>10}{pct(e['with'], 2):>10}"
              f"{100 * (e['with'] - e['without']):>+9.2f}pt{e['helped']:>8} of {e['pairs']}")


def print_orders(rows: list[dict], base_order: str) -> None:
    print(f"\nORDER STUDY | best combination on val: {token_name(rows[0]['features'])}")
    if len(rows) == 1:
        print("the other orders are not finished yet")
        return
    cut, ref = rows[0]["cut"], rows[0]
    head = (f"{'order':<22}{'val LER':>10}{'best epoch':>12}{'test LER +- SE':>21}{f'r<={cut}':>9}{f'r>{cut}':>9}"
            f"   test vs {base_order}")
    print(head + "\n" + "-" * len(head))
    for v in rows:
        epochs = "/".join(str(run["best_epoch"]) for run in v["runs"])                          #one per seed
        diff = ""
        if v is not ref:
            gap = (v["test"] - ref["test"]) / math.sqrt(v["se"] ** 2 + ref["se"] ** 2)
            diff = f"{'worse' if gap > 0 else 'better'} by {abs(gap):.1f} SE"
        print(f"{v['order']:<22}{pct(v['val']):>10}{epochs:>12}{pct(v['test']):>10} +- {pct(v['se']):<7}"
              f"{pct(v['trained'], 2):>9}{pct(v['unseen'], 2):>9}   {diff}")
    print("best epoch = the epoch kept (lowest val LER), one per seed. A gap above about 2 SE is unlikely to be noise.")


def write_csv(s: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combination", "order", "features", "seeds", "params", "minutes", "val_ler", "val_ler_min",
                    "val_ler_max", "test_ler", "test_se", "test_ler_trained_r", "test_ler_unseen_r"])
        for v in sorted(s.values(), key=lambda v: v["val"]):
            w.writerow([v["combo"], v["order"], " ".join(v["features"]), v["seeds"], v["params"], f"{v['minutes']:.1f}",
                        f"{v['val']:.6f}", f"{v['val_lo']:.6f}", f"{v['val_hi']:.6f}", f"{v['test']:.6f}",
                        f"{v['se']:.6f}", f"{v['trained']:.6f}", "" if v["unseen"] is None else f"{v['unseen']:.6f}"])


# ------------------------------------------------------------------------------- graphs
def plot_features(rows: list[dict], base: dict[int, dict], path: Path) -> None:
    chosen = rows[0]["combo"]                                                                   #rows come ranked by val
    rows = sorted(rows, key=lambda v: v["test"])
    fig, (ax,) = new_figure(10, 1.9 + 0.42 * len(rows), "Test error of every feature combination",
                            "test LER, mean over the round counts and seeds (log scale, lower is better), "
                            "line = +- 1 standard error")
    y = list(range(len(rows)))[::-1]                                                            #lowest test LER at the top
    ax.errorbar([v["test"] for v in rows], y, xerr=[v["se"] for v in rows], fmt="o", color=BLUE, markersize=8,
                elinewidth=1.5, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for v, yy in zip(rows, y):
        best = v["combo"] == chosen
        label(ax, pct(v["test"], 2) + ("  (chosen on val)" if best else ""), (v["test"] + v["se"], yy), 6, bold=best)
    ax.set_yticks(y, [token_name(v["features"]) for v in rows], color=INK, fontsize=9)
    ax.grid(False, axis="y")
    refs = [reference(base, "always0"), reference(base, "mwpm")]
    lo = min([v["test"] - v["se"] for v in rows] + refs)
    hi = max([v["test"] + v["se"] for v in rows] + refs)
    log_axis(ax, "x", max(lo, 1e-5) / 1.5, hi * 2.5)
    plots.reference_line(ax, refs[0], "always-0", True, "--")
    plots.reference_line(ax, refs[1], "MWPM", True, ":")
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xlabel("test LER", color=INK_2)
    fig.savefig(path, dpi=150)
    plots.plt.close(fig)


def plot_effects(effects: list[dict], path: Path) -> None:
    fig, (ax,) = new_figure(9, 2.2 + 0.6 * len(effects), "What each feature group adds",
                            "mean test LER of the combinations without the group and of the same combinations "
                            "with it (log scale)")
    y = list(range(len(effects)))[::-1]
    for e, yy in zip(effects, y):
        ax.plot([e["without"], e["with"]], [yy, yy], color=GRID, linewidth=3, zorder=1, solid_capstyle="round")
        low, high = sorted((e["without"], e["with"]))
        label(ax, pct(low, 2), (low, yy), -9, ha="right")                                       #lower value on the left of its dot,
        label(ax, pct(high, 2), (high, yy), 9)                                                  #higher on the right: never overlap
    for key, color in (("without", INK_2), ("with", BLUE)):
        ax.scatter([e[key] for e in effects], y, s=80, color=color, zorder=3, label=key, edgecolor=SURFACE,
                   linewidth=1.5)
    ax.set_yticks(y, [f"{e['group']}\nhelped in {e['helped']} of {e['pairs']}" for e in effects], color=INK, fontsize=9)
    values = [v for e in effects for v in (e["without"], e["with"])]
    log_axis(ax, "x", min(values) / 1.8, max(values) * 1.8)
    ax.set_ylim(-0.6, len(effects) - 0.4)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlabel("test LER", color=INK_2)
    fig.savefig(path, dpi=150)
    plots.plt.close(fig)


def plot_rounds(rows: list[dict], base: dict[int, dict], path: Path) -> None:
    lines = [{"name": "always-0", "values": {r: b["always0"] / b["shots"] for r, b in base.items()},
              "color": INK_2, "style": "--"},
             {"name": "MWPM", "values": {r: b["mwpm"] / b["shots"] for r, b in base.items()}, "color": INK,
              "style": ":"}]
    for v, color in zip(rows, SERIES):
        lines.append({"name": v["order"], "values": v["per_round"], "color": color, "style": "-"})
    plots.plot_by_rounds(lines, rows[0]["cut"], path, "Test error against the number of rounds",
                         f"best combination ({token_name(rows[0]['features'])}) in every order, mean over seeds, "
                         "log scale", "test LER")


def plot_orders(rows: list[dict], val_base: dict[int, dict], path: Path) -> None:
    fig, (ax1, ax2) = new_figure(12, 5.2, "Round order and curriculum",
                                 f"best combination ({token_name(rows[0]['features'])}), first seed", panels=2)
    errors, losses = [], []
    for v, color in zip(rows, SERIES):
        run = v["runs"][0]
        y = [e["val_ler"] for e in run["epochs"]]
        errors += y
        ax1.plot([e["epoch"] for e in run["epochs"]], y, color=color, linewidth=2, marker="o", markersize=6,
                 markeredgecolor=SURFACE, label=v["order"])
        h = run["history"]
        losses += h["train_loss"]
        ax2.plot(h["step"], h["train_loss"], color=color, linewidth=1.3, label=v["order"])
    for key, name, style in (("always0", "always-0", "--"), ("mwpm", "MWPM", ":")):
        value = reference(val_base, key)                                                        #on the same val shots
        errors.append(value)
        plots.reference_line(ax1, value, name, False, style)
    log_axis(ax1, "y", min(errors) / 1.5, max(errors) * 1.5)
    ax1.set_title("val LER after each epoch (every val shot, mean over r)", loc="left", fontsize=10, color=INK)
    ax1.set_xlabel("epoch", color=INK_2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    if losses:
        log_axis(ax2, "y", min(losses) / 1.3, max(losses) * 1.3, percent=False)
    ax2.set_title("training loss (mean between quick checks)", loc="left", fontsize=10, color=INK)
    ax2.set_xlabel("step", color=INK_2)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(path, dpi=150)
    plots.plt.close(fig)


# ------------------------------------------------------------------------------- main
def newest(exp_root: Path = EXPERIMENTS_DIR) -> Path:
    folders = sorted(p for p in exp_root.glob("*") if (p / "setup.json").exists())
    if not folders:
        raise SystemExit(f"No experiment results in {exp_root}.\nshow_results.py only summarizes experiments.py "
                         "(run that first). A single train.py run is reported by evaluate.py.")
    return folders[-1]


def main(exp_dir: Optional[Path] = None) -> None:
    exp_dir = newest() if exp_dir is None else Path(exp_dir)
    setup, results = load(exp_dir)
    if not results:
        raise SystemExit(f"no finished runs in {exp_dir} yet")
    base, base_order = setup["baselines"]["test"], setup["base_order"]
    s = summaries(results, base)
    rows = sorted((v for v in s.values() if v["order"] == base_order), key=lambda v: v["val"])  #ranked by val
    print_feature_table(rows, base, base_order, exp_dir.name)
    effects = feature_effects(rows, list(setup["feature_groups"]))
    if effects:
        print_effects(effects)
    best = rows[0]["combo"]
    orders = sorted((v for v in s.values() if v["combo"] == best), key=lambda v: setup["orders"].index(v["order"]))
    print_orders(orders, base_order)
    write_csv(s, exp_dir / "results.csv")
    saved = ["results.csv", "features.png", "rounds.png"]
    plot_features(rows, base, exp_dir / "features.png")
    if effects:
        plot_effects(effects, exp_dir / "feature_effects.png")
        saved.append("feature_effects.png")
    plot_rounds(orders, base, exp_dir / "rounds.png")
    if len(orders) > 1:
        plot_orders(orders, setup["baselines"]["val"], exp_dir / "order.png")
        saved.append("order.png")
    print(f"\nsaved {', '.join(saved)} in {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", help="a folder name in experiments/ (default: the newest)")
    args = parser.parse_args()
    main(None if args.run is None else EXPERIMENTS_DIR / args.run)
