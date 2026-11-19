"""More graphs from the results of experiments.py: training curves, every seed, every round count, two runs compared.

Run from anywhere:  python code/pipeline/explore_results.py [data run] [--compare <other run>]
  data run    a folder name in experiments/ (default: the newest, as in show_results.py)
  --compare   a second folder in experiments/ to draw next to the first, e.g. your renamed d_model_16
Reads setup.json, every <run>/result.json and the history.json that train.py writes in every training folder
(the training loss and the quick checks every 100 steps, and the val errors at every r after each epoch).
Prints a table of every setting and saves in experiments/<data run>/explore/:
  train_loss.png         training loss against step for the best settings, one panel each, a line per seed
  val_by_epoch.png       val LER after each epoch for the same settings: stuck seeds, runs still improving
  seeds.png              val LER of every seed of every setting
  quick_checks.png       the quick checks (val LER on the first val shots of r = 3, 10, 20, 30), best setting
  learning_by_round.png  per-round val error / MWPM's at every r after each epoch (best setting, its best seed)
  test_heatmap.png       per-round test error / MWPM's for every setting and every r (middle seed)
  per_round_ratio.png    the same against r for the best settings, next to the target
  within_target.png      how many trained round counts each seed has within the target
  per_round.csv          every run and every test r: errors, LER, per-round error, x MWPM, within the target
With --compare also: compare_val.png, compare_per_round.png, compare_train_loss.png.
Per-round error = (1 - (1 - 2 LER)^(1/r)) / 2. LER grows with r for any decoder; the per-round error does not, so
per-round error / MWPM's per-round error says how far behind MWPM a model is at each r. The target is 2x.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

import plots
import show_results
from config import EXPERIMENTS_DIR
from evaluate import result_rows
from plots import BLUE, GRID, INK, INK_2, ORANGE, SERIES, SURFACE, log_axis, pct
from train import per_round_error

# ======================================== SETTINGS ===========================================
PANELS = 8                                                                                      #settings in train_loss.png and val_by_epoch.png
TOP = 4                                                                                         #settings in per_round_ratio.png
SMOOTH = 5                                                                                      #running mean over this many points (1 point = 100 steps)
STUCK = 0.5                                                                                     #stuck seed: val LER above STUCK x always-0's
MAX_RATIO = 32                                                                                  #heatmap colours stop here (x MWPM)
# =============================================================================================
OTHER = INK_2                                                                                   #colour of the second run in the compare graphs


# ------------------------------------------------------------------------------- numbers
def load(exp_dir: Path) -> tuple[dict, list[dict]]:
    setup, results = show_results.load(exp_dir)
    for run in results:
        path = exp_dir / run["name"] / run["run_folder"] / "history.json"
        run["hist"] = json.loads(path.read_text()) if path.exists() else None                   #written when training finishes
    return setup, results


def settings(results: list[dict]) -> list[list[dict]]:                                          #runs grouped by setting, best val first
    groups: dict = {}
    for run in results:
        groups.setdefault((run["combo"], run["order"]), []).append(run)
    out = [sorted(g, key=lambda run: run["seed"]) for g in groups.values()]
    return sorted(out, key=lambda g: middle([run["val_ler"] for run in g]))


def middle(values: list[float]) -> float:                                                       #the middle seed: one stuck seed cannot move it
    return sorted(values)[len(values) // 2]


def name(g: list[dict], base_order: str) -> str:                                                #e.g. "x, y, t" or "x, y, t | unmixed"
    tokens = ", ".join(g[0]["features"][1:]) or "event only"
    return tokens if g[0]["order"] == base_order else f"{tokens} | {g[0]['order']}"


def cut(run: dict) -> int:                                                                      #last trained r
    return run["config"]["rounds"]["long_bucket"][1]


def target(run: dict) -> float:
    return run["config"]["evaluation"]["target_x_mwpm"]


def ratio(errors: float, b: dict, r: int) -> float:                                             #per-round error / MWPM's on the same shots
    eps_mwpm = per_round_error(b["mwpm"] / b["shots"], r)
    return per_round_error(errors / b["shots"], r) / eps_mwpm if eps_mwpm > 0 else math.inf


def middle_run(g: list[dict]) -> dict:                                                          #the seed with the middle val LER
    return sorted(g, key=lambda run: run["val_ler"])[len(g) // 2]


def run_ratio(run: dict, base: dict[int, dict]) -> dict[int, float]:                            #x MWPM per round at every test r
    return {r: ratio(run["test_errors"][str(r)], b, r) for r, b in base.items()}


def test_rows(run: dict, base: dict[int, dict]) -> list[dict]:                                  #evaluate.py's rows of one run
    return result_rows({int(r): n for r, n in run["test_errors"].items()}, base, target(run))


def within(run: dict, base: dict[int, dict]) -> int:                                            #trained r within the target
    return sum(row["within_target"] for row in test_rows(run, base) if row["r"] <= cut(run))


def smooth(values: list[float], k: int = SMOOTH) -> list[float]:                                #running mean of the last k points
    return [sum(values[max(0, i - k + 1):i + 1]) / len(values[max(0, i - k + 1):i + 1])
            for i in range(len(values))]


# ------------------------------------------------------------------------------- table and csv
def print_table(groups: list[list[dict]], base: dict, val_base: dict, base_order: str, run: str) -> None:
    a0 = show_results.reference(val_base, "always0")
    n_trained = len([r for r in base if r <= cut(groups[0][0])])
    print(f"\nEVERY SETTING | data {run} | ranked by the middle seed's val LER | target "
          f"{target(groups[0][0]):g}x MWPM per round")
    head = (f"{'setting':<40}{'val LER of each seed':>24}{'best epoch':>12}"
            f"{f'r within target (of {n_trained})':>27}{'end loss':>10}  note")
    print(head + "\n" + "-" * (len(head) + 16))
    for g in groups:
        vals = " ".join(f"{100 * run['val_ler']:.2f}" for run in g) + "%"
        epochs = "/".join(str(run["best_epoch"]) for run in g)
        hits = "/".join(str(within(run, base)) for run in g)
        losses = [run["hist"]["train_loss"][-1] for run in g if run["hist"]]
        loss = f"{sum(losses) / len(losses):.4f}" if losses else "n/a"
        stuck = sum(run["val_ler"] > STUCK * a0 for run in g)
        last = g[0]["config"]["training"]["num_epochs"]
        if middle([run["val_ler"] for run in g]) > 0.95 * a0:
            notes = ["no better than always-0"]
        else:
            notes = ([f"{stuck} stuck"] if stuck else []) + \
                    (["still improving"] if all(run["best_epoch"] == last for run in g) else [])
        print(f"{name(g, base_order):<40}{vals:>24}{epochs:>12}{hits:>27}{loss:>10}  {', '.join(notes)}")
    print(f"stuck = val LER above {STUCK:g} x always-0's ({pct(a0, 2)}). still improving = every seed kept "
          "its last epoch. end loss = mean training loss of the last 100 steps, mean over seeds.")


def write_csv(groups: list[list[dict]], base: dict[int, dict], path: Path) -> None:
    keys = ["r", "shots", "errors", "ler", "always0", "mwpm", "eps", "eps_mwpm", "x_mwpm", "within_target"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combination", "order", "features", "seed", "val_ler", "trained"] + keys)
        for g in groups:
            for run in g:
                for row in test_rows(run, base):
                    w.writerow([run["combo"], run["order"], " ".join(run["features"]), run["seed"],
                                f"{run['val_ler']:.6g}", row["r"] <= cut(run)] +
                               [f"{row[k]:.6g}" if isinstance(row[k], float) else row[k] for k in keys])


# ------------------------------------------------------------------------------- graph frame
def style(ax) -> None:                                                                          #the look of plots.new_figure
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grid_figure(n: int, cols: int, title: str, subtitle: str, row_h: float = 2.8, width: float = 13):
    rows = math.ceil(n / cols)
    height = 1.05 + row_h * rows
    fig, axes = plots.plt.subplots(rows, cols, figsize=(width, height), facecolor=SURFACE,
                                   layout="constrained", squeeze=False)
    fig.text(0.012, 1 - 0.18 / height, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.52 / height, subtitle, fontsize=9.5, color=INK_2, va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.9 / height))                               #plots start below the title
    axes = [ax for row in axes for ax in row]
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        style(ax)
    return fig, axes[:n]


def save(fig, path: Path) -> str:
    fig.savefig(path, dpi=150)
    plots.plt.close(fig)
    return path.name


def ratio_axis(ax, which: str, hi: float) -> None:                                              #log axis labelled 1x, 2x, 5x ...
    getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"set_{which}lim")(0.8, hi)
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator([v for v in (1, 2, 5, 10, 20, 50, 100) if v <= hi]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))


def heat(ax, matrix: np.ndarray, x: list[int], y_labels: list[str], goal: float):
    """Coloured cells: blue = within the target, white = at it, orange = behind it (log of x MWPM)."""
    cmap = mcolors.LinearSegmentedColormap.from_list("x_mwpm", [BLUE, SURFACE, ORANGE])
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=math.log2(goal), vmax=math.log2(MAX_RATIO))
    values = np.log2(np.clip(matrix, 1, MAX_RATIO))                                             #below 1x looks like 1x
    mesh = ax.pcolormesh(np.arange(len(x) + 1), np.arange(len(y_labels) + 1), values, cmap=cmap,
                         norm=norm, edgecolors=SURFACE, linewidth=0.4)
    ax.set_yticks(np.arange(len(y_labels)) + 0.5, y_labels, color=INK, fontsize=8.5)
    every = max(1, len(x) // 12)
    ax.set_xticks(np.arange(0, len(x), every) + 0.5, [str(v) for v in x[::every]], fontsize=8.5)
    ax.grid(False)
    ax.invert_yaxis()                                                                           #first row at the top
    bar = ax.figure.colorbar(mesh, ax=ax, fraction=0.03, pad=0.01)
    ticks = [v for v in (1, 2, 4, 8, 16, 32, 64) if v <= MAX_RATIO]
    bar.set_ticks([math.log2(v) for v in ticks], labels=[f"{v}x" for v in ticks])
    bar.ax.tick_params(colors=INK_2, labelsize=8, length=0)
    bar.outline.set_visible(False)
    return mesh


# ------------------------------------------------------------------------------- graphs of one run
def plot_train_loss(groups: list[list[dict]], base_order: str, path: Path) -> Optional[str]:
    groups = [g for g in groups if g[0]["order"] == base_order and all(run["hist"] for run in g)]
    groups = groups[:PANELS]
    if not groups:
        return None
    fig, axes = grid_figure(len(groups), 4, "Training loss of the best settings",
                            f"BCE loss, mean of each 100 steps, running mean over {SMOOTH} points, a line "
                            "per seed. Dotted lines: end of each epoch. Log scale.")
    values = [v for g in groups for run in g for v in smooth(run["hist"]["train_loss"])]
    for ax, g in zip(axes, groups):
        for run, color in zip(g, SERIES):
            h = run["hist"]
            ax.plot(h["step"], smooth(h["train_loss"]), color=color, linewidth=1.2,
                    label=f"seed {run['seed']}")
        for e in g[0]["hist"]["epochs"]:
            ax.axvline(e["step"], color=INK_2, alpha=0.35, linestyle=":", linewidth=1)
        log_axis(ax, "y", min(values) / 1.2, max(values) * 1.2, percent=False)
        ax.set_title(f"{name(g, base_order)}   val {pct(middle([r['val_ler'] for r in g]), 2)}",
                     loc="left", fontsize=9.5, color=INK)
    for ax in axes[-4:]:
        ax.set_xlabel("step", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    return save(fig, path)


def plot_val_by_epoch(groups: list[list[dict]], val_base: dict, base_order: str, path: Path) -> str:
    groups = [g for g in groups if g[0]["order"] == base_order][:PANELS]
    a0, mw = show_results.reference(val_base, "always0"), show_results.reference(val_base, "mwpm")
    fig, axes = grid_figure(len(groups), 4, "Val LER after every epoch",
                            "every val shot, mean over r, a line per seed; the large dot is the epoch kept. "
                            "Dashed: always-0, dotted: MWPM. Log scale.")
    values = [e["val_ler"] for g in groups for run in g for e in run["epochs"]] + [a0, mw]
    for ax, g in zip(axes, groups):
        for run, color in zip(g, SERIES):
            ep = run["epochs"]
            ax.plot([e["epoch"] for e in ep], [e["val_ler"] for e in ep], color=color, linewidth=1.5,
                    marker="o", markersize=3.5, label=f"seed {run['seed']}")
            best = ep[run["best_epoch"] - 1]
            ax.plot(best["epoch"], best["val_ler"], "o", color=color, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(a0, color=INK_2, linestyle="--", linewidth=1.1)
        ax.axhline(mw, color=INK, linestyle=":", linewidth=1.1)
        log_axis(ax, "y", min(values) / 1.3, max(values) * 1.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(name(g, base_order), loc="left", fontsize=9.5, color=INK)
    for ax in axes[-4:]:
        ax.set_xlabel("epoch", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    return save(fig, path)


def plot_seeds(groups: list[list[dict]], val_base: dict, base_order: str, path: Path) -> str:
    a0, mw = show_results.reference(val_base, "always0"), show_results.reference(val_base, "mwpm")
    fig, (ax,) = plots.new_figure(11, 1.9 + 0.36 * len(groups), "Val LER of every seed",
                                  f"a dot per seed, orange = stuck (val LER above {STUCK:g} x always-0's), "
                                  "ranked by the middle seed. Log scale.")
    y = list(range(len(groups)))[::-1]
    values = [a0, mw]
    for g, yy in zip(groups, y):
        vals = [run["val_ler"] for run in g]
        values += vals
        ax.plot([min(vals), max(vals)], [yy, yy], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        colors = [ORANGE if v > STUCK * a0 else BLUE for v in vals]
        ax.scatter(vals, [yy] * len(vals), s=45, color=colors, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_yticks(y, [name(g, base_order) for g in groups], color=INK, fontsize=8.5)
    ax.grid(False, axis="y")
    log_axis(ax, "x", min(values) / 1.5, max(values) * 1.8)
    plots.reference_line(ax, a0, "always-0", True, "--")
    plots.reference_line(ax, mw, "MWPM", True, ":")
    ax.set_ylim(-0.7, len(groups) - 0.4)
    ax.set_xlabel("val LER", color=INK_2)
    return save(fig, path)


def plot_quick_checks(g: list[dict], base_order: str, path: Path) -> Optional[str]:
    if not all(run["hist"] for run in g):
        return None
    rs = sorted(int(r) for r in g[0]["hist"]["val"])
    shots = g[0]["config"]["training"]["eval_check_shots"]
    fig, axes = grid_figure(len(rs), len(rs), f"Quick checks during training: {name(g, base_order)}",
                            f"val LER on the first {shots:,} val shots of each r, every 100 steps, running "
                            f"mean over {SMOOTH} points, a line per seed. Few shots, so it is noisy.",
                            row_h=3.4)
    for ax, r in zip(axes, rs):
        for run, color in zip(g, SERIES):
            h = run["hist"]
            ax.plot(h["step"], [100 * v for v in smooth(h["val"][str(r)])], color=color, linewidth=1.2,
                    label=f"seed {run['seed']}")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_title(f"r = {r}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("step", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    return save(fig, path)


def plot_learning_by_round(g: list[dict], val_base: dict, base_order: str, path: Path) -> Optional[str]:
    run = min(g, key=lambda x: x["val_ler"])                                                    #the best seed of the best setting
    if not run["hist"] or "per_round" not in run["hist"]["epochs"][0]:
        return None
    rs = sorted(val_base)
    epochs = run["hist"]["epochs"]
    matrix = np.array([[ratio(e["per_round"][str(r)], val_base[r], r) for e in epochs] for r in rs])
    fig, (ax,) = plots.new_figure(10, 1.6 + 0.2 * len(rs),
                                  f"How each round count is learned: {name(g, base_order)}, seed {run['seed']}",
                                  "per-round val error / MWPM's after every epoch. Blue = within the "
                                  f"{target(run):g}x target, orange = behind it. The curriculum adds r in "
                                  "the first epochs.")
    heat(ax, matrix, [e["epoch"] for e in epochs], [str(r) for r in rs], target(run))
    ax.set_xlabel("epoch", color=INK_2)
    ax.set_ylabel("r", color=INK_2)
    return save(fig, path)


def plot_test_heatmap(groups: list[list[dict]], base: dict, base_order: str, path: Path) -> str:
    rs = sorted(base)
    values = [run_ratio(middle_run(g), base) for g in groups]
    matrix = np.array([[v[r] for r in rs] for v in values])
    fig, (ax,) = plots.new_figure(13, 1.7 + 0.32 * len(groups), "Where each setting falls behind MWPM",
                                  "per-round test error / MWPM's at every r, middle seed of each setting. "
                                  f"Blue = within the {target(groups[0][0]):g}x target, orange = behind it. "
                                  "Right of the line: never trained on.")
    heat(ax, matrix, rs, [name(g, base_order) for g in groups], target(groups[0][0]))
    edge = len([r for r in rs if r <= cut(groups[0][0])])
    ax.axvline(edge, color=INK, linewidth=1.5)
    ax.set_xlabel("rounds r", color=INK_2)
    return save(fig, path)


def ratio_lines(ax, lines: list[dict], goal: float, edge: int) -> None:
    """Per-round error / MWPM's against r; dict keys: label, values {r: x}, color, width."""
    hi = max(v for ln in lines for v in ln["values"].values() if v < math.inf)
    for ln in lines:
        rs = sorted(ln["values"])
        ax.plot(rs, [ln["values"][r] for r in rs], color=ln["color"], linewidth=ln.get("width", 2),
                marker="o", markersize=2.5, label=ln["label"])
    ax.axhline(goal, color=INK, linestyle="--", linewidth=1.2)
    ax.annotate(f"target {goal:g}x", (1, goal), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.axhline(1, color=INK_2, linestyle=":", linewidth=1.1)
    ax.axvspan(edge + 0.5, max(max(ln["values"]) for ln in lines) + 0.5, color=GRID, alpha=0.45, zorder=0)
    ax.annotate("never trained on", (edge + 1, 1), xycoords=("data", "axes fraction"), xytext=(0, -14),
                textcoords="offset points", fontsize=8.5, color=INK_2)
    ratio_axis(ax, "y", max(hi * 1.3, goal * 2))
    ax.set_xlabel("rounds r", color=INK_2)
    ax.set_ylabel("per-round error / MWPM's", color=INK_2)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)


def plot_per_round_ratio(groups: list[list[dict]], base: dict, base_order: str, path: Path) -> str:
    edge, goal = cut(groups[0][0]), target(groups[0][0])
    n = len([r for r in base if r <= edge])
    lines = []
    for g, color in zip(groups[:TOP], SERIES):
        values = run_ratio(middle_run(g), base)
        hits = sum(values[r] <= goal for r in base if r <= edge)
        lines.append({"label": f"{name(g, base_order)}: {hits} of {n} trained r within {goal:g}x",
                      "values": values, "color": color})
    fig, (ax,) = plots.new_figure(11, 5.8, "How far behind MWPM each round count is",
                                  f"per-round test error / MWPM's, the {len(lines)} best settings on val, "
                                  "middle seed of each (one stuck seed cannot pull it up). Log scale.")
    ratio_lines(ax, lines, goal, edge)
    return save(fig, path)


def plot_within_target(groups: list[list[dict]], base: dict, base_order: str, path: Path) -> str:
    n = len([r for r in base if r <= cut(groups[0][0])])
    fig, (ax,) = plots.new_figure(10, 1.9 + 0.36 * len(groups), "Round counts within the target",
                                  f"trained round counts (of {n}) whose per-round test error is within "
                                  f"{target(groups[0][0]):g}x MWPM's, a dot per seed")
    y = list(range(len(groups)))[::-1]
    for g, yy in zip(groups, y):
        hits = [within(run, base) for run in g]
        ax.plot([min(hits), max(hits)], [yy, yy], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        ax.scatter(hits, [yy] * len(hits), s=45, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_yticks(y, [name(g, base_order) for g in groups], color=INK, fontsize=8.5)
    ax.grid(False, axis="y")
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.7, len(groups) - 0.4)
    ax.set_xlabel("trained round counts within the target", color=INK_2)
    return save(fig, path)


# ------------------------------------------------------------------------------- two runs compared
def run_label(results: list[dict], folder: str) -> str:                                         #e.g. "d_model_16 (d_model 16)"
    d = {run["config"]["training"]["encoder"]["d_model"] for run in results}
    return f"{folder} (d_model {d.pop()})" if len(d) == 1 else folder


def plot_compare_val(runs: list[tuple[str, list[list[dict]], str]], path: Path) -> str:
    """Val LER of every feature combination (base order) in both runs: dot = middle seed, line = best to worst."""
    (label1, g1, order1), (label2, g2, order2) = runs
    first = [g for g in g1 if g[0]["order"] == order1]
    second = {g[0]["combo"]: g for g in g2 if g[0]["order"] == order2}
    pairs = [(g, second[g[0]["combo"]]) for g in first if g[0]["combo"] in second]
    fig, (ax,) = plots.new_figure(11, 2.2 + 0.42 * len(pairs), "Val LER of every feature combination",
                                  "dot = middle seed, line = best to worst seed, same data and settings "
                                  "apart from the model. Log scale.")
    y = np.arange(len(pairs))[::-1]
    values = []
    for k, (label, color, dy) in enumerate(((label1, BLUE, 0.13), (label2, OTHER, -0.13))):
        for (a, b), yy in zip(pairs, y):
            vals = [run["val_ler"] for run in (a, b)[k]]
            values += vals
            ax.plot([min(vals), max(vals)], [yy + dy] * 2, color=color, alpha=0.45, linewidth=2)
            ax.scatter(middle(vals), yy + dy, s=40, color=color, edgecolor=SURFACE, linewidth=1, zorder=3,
                       label=label if yy == y[0] else None)
    ax.set_yticks(y, [name(a, order1) for a, _ in pairs], color=INK, fontsize=8.5)
    ax.grid(False, axis="y")
    log_axis(ax, "x", min(values) / 1.5, max(values) * 1.5)
    ax.set_ylim(-0.7, len(pairs) - 0.4)
    ax.set_xlabel("val LER", color=INK_2)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    return save(fig, path)


def plot_compare_per_round(runs: list[tuple[str, list[list[dict]], str]], base: dict, path: Path) -> str:
    lines = []
    for (label, groups, order), color in zip(runs, (BLUE, OTHER)):
        g = groups[0]
        lines.append({"label": f"{label}: {name(g, order)}", "values": run_ratio(middle_run(g), base),
                      "color": color})
    fig, (ax,) = plots.new_figure(11, 5.8, "Best setting of each run against MWPM",
                                  "per-round test error / MWPM's, middle seed of each run's best setting. Log scale.")
    ratio_lines(ax, lines, target(runs[0][1][0][0]), cut(runs[0][1][0][0]))
    return save(fig, path)


def plot_compare_train_loss(runs: list[tuple[str, list[list[dict]], str]], path: Path) -> Optional[str]:
    fig, (ax,) = plots.new_figure(11, 5.4, "Training loss of the best setting of each run",
                                  f"a line per seed, running mean over {SMOOTH} points. Log scale.")
    values = []
    for (label, groups, order), color in zip(runs, (BLUE, OTHER)):
        for k, run in enumerate(groups[0]):
            if run["hist"]:
                y = smooth(run["hist"]["train_loss"])
                values += y
                ax.plot(run["hist"]["step"], y, color=color, linewidth=1.1, alpha=0.85,
                        label=f"{label}: {name(groups[0], order)}" if k == 0 else None)
    if not values:
        plots.plt.close(fig)
        return None
    log_axis(ax, "y", min(values) / 1.2, max(values) * 1.2, percent=False)
    ax.set_xlabel("step", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return save(fig, path)


# ------------------------------------------------------------------------------- main
def main(exp_dir: Optional[Path] = None, other: Optional[Path] = None) -> Path:
    exp_dir = show_results.newest() if exp_dir is None else Path(exp_dir)
    setup, results = load(exp_dir)
    if not results:
        raise SystemExit(f"no finished runs in {exp_dir} yet")
    base, val_base, order = setup["baselines"]["test"], setup["baselines"]["val"], setup["base_order"]
    groups = settings(results)
    out = exp_dir / "explore"
    out.mkdir(exist_ok=True)
    print_table(groups, base, val_base, order, exp_dir.name)
    write_csv(groups, base, out / "per_round.csv")
    best = next(g for g in groups if g[0]["order"] == order)                                    #best feature combination on val
    saved = ["per_round.csv",
             plot_train_loss(groups, order, out / "train_loss.png"),
             plot_val_by_epoch(groups, val_base, order, out / "val_by_epoch.png"),
             plot_seeds(groups, val_base, order, out / "seeds.png"),
             plot_quick_checks(best, order, out / "quick_checks.png"),
             plot_learning_by_round(best, val_base, order, out / "learning_by_round.png"),
             plot_test_heatmap(groups, base, order, out / "test_heatmap.png"),
             plot_per_round_ratio(groups, base, order, out / "per_round_ratio.png"),
             plot_within_target(groups, base, order, out / "within_target.png")]
    missing = sum(run["hist"] is None for run in results)
    if missing:
        print(f"note: {missing} runs have no history.json, so they are left out of the training curves")
    if other is not None:
        setup2, results2 = load(Path(other))
        if not results2:
            raise SystemExit(f"no finished runs in {other}")
        if setup2["baselines"]["test"] != base:
            print(f"note: {Path(other).name} used other test data, so the per-round graph compares "
                  "against this run's MWPM")
        runs = [(run_label(results, exp_dir.name), groups, order),
                (run_label(results2, Path(other).name), settings(results2), setup2["base_order"])]
        saved += [plot_compare_val(runs, out / "compare_val.png"),
                  plot_compare_per_round(runs, base, out / "compare_per_round.png"),
                  plot_compare_train_loss(runs, out / "compare_train_loss.png")]
    print(f"\nsaved {', '.join(s for s in saved if s)} in {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", help="a folder name in experiments/ (default: the newest)")
    parser.add_argument("--compare", metavar="OTHER", help="a second folder name in experiments/ to compare")
    args = parser.parse_args()
    main(None if args.run is None else EXPERIMENTS_DIR / args.run,
         None if args.compare is None else EXPERIMENTS_DIR / args.compare)
