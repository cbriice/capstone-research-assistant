"""Run training, then write the analysis plots from analyze_train.ipynb to disk as PNGs.

Run from anywhere (it chdirs to code/ so train.py's relative paths resolve):
    python3 claude/plot_training.py            # from code/
    python3 claude/plot_training.py --tag cos  # optional suffix on the output folder name

Output (git-tracked): code/train_plots/<timestamp>[_tag]/
    loss.png      val BCE per round family vs eval step
    ler.png       val LER per round family vs eval step
    metrics.json  raw lers/losses + training config, so plots can be redrawn without retraining
"""

import argparse, json, os, sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed over remote session
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent.parent
os.chdir(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))

from pipeline.config import load_config
from pipeline.train import run_training

OUT_ROOT = CODE_DIR / "train_plots"

# fixed categorical order (series color follows round family, never rank)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_series(series: dict, num_epochs: int, ylabel: str, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    n = len(next(iter(series.values())))
    k = n / num_epochs                                   # eval points per epoch
    for e in range(1, num_epochs):
        ax.axvline(e * k - 0.5, color=INK_2, lw=1, ls="--", alpha=0.5)

    for (r, ys), c in zip(series.items(), COLORS):
        ax.plot(range(len(ys)), ys, color=c, lw=2, label=f"r={r}")

    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("eval step (every 100 train steps; dashed = epoch boundary)", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.08)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. 'cosine'")
    args = ap.parse_args()

    cfg = load_config()
    E = cfg["training"]["num_epochs"]

    lers, losses, total_steps = run_training()

    name = datetime.now().strftime("%Y-%m-%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    plot_series(losses, E, "val BCE (nats)", "Val loss per round family", out / "loss.png")
    plot_series(lers, E, "logical error rate", "Val LER per round family", out / "ler.png")

    (out / "metrics.json").write_text(json.dumps({
        "total_steps": total_steps,
        "config": cfg,
        "lers": {str(r): v for r, v in lers.items()},
        "losses": {str(r): v for r, v in losses.items()},
    }, indent=2))

    print(f"wrote plots to {out.relative_to(CODE_DIR.parent)}")


if __name__ == "__main__":
    main()
