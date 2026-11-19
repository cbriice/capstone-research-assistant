"""Shared look of every graph (evaluate.py and show_results.py): colours, figure frame, log axes, labels."""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")                                                                           #no window: only write PNG files
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1f9d6a", "#8a5cd1"                        #series colours, in this order
SERIES = (BLUE, ORANGE, GREEN, PURPLE)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"                          #text, secondary text, grid lines, background


def pct(x: Optional[float], digits: int = 3) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def new_figure(width: float, height: float, title: str, subtitle: str, panels: int = 1):
    fig, axes = plt.subplots(1, panels, figsize=(width, height), facecolor=SURFACE, layout="constrained")
    fig.text(0.012, 1 - 0.18 / height, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.52 / height, subtitle, fontsize=9.5, color=INK_2, va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.85 / height))                              #plots start below the title and subtitle
    axes = list(axes) if panels > 1 else [axes]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axes


def log_ticks(lo: float, hi: float) -> list[float]:                                             #tick values 1, 2, 5 x 10^k, more if the range is narrow
    for steps in ((1, 2, 5), (1, 2, 3, 5, 7), range(1, 10)):
        ticks = [m * 10.0 ** k for k in range(-7, 3) for m in steps if lo <= m * 10.0 ** k <= hi]
        if len(ticks) >= 3:
            return ticks
    return ticks


def log_axis(ax, which: str, lo: float, hi: float, percent: bool = True) -> None:               #which = "x" or "y"
    getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"set_{which}lim")(lo, hi)
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator(log_ticks(lo, hi)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%" if percent else f"{v:g}"))


def label(ax, text: str, xy, dx: float, ha: str = "left", bold: bool = False) -> None:          #text with a background box
    ax.annotate(text, xy, xytext=(dx, 0), textcoords="offset points", va="center", ha=ha, fontsize=8.5, color=INK,
                fontweight="bold" if bold else "normal", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5), zorder=4)


def reference_line(ax, value: Optional[float], name: str, vertical: bool, style: str) -> None:
    if value is None:
        return
    color = INK if name == "MWPM" else INK_2
    if vertical:                                                                                #a labelled line across the plot
        ax.axvline(value, color=color, linestyle=style, linewidth=1.3, zorder=1)
        ax.annotate(f"{name} {pct(value, 2)}", (value, 1), xycoords=("data", "axes fraction"), xytext=(4, 8),
                    textcoords="offset points", fontsize=8.5, color=color, va="bottom")
    else:
        ax.axhline(value, color=color, linestyle=style, linewidth=1.3, zorder=1, label=name)


def plot_by_rounds(lines: list[dict], cut: int, path: Path, title: str, subtitle: str, ylabel: str) -> None:
    """One value per round count for every line; dict keys: name, values {r: v}, color, style, and optional se {r: v}.
    A shaded band marks the round counts the model never trained on (r > cut)."""
    fig, (ax,) = new_figure(10, 5.8, title, subtitle)
    values, ends = [], []
    for ln in lines:
        rs = [r for r in sorted(ln["values"]) if ln["values"][r] > 0]                           #a log axis cannot show 0
        if not rs:
            continue
        y = [ln["values"][r] for r in rs]
        values += y
        model = ln["style"] == "-"
        if ln.get("se"):
            ax.fill_between(rs, [max(v - ln["se"][r], v / 3) for v, r in zip(y, rs)],           #+- 1 standard error, as a band
                            [v + ln["se"][r] for v, r in zip(y, rs)], color=ln["color"], alpha=0.15, linewidth=0)
        ax.plot(rs, y, color=ln["color"], linestyle=ln["style"], linewidth=2 if model else 1.4, label=ln["name"],
                marker="o" if model else None, markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.6)
        ends.append([y[-1], rs[-1], ln["name"]])
    lo, hi = min(values) / 1.6, max(values) * 1.6
    gap = (hi / lo) ** 0.045                                                                    #line labels at least 4.5% of the axis apart
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] * gap)                                      #push a label up if it would overlap the one below
    for y_end, r_end, name in ends:
        label(ax, name, (r_end, y_end), 7)
    rmax = max(r for ln in lines for r in ln["values"])
    if rmax > cut:
        ax.axvspan(cut + 0.5, rmax + 0.5, color=GRID, alpha=0.45, zorder=0, linewidth=0)
        ax.annotate(f"never trained on (r > {cut})", (cut + 1, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points", fontsize=8.5, color=INK_2)
    log_axis(ax, "y", lo, hi)
    ax.set_xlim(0, rmax * 1.22)                                                                 #room for the line labels on the right
    ax.set_xticks(list(range(0, rmax + 1, 5)))
    ax.set_xlabel("rounds r", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)
