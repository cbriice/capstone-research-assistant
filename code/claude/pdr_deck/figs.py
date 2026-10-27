"""Figures for the PDR deck. Run with code/.venv/bin/python (needs stim, pymatching, matplotlib)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)

SCARLET = "#BB0000"
INK = "#212121"
GRAY = "#636D6E"
LIGHT = "#F2F2F2"
MID = "#C8CCCD"
XCOL = "#F4C26B"   # X-type checks (measure X parity, catch Z errors)
ZCOL = "#8DB3D6"   # Z-type checks (measure Z parity, catch X errors)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "savefig.dpi": 300,
})

# Carson's labels (memory: carson-qubit-indexing). Stim coordinates, y points down.
DATA = {f"d{i}": (1 + 2 * (i % 3), 1 + 2 * (i // 3)) for i in range(9)}
CHECKS = {  # name: (type, (x, y), data qubits)
    "X0": ("X", (2, 0), ["d0", "d1"]),
    "Z2": ("Z", (2, 2), ["d0", "d1", "d3", "d4"]),
    "X2": ("X", (4, 2), ["d1", "d2", "d4", "d5"]),
    "Z3": ("Z", (6, 2), ["d2", "d5"]),
    "Z0": ("Z", (0, 4), ["d3", "d6"]),
    "X1": ("X", (2, 4), ["d3", "d4", "d6", "d7"]),
    "Z1": ("Z", (4, 4), ["d4", "d5", "d7", "d8"]),
    "X3": ("X", (4, 6), ["d7", "d8"]),
}
READING_ORDER = ["X0", "Z2", "X2", "Z3", "Z0", "X1", "Z1", "X3"]  # sorted by (y, x), as the pipeline does


def plaquette(ax, name, fired=False, faded=False, label=True, z=0.0, fs=12):
    typ, (cx, cy), qs = CHECKS[name]
    col = XCOL if typ == "X" else ZCOL
    alpha = 0.35 if faded else 1.0
    if len(qs) == 4:
        pts = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx + 1, cy + 1), (cx - 1, cy + 1)]
        ax.add_patch(Polygon(pts, closed=True, fc=col, ec="white", lw=2, alpha=alpha, zorder=1))
    else:
        (x1, y1), (x2, y2) = DATA[qs[0]], DATA[qs[1]]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if y1 == y2:  # horizontal pair: bulge up or down
            th1, th2 = (180, 360) if cy < my else (0, 180)
        else:
            th1, th2 = (90, 270) if cx < mx else (270, 90)
        ax.add_patch(Wedge((mx, my), 1.0, th1, th2, fc=col, ec="white", lw=2, alpha=alpha, zorder=1))
    if fired:
        ax.add_patch(Circle((cx, cy), 0.36, fc=SCARLET, ec="white", lw=1.5, zorder=5))
    if label:
        tx, ty = cx, cy
        if len(qs) == 2:  # nudge label into the half-disk
            tx += 0.45 * np.sign(cx - 3) if cx in (0, 6) else 0
            ty += 0.45 * np.sign(cy - 3) if cy in (0, 6) else 0
        ax.text(tx, ty, name, ha="center", va="center", fontsize=fs, fontweight="bold",
                color="white" if fired else INK, zorder=6, alpha=1 if not faded or fired else 0.6)


def data_qubits(ax, err=None, err_label=None, fs=11, r=0.42):
    for q, (x, y) in DATA.items():
        hit = q == err
        if hit and err_label:
            ax.add_patch(Circle((x, y), r * 1.12, fc=INK, ec="white", lw=2, zorder=7))
            ax.text(x, y, err_label, ha="center", va="center", fontsize=fs + 5, fontweight="bold",
                    color="white", zorder=8)
            continue
        ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=1.4, zorder=3))
        ax.text(x, y, q, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def lattice_axes(ax, pad=1.3):
    ax.set_xlim(-pad, 6 + pad)
    ax.set_ylim(6 + pad, -pad)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------- 1. lattice
def fig_lattice():
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for n in CHECKS:
        plaquette(ax, n, fs=13)
    data_qubits(ax, fs=12)
    lattice_axes(ax, pad=1.15)
    save(fig, "lattice.png")


# ---------------------------------------------------------------- 2. error types
def fig_error_types():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    cases = [
        ("X", {"Z2", "Z1"}),
        ("Z", {"X1", "X2"}),
        ("Y", {"Z2", "Z1", "X1", "X2"}),
    ]
    for ax, (e, fired) in zip(axes, cases):
        for n in CHECKS:
            plaquette(ax, n, fired=n in fired, faded=n not in fired, fs=11)
        data_qubits(ax, err="d4", err_label=e, fs=10)
        lattice_axes(ax, pad=1.1)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "error_types.png")


def fig_meas_error():
    """One check over five rounds: a misread shows up as two detection events in time."""
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    meas = [0, 0, 1, 0, 0]
    ev = [0, 0, 1, 1, 0]
    for t in range(5):
        for row, vals, lab in [(0, meas, "measured Z1"), (1, ev, "detection event")]:
            v = vals[t]
            fc = SCARLET if (v and row == 1) else (INK if v else "white")
            ax.add_patch(Rectangle((t, -row), 0.86, 0.86, fc=fc, ec=MID, lw=1.2))
            ax.text(t + 0.43, -row + 0.43, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v else GRAY, fontweight="bold")
        ax.text(t + 0.43, 1.15, f"r{t + 1}", ha="center", va="center", fontsize=11, color=GRAY)
    ax.text(-0.15, 0.43, "Z1 reads", ha="right", va="center", fontsize=12)
    ax.text(-0.15, -0.57, "event", ha="right", va="center", fontsize=12, color=SCARLET, fontweight="bold")
    ax.set_xlim(-1.9, 5.0)
    ax.set_ylim(-1.3, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "meas_error.png")


# ---------------------------------------------------------------- 3. Hc = s
def fig_hcs():
    zs = ["Z0", "Z1", "Z2", "Z3"]
    H = np.array([[1 if f"d{j}" in CHECKS[z][2] else 0 for j in range(9)] for z in zs])
    c = np.zeros(9, int)
    c[4] = 1
    s = H @ c % 2
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    cs = 0.62

    def cell(x, y, v, hl=False):
        fc = (SCARLET if hl else INK) if v else "white"
        ax.add_patch(Rectangle((x, y), cs * 0.92, cs * 0.92, fc=fc, ec=MID, lw=1))
        ax.text(x + cs * 0.46, y + cs * 0.46, str(v), ha="center", va="center", fontsize=13,
                color="white" if v else "#9AA2A4", fontweight="bold")

    x0 = 0.9
    yc = 4 * cs + 0.75   # c row sits above H, aligned with its columns
    for j in range(9):
        cell(x0 + j * cs, yc, c[j], hl=True)
        ax.text(x0 + j * cs + cs * 0.46, yc + cs + 0.08, f"d{j}", ha="center", va="bottom", fontsize=12,
                color=SCARLET if j == 4 else GRAY, fontweight="bold" if j == 4 else None)
    ax.text(x0 - 0.12, yc + cs * 0.46, "c", ha="right", va="center", fontsize=16, fontweight="bold",
            fontstyle="italic")
    ax.text(x0 + 9 * cs + 0.2, yc + cs * 0.46, "error: X on d4 (unknown)", ha="left", va="center",
            fontsize=13, color=SCARLET)
    for i, z in enumerate(zs):
        y = (3 - i) * cs
        ax.text(x0 - 0.12, y + cs * 0.46, z, ha="right", va="center", fontsize=13, fontweight="bold")
        for j in range(9):
            cell(x0 + j * cs, y, H[i, j], hl=(j == 4))
    ax.add_patch(Rectangle((x0 + 4 * cs - 0.05, -0.05), cs + 0.04, 4 * cs + 0.04, fc="none", ec=SCARLET, lw=2.2))
    ax.text(x0 - 0.75, 2 * cs, "H", ha="right", va="center", fontsize=20, fontweight="bold", fontstyle="italic")
    ax.text(x0 + 4.5 * cs, -0.25, "H: which checks see which qubit", ha="center", va="top", fontsize=13,
            color=GRAY)

    xs = x0 + 9 * cs + 1.3
    ax.add_patch(FancyArrowPatch((x0 + 9 * cs + 0.1, 2 * cs), (xs - 0.15, 2 * cs), arrowstyle="-|>",
                                 mutation_scale=18, color=INK, lw=1.6))
    for i, z in enumerate(zs):
        y = (3 - i) * cs
        cell(xs, y, s[i], hl=True)
        ax.text(xs + cs + 0.08, y + cs * 0.46, z, ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(xs + cs * 0.46, -0.25, "s (observed)", ha="center", va="top", fontsize=13, color=GRAY)
    ax.text(xs + cs * 0.46, 4 * cs + 0.12, "s", ha="center", va="bottom", fontsize=16, fontweight="bold",
            fontstyle="italic")
    ax.set_xlim(-0.3, xs + 1.4)
    ax.set_ylim(-0.75, yc + cs + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "hcs.png")


# ---------------------------------------------------------------- 4. equations
def eq_png(name, tex, fs=22, color=INK, w=8, h=0.8):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.0, 0.5, tex, fontsize=fs, color=color, va="center", ha="left")
    save(fig, name)


def fig_equations():
    eq_png("eq_prob.png", r"$P(c)\;=\;\prod_{j}\;p_j^{\,c_j}\,(1-p_j)^{\,1-c_j}$", fs=24, w=6)
    eq_png("eq_opt.png", r"$\hat{\ell}\;=\;\arg\max_{\ell\,\in\,\{0,1\}}\;\sum_{c\,:\;Hc\,=\,s,\;\;Lc\,=\,\ell}\;P(c)$",
           fs=24, w=7.5, h=1.0)
    eq_png("eq_mwpm.png", r"$\hat{c}\;=\;\arg\min_{c\,:\;Hc\,=\,s}\;\sum_j w_j\,c_j\,,\qquad w_j=\ln\frac{1-p_j}{p_j}$",
           fs=22, w=7.5, h=1.0)
    eq_png("eq_hcs.png", r"$H\,c\;=\;s\;\;(\mathrm{mod}\;2)$", fs=28, w=4)


# ---------------------------------------------------------------- 5. space-time MWPM
def fig_spacetime(seed_shot=145):
    import stim
    from pymatching import Matching
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r, p = 4, 0.006
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=r,
                               after_clifford_depolarization=p, before_round_data_depolarization=p,
                               before_measure_flip_probability=p, after_reset_flip_probability=p)
    m = Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    co = c.get_detector_coordinates()
    D, O = c.compile_detector_sampler(seed=7).sample(20000, separate_observables=True)
    shot = D[seed_shot]
    edges = m.decode_to_edges_array(shot)
    pred = m.decode(shot)[0]
    defects = np.nonzero(shot)[0]
    print("defects", [tuple(co[k]) for k in defects], "edges", edges.tolist(), "pred", pred, "label", O[seed_shot, 0])

    fig = plt.figure(figsize=(7.6, 7.2))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    T = sorted({co[k][2] for k in range(len(co))})
    zsc = 3.2  # vertical spacing between rounds

    def P(x, y, t):  # lattice x,y -> 3d
        return (x, 6 - y, t * zsc)

    for t in T:
        for n, (typ, (cx, cy), qs) in CHECKS.items():
            col = XCOL if typ == "X" else ZCOL
            if len(qs) == 4:
                pts = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx + 1, cy + 1), (cx - 1, cy + 1)]
            else:
                (x1, y1), (x2, y2) = DATA[qs[0]], DATA[qs[1]]
                pts = [(x1, y1), (x2, y2), (cx, cy)]
            poly = Poly3DCollection([[P(x, y, t) for x, y in pts]], facecolor=col, edgecolor="white",
                                    alpha=0.6, linewidths=0.8)
            poly.set_zorder(0)
            ax.add_collection3d(poly)
        for n, (typ, (cx, cy), _) in CHECKS.items():
            ax.scatter(*P(cx, cy, t), s=26, c="white", edgecolors=GRAY, linewidths=0.8, depthshade=False, zorder=1)
        lab = f"Round {int(t) + 1}" if t < r else "Final readout"
        ax.text(*P(-0.5, 6.0, t), lab, fontsize=15, ha="right", va="center", zorder=10)

    kinds = []
    for a, b in edges:
        A = co[a]
        if b == -1:
            # draw to the nearest boundary of the check's sector
            x, y, t = A
            typ = "Z" if any(CHECKS[n][1] == (x, y) and CHECKS[n][0] == "Z" for n in CHECKS) else "X"
            if typ == "Z":  # Z checks: X-error strings end on left/right boundaries
                bx, by = (-0.9, y) if x <= 3 else (6.9, y)
            else:
                bx, by = (x, -0.9) if y <= 3 else (x, 6.9)
            xs, ys, zs = zip(P(x, y, t), P(bx, by, t))
            lab = "X" if typ == "Z" else "Z"
            mid = P((x + bx) / 2, (y + by) / 2, t)
            ax.scatter(*P(bx, by, t), s=30, marker="s", c=SCARLET, depthshade=False, zorder=8)
        else:
            B = co[b]
            xs, ys, zs = zip(P(*A), P(*B))
            mid = P(0.7 * A[0] + 0.3 * B[0], 0.7 * A[1] + 0.3 * B[1], (A[2] + B[2]) / 2)
            if A[:2] == B[:2]:
                lab = "M"
            elif A[2] == B[2]:
                typ = "Z" if any(CHECKS[n][1] == tuple(A[:2]) and CHECKS[n][0] == "Z" for n in CHECKS) else "X"
                lab = "X" if typ == "Z" else "Z"
            else:
                lab = "hook"
        kinds.append(lab)
        ax.plot(xs, ys, zs, color=SCARLET, lw=2.6, zorder=9)
        ax.text(mid[0] + 0.2, mid[1], mid[2] + 0.5, lab, color=SCARLET, fontsize=17, fontweight="bold", zorder=11)
    for k in defects:
        ax.scatter(*P(*co[k]), s=170, c=SCARLET, edgecolors="white", linewidths=1.2, depthshade=False, zorder=10)
    print("edge kinds", kinds)

    ax.quiver(7.6, 6.6, 0, 0, 0, max(T) * zsc, color=INK, arrow_length_ratio=0.06, lw=1.6)
    ax.text(7.6, 6.6, max(T) * zsc + 1.0, "time", fontsize=15, ha="center")
    ax.set_xlim(-1.0, 7.5)
    ax.set_ylim(-1.0, 7.5)
    ax.set_zlim(0, max(T) * zsc)
    ax.set_box_aspect((1, 1, 1.35))
    ax.view_init(elev=22, azim=-60)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "spacetime_mwpm.png")


# ---------------------------------------------------------------- 6. hyperedge decomposition
def fig_hyperedge():
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    fired = {"Z2", "Z1", "X1", "X2"}
    for n in CHECKS:
        plaquette(ax, n, fired=n in fired, faded=n not in fired, fs=11)
    data_qubits(ax, err="d4", err_label="Y", fs=10)
    # the two graph edges MWPM keeps
    for (a, b), col in [(("Z2", "Z1"), SCARLET), (("X1", "X2"), SCARLET)]:
        (x1, y1), (x2, y2) = CHECKS[a][1], CHECKS[b][1]
        ax.plot([x1, x2], [y1, y2], color=col, lw=2.2, ls=(0, (4, 3)), zorder=4.5)
    lattice_axes(ax, pad=1.1)
    save(fig, "hyperedge.png")


# ---------------------------------------------------------------- 7. preprocessing
def fig_preprocess():
    # the real shot from the data notes (section 11), relabelled to Carson's names
    stim_order = (["Z0", "Z2", "Z1", "Z3"] + ["X0", "Z2", "X2", "Z3", "Z0", "X1", "Z1", "X3"] * 2
                  + ["Z0", "Z2", "Z1", "Z3"])
    rounds = [0] * 4 + [1] * 8 + [2] * 8 + [3] * 4
    flat = np.zeros(24, int)
    flat[[2, 8, 9, 18]] = 1
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    w = 0.42
    y0 = 4.0
    rcol = ["#E9ECED", "#DADFE0", "#E9ECED", "#DADFE0"]
    for k in range(24):
        x = 0.3 + k * w
        ax.add_patch(Rectangle((x, y0), w * 0.92, w * 0.92, fc=SCARLET if flat[k] else "white", ec=MID, lw=1))
        ax.text(x + w * 0.46, y0 + w * 0.46, str(flat[k]), ha="center", va="center", fontsize=10,
                color="white" if flat[k] else "#9AA2A4", fontweight="bold")
        ax.text(x + w * 0.46, y0 + w + 0.05, stim_order[k], ha="center", va="bottom", fontsize=9, color=GRAY)
    for rr, (a, b) in enumerate([(0, 4), (4, 12), (12, 20), (20, 24)]):
        xa, xb = 0.3 + a * w, 0.3 + b * w - 0.04
        ax.plot([xa, xb], [y0 - 0.12, y0 - 0.12], color=INK, lw=1.5)
        ax.text((xa + xb) / 2, y0 - 0.22, ["round 1", "round 2", "round 3", "final"][rr], ha="center", va="top",
                fontsize=10, color=INK)
    ax.text(0.3, y0 + 0.85, "Stim output: one flat row per shot  [N, 24]", fontsize=14, fontweight="bold",
            ha="left")

    # grid
    gx, gy, g = 1.2, 0.35, 0.5
    grid = np.full((4, 8), -1)
    for k in range(24):
        grid[rounds[k], READING_ORDER.index(stim_order[k])] = flat[k]
    for i in range(4):
        for j in range(8):
            x, y = gx + j * g, gy + (3 - i) * g
            v = grid[i, j]
            if v < 0:
                ax.add_patch(Rectangle((x, y), g * 0.92, g * 0.92, fc="white", ec=MID, lw=1, hatch="////"))
            else:
                ax.add_patch(Rectangle((x, y), g * 0.92, g * 0.92, fc=SCARLET if v else "white", ec=MID, lw=1))
                ax.text(x + g * 0.46, y + g * 0.46, str(v), ha="center", va="center", fontsize=11,
                        color="white" if v else "#9AA2A4", fontweight="bold")
        ax.text(gx - 0.1, gy + (3 - i) * g + g * 0.46, ["r1", "r2", "r3", "F"][i], ha="right", va="center",
                fontsize=11)
    for j, n in enumerate(READING_ORDER):
        ax.text(gx + j * g + g * 0.46, gy + 4 * g + 0.05, n, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
    ax.text(gx, gy - 0.12, "events grid  [rounds, checks]", fontsize=12, va="top", color=INK)

    # arrows from flat cells with events to grid
    for k in [2, 8, 9, 18]:
        i, j = rounds[k], READING_ORDER.index(stim_order[k])
        a = (0.3 + k * w + w * 0.46, y0 - 0.02)
        b = (gx + j * g + g * 0.46, gy + (3 - i) * g + g * 0.95)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=SCARLET, lw=1.0, alpha=0.7,
                                     connectionstyle="arc3,rad=0.0"))

    # channel stack
    sx, sy = 6.6, 0.35
    names = ["events", "mask (slot exists)", "type (X / Z)", "final-row flag", "cumulative XOR"]
    for c_ in range(5)[::-1]:
        ox, oy = sx + c_ * 0.3, sy + c_ * 0.4
        fc = ["#FFFFFF", "#F7F7F7", "#EFEFEF", "#E7E7E7", "#DFDFDF"][c_]
        ax.add_patch(Rectangle((ox, oy), 8 * 0.22, 4 * 0.22, fc=fc, ec=GRAY, lw=1))
        for i in range(4):
            for j in range(8):
                if c_ == 0 and grid[i, j] == 1:
                    ax.add_patch(Rectangle((ox + j * 0.22, oy + (3 - i) * 0.22), 0.22, 0.22, fc=SCARLET, ec="none"))
        lx = sx + 4 * 0.3 + 8 * 0.22 + 0.25
        ax.plot([ox + 8 * 0.22, lx - 0.05], [oy + 4 * 0.22 - 0.12] * 2, color=MID, lw=0.8)
        ax.text(lx, oy + 4 * 0.22 - 0.12, names[c_], fontsize=12, va="center", color=INK)
    ax.text(sx, sy - 0.12, "model input  [N, 4, 8, 5]", fontsize=12, va="top", color=INK)
    ax.add_patch(FancyArrowPatch((gx + 8 * g + 0.15, gy + 1.0), (sx - 0.15, sy + 0.5), arrowstyle="-|>",
                                 mutation_scale=16, color=INK, lw=1.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.4, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "preprocess.png")


# ---------------------------------------------------------------- 8. architecture glyphs
def glyph(name, draw):
    fig, ax = plt.subplots(figsize=(1.9, 1.2))
    draw(ax)
    ax.set_xlim(0, 1.9)
    ax.set_ylim(0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, f"glyph_{name}.png")


def fig_glyphs():
    def mlp(ax):
        layers = [np.linspace(0.15, 1.05, 6), np.linspace(0.3, 0.9, 4), [0.6]]
        xs = [0.3, 0.95, 1.6]
        for a in range(2):
            for y1 in layers[a]:
                for y2 in layers[a + 1]:
                    ax.plot([xs[a], xs[a + 1]], [y1, y2], color=MID, lw=0.7, zorder=1)
        for x, ys in zip(xs, layers):
            for y in ys:
                ax.add_patch(Circle((x, y), 0.07, fc=SCARLET if x == 0.3 and y in layers[0][[1, 4]] else "white",
                                    ec=INK, lw=1, zorder=2))

    def cnn(ax):
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((0.35 + j * 0.24, 0.1 + i * 0.24), 0.24, 0.24, fc="white", ec=MID, lw=0.8))
        ax.add_patch(Rectangle((0.35 + 1 * 0.24, 0.1 + 1 * 0.24), 0.48, 0.48, fc=SCARLET, alpha=0.25, ec=SCARLET,
                               lw=1.6))
        ax.add_patch(Rectangle((0.35 + 1 * 0.24, 0.1 + 1 * 0.24), 0.24, 0.24, fc=SCARLET, ec="none"))
        ax.annotate("", xy=(1.62, 0.6), xytext=(1.38, 0.6), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))

    def rnn(ax):
        for k in range(4):
            x = 0.12 + k * 0.45
            ax.add_patch(Rectangle((x, 0.45), 0.3, 0.3, fc="white", ec=INK, lw=1.1))
            ax.text(x + 0.15, 0.6, "h", ha="center", va="center", fontsize=9)
            ax.text(x + 0.15, 0.2, f"r{k + 1}", ha="center", va="center", fontsize=8, color=GRAY)
            ax.annotate("", xy=(x + 0.15, 0.45), xytext=(x + 0.15, 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=SCARLET, lw=1))
            if k < 3:
                ax.annotate("", xy=(x + 0.45, 0.6), xytext=(x + 0.3, 0.6),
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))

    def transformer(ax):
        xs = np.linspace(0.2, 1.7, 6)
        for x in xs:
            ax.add_patch(Rectangle((x - 0.08, 0.15), 0.16, 0.22, fc="white", ec=INK, lw=1))
        for a in range(6):
            for b in range(a + 1, 6):
                mid = (xs[a] + xs[b]) / 2
                h = 0.15 + 0.13 * (b - a)
                t = np.linspace(0, np.pi, 30)
                ax.plot(mid + (xs[b] - xs[a]) / 2 * np.cos(t), 0.4 + h * np.sin(t),
                        color=SCARLET if a == 1 else MID, lw=1.1 if a == 1 else 0.6)

    def gnn(ax):
        pts = np.array([[0.3, 0.3], [0.7, 0.9], [0.9, 0.35], [1.3, 0.8], [1.6, 0.3], [1.15, 0.15]])
        E = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (2, 5), (4, 5), (3, 5)]
        for a, b in E:
            ax.plot(*zip(pts[a], pts[b]), color=MID, lw=1.2, zorder=1)
        for k, (x, y) in enumerate(pts):
            ax.add_patch(Circle((x, y), 0.09, fc=SCARLET if k in (2, 3) else "white", ec=INK, lw=1, zorder=2))

    for n, f in [("mlp", mlp), ("cnn", cnn), ("rnn", rnn), ("transformer", transformer), ("gnn", gnn)]:
        glyph(n, f)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    fns = dict(lattice=fig_lattice, errors=fig_error_types, meas=fig_meas_error, hcs=fig_hcs, eq=fig_equations,
               spacetime=fig_spacetime, hyper=fig_hyperedge, pre=fig_preprocess, glyphs=fig_glyphs)
    for k, f in fns.items():
        if "all" in which or k in which:
            f()
    print("ok")
