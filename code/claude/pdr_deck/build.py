"""Build the Team 22 PDR deck on top of the Problem Statement deck (OSU template)."""
import copy
import os

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
SRC = "/Users/carson/code/school/research-assistant/PDR Presentation Materials/Team22_Problem_Statement.pptx"
OUT = os.path.join(HERE, "Team22_PDR.pptx")

SCARLET = RGBColor(0xBB, 0x00, 0x00)
INK = RGBColor(0x21, 0x21, 0x21)
GRAY = RGBColor(0x63, 0x6D, 0x6E)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
MID = RGBColor(0xC8, 0xCC, 0xCD)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
XCOL = RGBColor(0xF4, 0xC2, 0x6B)
ZCOL = RGBColor(0x8D, 0xB3, 0xD6)

prs = Presentation(SRC)
orig = list(prs.slides)
S_TITLE, S_TEAM, S_BG, S_PDEF, S_SCOPE, S_CONS, S_STAKE, S_NEEDS, S_REQ, S_REFS = orig
LAYOUT = {l.name: l for m in prs.slide_masters for l in m.slide_layouts}
TEXT_LAYOUT = S_TEAM.slide_layout
SLDNUM = next(sh for sh in S_TEAM.shapes if sh.is_placeholder and sh.placeholder_format.idx == 13)._element

# template icons, reused as pictures
ICON = {}
for sl, nm, key in [(S_PDEF, "Icon 60", "check"), (S_PDEF, "Icon 62", "cross"), (S_BG, "Icon 47", "clock"),
                    (S_BG, "Icon 50", "target"), (S_PDEF, "Icon 65", "goal"), (S_BG, "Icon 44", "chip"),
                    (S_BG, "Icon 21", "noise")]:
    sh = next(s for s in sl.shapes if s.name == nm)
    p = os.path.join(HERE, f"icon_{key}.png")
    open(p, "wb").write(sh.image.blob)
    ICON[key] = p
for sl, nm, key in [(S_SCOPE, "Icon 68", "data"), (S_SCOPE, "Icon 72", "graph"), (S_SCOPE, "Icon 76", "brain"),
                    (S_SCOPE, "Icon 80", "scale"), (S_SCOPE, "Icon 87", "fpga")]:
    sh = next((s for s in sl.shapes if s.name == nm), None)
    if sh is not None:
        p = os.path.join(HERE, f"icon_{key}.png")
        open(p, "wb").write(sh.image.blob)
        ICON[key] = p

TITLE_TOP, TITLE_H = Inches(0.86), Inches(0.9)


# ------------------------------------------------------------------ helpers
def nb(t):
    for a in ("d = ", "p = ", "r = "):
        t = t.replace(a, a[0] + "\u00a0=\u00a0")
    return t


def nostyle(conn):
    st = conn._element.find(qn("p:style"))
    if st is not None:
        conn._element.remove(st)
    return conn


def style_title(slide, text):
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.48), TITLE_TOP, Inches(9.07), TITLE_H
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    tf.text = nb(text)
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(24)
            r.font.name = "Arial"
            r.font.color.rgb = INK
    return t


def new_slide(title, notes=None, layout=None):
    s = prs.slides.add_slide(layout or TEXT_LAYOUT)
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx not in (0,):
            ph._element.getparent().remove(ph._element)
    s.shapes._spTree.append(copy.deepcopy(SLDNUM))
    if title is not None:
        style_title(s, title)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         margin=0, italic=False, spacing=None, name=None):
    """runs: str, or list of paragraphs; each paragraph a str or list of (text, opts) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        segs = para if isinstance(para, list) else [(para, {})]
        for item in segs:
            seg, o = (item, {}) if isinstance(item, str) else item
            r = p.add_run()
            r.text = nb(seg)
            f = r.font
            if o.get("baseline"):
                r._r.get_or_add_rPr().set("baseline", str(o["baseline"]))
            f.name = "Arial"
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, lw=1.0,
        dash=None, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = dash
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def icon_circle(slide, x, y, d, icon=None, label=None, fill=SCARLET, fs=14):
    box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    if icon:
        k = d * 0.52
        slide.shapes.add_picture(ICON[icon], Inches(x + (d - k) / 2), Inches(y + (d - k) / 2), Inches(k), Inches(k))
    if label:
        text(slide, x, y, d, d, label, size=fs, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)


def pic(slide, name, x, y, w=None, h=None):
    return slide.shapes.add_picture(os.path.join(FIG, name), Inches(x), Inches(y),
                                    Inches(w) if w else None, Inches(h) if h else None)


def arrow(slide, x1, y1, x2, y2, color=INK, w=1.5):
    c = nostyle(slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)))
    c.line.color.rgb = color
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def caption(slide, x, y, w, s, h=0.3, align=PP_ALIGN.LEFT):
    return text(slide, x, y, w, h, s, size=11, color=GRAY, align=align)


def table(slide, x, y, w, rows, col_w, size=12, header_size=12, row_h=0.32, bold_first_col=False):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for a in ("bandRow", "firstRow"):
        tblPr.set(a, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # No Style, Table Grid
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = SCARLET if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
            tf = cell.text_frame
            tf.word_wrap = True
            opts = {}
            if isinstance(val, tuple):
                val, opts = val
            tf.text = ""
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            r.font.name = "Arial"
            r.font.size = Pt(header_size if i == 0 else opts.get("size", size))
            r.font.bold = i == 0 or opts.get("bold", False) or (bold_first_col and j == 0)
            r.font.color.rgb = WHITE if i == 0 else opts.get("color", INK)
            tf.paragraphs[0].alignment = opts.get("align", PP_ALIGN.LEFT)
            # thin borders
            tcPr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag), w="6350")
                sf = etree.SubElement(ln, qn("a:solidFill"))
                etree.SubElement(sf, qn("a:srgbClr"), val="D9D9D9")
    return gt


def retitle(slide, text_):
    style_title(slide, text_)


def set_notes(slide, notes):
    slide.notes_slide.notes_text_frame.text = notes


def chart_fonts(chart, size=11):
    chart.font.name = "Arial"
    chart.font.size = Pt(size)
    chart.font.color.rgb = GRAY


# ================================================================== slides
slides_order = []

# ---- 1 Title (reuse)
st = S_TITLE.shapes[1].text_frame
last = st.paragraphs[-1]
for r in last.runs[1:]:
    r.text = ""
last.runs[0].text = "Preliminary Design Review  ·  10/29/2026  ·  ECE 3906: Capstone Design 1"
set_notes(S_TITLE, "Suggested presenter: Carson (~15 s).\n"
          "We are Team 22. Our sponsor, Dr. Youngseok Kim, asked us to build machine-learning decoders for quantum "
          "error correction. In the next ten minutes: why decoding is hard, where the standard decoder falls short, "
          "and how we plan to close that gap.")
slides_order.append(S_TITLE)

# ---- 2 Team (reuse)
set_notes(S_TEAM, "Each member introduces themselves and their role (~10 s each).")
slides_order.append(S_TEAM)

# ---- 3 Classical vs quantum
s = new_slide("Quantum bits can't be protected the classical way: we can't copy them, look at them, "
              "or assume one error type",
              "Suggested presenter: Carson (~45 s).\n"
              "Classical memory protects a bit by copying it and taking a majority vote: if one copy flips, the "
              "other two outvote it. Qubits break all three steps. A qubit holds a superposition, alpha|0> + beta|1>, "
              "and the no-cloning theorem forbids copying an unknown state [3]. Reading a qubit collapses it, so we "
              "can't look at the data to check it. And there are two independent kinds of error: a bit flip (X) and a "
              "phase flip (Z), plus both at once (Y).\n"
              "The fix is indirect: measure parities of groups of qubits. A parity tells us whether something "
              "changed without revealing the stored value. Those parity bits are the syndrome, and turning the "
              "syndrome into a correction is the decoder's job.")
box(s, 0.5, 1.95, 4.3, 3.0)
text(s, 0.75, 2.08, 3.8, 0.35, "Classical bit: copy and vote", size=17, bold=True)
# bit boxes: 0 -> 000 -> 010 -> 0
bx = [("0", False), None, ("0", False), ("1", True), ("0", False), None, ("0", False)]
labels = ["store", "", "noise flips one", "", "", "", "majority"]
xx = 0.75
for k, b in enumerate(bx):
    if b is None:
        text(s, xx, 2.75, 0.3, 0.45, "→", size=20, color=GRAY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        xx += 0.38
        continue
    v, hl = b
    box(s, xx, 2.75, 0.45, 0.45, fill=SCARLET if hl else WHITE, line=MID, shape=MSO_SHAPE.RECTANGLE)
    text(s, xx, 2.75, 0.45, 0.45, v, size=18, bold=True, color=WHITE if hl else INK, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    xx += 0.5 if k in (2, 3) else 0.55
text(s, 0.75, 3.3, 3.8, 0.3, [[("0  →  000  →  010  →  0", {"color": GRAY})]], size=13)
text(s, 0.75, 3.75, 3.85, 1.0, [[("✓ ", {"color": SCARLET, "bold": True}), ("Works because we can copy, read, and "
                                                                          "only bit flips occur", {})]], size=14)
box(s, 5.2, 1.95, 4.3, 3.0)
text(s, 5.45, 2.08, 3.8, 0.35, [[("Qubit: ", {}), ("α|0⟩ + β|1⟩", {"color": SCARLET})]], size=17, bold=True)
rows3 = [("Can't copy", "no-cloning theorem"), ("Can't look", "measuring collapses the state"),
         ("Two error types", "bit flip X and phase flip Z (Y = both)")]
for k, (a, b) in enumerate(rows3):
    yy = 2.62 + k * 0.72
    s.shapes.add_picture(ICON["cross"], Inches(5.45), Inches(yy + 0.04), Inches(0.26), Inches(0.26))
    text(s, 5.85, yy, 3.5, 0.65, [[(a, {"bold": True})], [(b, {"color": GRAY, "size": 13})]], size=15)
b = box(s, 0.5, 5.12, 9.0, 0.55, fill=SCARLET)
text(s, 0.75, 5.12, 8.6, 0.55, "Instead: measure parities of qubit groups, then decode the syndrome classically", size=16, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
slides_order.append(s)

# ---- 4 Surface code lattice
s = new_slide("The d = 3 surface code stores one logical qubit in 9 data qubits watched by 8 parity checks",
              "Suggested presenter: Carson (~45 s).\n"
              "This is the code we work with: the distance-3 rotated surface code [3]. Nine data qubits (white "
              "circles, d0 to d8) hold the information. Each colored tile is a check qubit that measures the parity "
              "of its 2 or 4 neighbors. Orange X-checks measure X parity, so they catch Z (phase) errors; blue "
              "Z-checks measure Z parity and catch X (bit) errors.\n"
              "All 8 checks are measured every round, about once per microsecond on Google's hardware [7], for as "
              "long as the qubit is stored. Distance 3 means at least two faults are needed to cause a logical error. "
              "Our sponsor scoped the project to d = 3, with 20 to 100+ rounds per experiment.")
pic(s, "lattice.png", 0.6, 1.85, h=3.85)
leg = [(WHITE, "Data qubit (d0–d8)", "holds the encoded information", MSO_SHAPE.OVAL),
       (XCOL, "X-check (X0–X3)", "measures X parity → detects Z errors", MSO_SHAPE.RECTANGLE),
       (ZCOL, "Z-check (Z0–Z3)", "measures Z parity → detects X errors", MSO_SHAPE.RECTANGLE)]
for k, (col, a, b_, shp) in enumerate(leg):
    yy = 2.05 + k * 0.82
    box(s, 5.0, yy + 0.05, 0.42, 0.42, fill=col, line=INK if shp == MSO_SHAPE.OVAL else None, shape=shp)
    text(s, 5.6, yy, 3.9, 0.7, [[(a, {"bold": True})], [(b_, {"color": GRAY, "size": 13})]], size=16)
box(s, 5.0, 4.6, 4.5, 0.95)
icon_circle(s, 5.18, 4.8, 0.55, icon="clock")
text(s, 5.9, 4.6, 3.5, 0.95, [[("Every check, every round", {"bold": True})],
                               [("~1 µs per round [7]; 20–100+ rounds", {"color": GRAY, "size": 13})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE)
slides_order.append(s)

# ---- 5 Error types
s = new_slide("Each error type lights up its own pattern of checks, in space and in time",
              "Suggested presenter: Carson (~45 s).\n"
              "Here the same lattice is hit by one error on d4. Red dots are checks whose parity flipped.\n"
              "An X error on d4 flips the two Z-checks that touch it, Z2 and Z1. A Z error flips the two X-checks, X1 "
              "and X2. A Y error is X and Z at once, so all four fire. Notice the Y pattern is the union of the "
              "other two; that coupling between X and Z will matter later.\n"
              "Checks themselves can be misread. On the right, Z1 reads 1 in round 3 only. We compare each check "
              "with its previous round, so a misread creates two detection events stacked in time. That's why "
              "decoding works on space-time patterns, not single snapshots.")
pic(s, "error_types.png", 0.45, 1.95, w=6.3)
for k, (a, b_) in enumerate([("X on d4", "Z-checks fire"), ("Z on d4", "X-checks fire"), ("Y on d4", "both fire")]):
    text(s, 0.45 + k * 2.1, 4.3, 2.1, 0.6, [[(a, {"bold": True})], [(b_, {"color": GRAY, "size": 13})]],
         size=15, align=PP_ALIGN.CENTER)
box(s, 6.95, 1.95, 2.6, 3.0)
text(s, 7.1, 2.05, 2.3, 0.35, "Misread check", size=15, bold=True, align=PP_ALIGN.CENTER)
pic(s, "meas_error.png", 7.05, 2.5, w=2.4)
text(s, 7.1, 3.85, 2.3, 1.0, "One bad reading → two events, one round apart", size=13, color=GRAY,
     align=PP_ALIGN.CENTER)
text(s, 0.5, 5.15, 9.0, 0.45, [[("Detection event", {"bold": True, "color": SCARLET}),
                                ("  =  a check's result changed since the previous round", {"color": INK})]],
     size=15)
slides_order.append(s)

# ---- 6 Hc = s
s = new_slide("Decoding is a classical inference problem: the hidden error c is seen only through its "
              "syndrome s = Hc",
              "Suggested presenter: Carson (~50 s).\n"
              "Write the error as a bit vector c: one entry per possible fault. Here, X errors on the nine data "
              "qubits. The check matrix H says which checks see which qubit. Multiplying mod 2 gives the syndrome: an "
              "X on d4 picks out column d4 of H, so Z1 and Z2 fire. We observe s; we never observe c.\n"
              "With full circuit noise over 3 rounds, H grows to 24 detectors by 219 independent fault mechanisms, "
              "taken from Stim's detector error model [8], [15].\n"
              "Two points make this a classical problem. The inputs are plain bits, and we don't need c itself, only "
              "whether it flips the logical qubit, which is one bit l = Lc. Many different errors give the same "
              "answer; that's called degeneracy. So decoding is binary classification from a bit pattern, which is "
              "exactly the kind of problem ML is good at.")
pic(s, "hcs.png", 0.5, 1.9, w=6.0)
cards = [("Observed", "syndrome s: plain classical bits"),
         ("Unknown", "error c: never measured"),
         ("Wanted", "one bit, l = Lc: did the logical qubit flip?")]
for k, (a, b_) in enumerate(cards):
    yy = 1.95 + k * 0.95
    box(s, 6.8, yy, 2.7, 0.82)
    text(s, 6.95, yy, 2.45, 0.82, [[(a, {"bold": True, "color": SCARLET})], [(b_, {"size": 13})]], size=15,
         anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.5, 5.0, 9.0, 0.6, [[("Full circuit noise, d = 3, 3 rounds:  ", {"color": GRAY}),
                               ("H is 24 detectors × 219 fault mechanisms", {"bold": True}),
                               ("  [8], [S]", {"color": GRAY})]], size=15)
slides_order.append(s)

# ---- 7 Optimal decoding intractable
s = new_slide("The optimal decoder sums the probability of every consistent error, which is intractable",
              "Suggested presenter: Carson (~50 s).\n"
              "Model each fault mechanism j as an independent coin flip with probability p_j. That's the detector "
              "error model [15]. Hardware and circuit design work to keep noise local and stochastic, so independence "
              "is a good approximation, though not a perfect one.\n"
              "Under that model the probability of an error pattern c is the product on top. The optimal decoder "
              "adds up the probability of every error consistent with the syndrome, split by which logical class it "
              "lands in, and picks the larger class.\n"
              "The catch: at d = 3 with 3 rounds, each class contains 2^194 error patterns, about 10^58. In general, "
              "exact maximum-likelihood decoding is #P-hard [19]. We could compute it exactly only for d = 3, 3 "
              "rounds, by brute force over all 2^25 syndrome outcomes, and we use that as our ceiling.")
text(s, 0.5, 1.95, 5.6, 0.35, "1.  Faults are independent random events", size=16, bold=True)
pic(s, "eq_prob.png", 0.85, 2.38, h=0.55)
text(s, 0.5, 3.15, 5.6, 0.35, "2.  Pick the logical class with more total probability", size=16, bold=True)
pic(s, "eq_opt.png", 0.85, 3.6, h=0.78)
text(s, 0.5, 4.55, 5.6, 1.0, [[("Every term counts: many different errors give the same syndrome and the same "
                                "logical flip (degeneracy).", {"color": GRAY})]], size=14)
box(s, 6.4, 1.95, 3.1, 3.55, fill=SCARLET)
text(s, 6.55, 2.1, 2.8, 0.4, "Terms per class", size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
text(s, 6.55, 2.5, 2.8, 1.0, [[("2", {}), ("194", {"size": 28, "baseline": 30000})]], size=54, bold=True, color=WHITE,
     align=PP_ALIGN.CENTER)
text(s, 6.55, 3.45, 2.8, 0.4, "≈ 10⁵⁸   (d = 3, 3 rounds)", size=15, color=WHITE, align=PP_ALIGN.CENTER)
text(s, 6.6, 4.05, 2.7, 1.3, "Exact decoding is #P-hard in general [19]; we computed it exactly only at "
     "d = 3, 3 rounds, as a ceiling", size=13, color=WHITE, align=PP_ALIGN.CENTER)
slides_order.append(s)

# ---- 8 MWPM
s = new_slide("Two approximations turn decoding into a fast graph problem: minimum-weight perfect matching (MWPM)",
              "Suggested presenter: Carson (~60 s).\n"
              "Two facts about the surface code make a shortcut possible.\n"
              "First, the error rate is low, so one most-likely error dominates the sum. Replace the sum with a max. "
              "Taking -log of the probability turns the max into a minimum total weight, where each fault costs "
              "w = ln((1-p)/p): rare faults are expensive.\n"
              "Second, each check connects to only two kinds of data qubit, so most single faults flip at most two "
              "detectors. Each fault becomes an edge between two detectors, and the decoder pairs up the lit "
              "detectors along the cheapest paths. That is minimum-weight perfect matching, solvable in polynomial "
              "time by Edmonds' blossom algorithm [20]. PyMatching does it at about a million errors per core-second "
              "[4], [10].\n"
              "On the right is a real Stim shot at d = 3 over 4 rounds, decoded by PyMatching. Red dots are detection "
              "events. X and Z edges are data-qubit errors within a round; the vertical M edge is a measurement error "
              "linking two rounds; the X edge at the bottom runs to the boundary. MWPM's strengths: fast, no "
              "training, and well understood. Its weakness is the next slide.")
text(s, 0.5, 1.95, 5.1, 0.6, [[("Low error rate ", {"bold": True, "color": SCARLET}),
                                ("→ one most-likely error dominates: max instead of sum", {})]], size=15)
text(s, 0.5, 2.6, 5.1, 0.6, [[("Graph-like faults ", {"bold": True, "color": SCARLET}),
                               ("→ each flips ≤ 2 detectors, so each fault is an edge", {})]], size=15)
box(s, 0.5, 3.3, 5.1, 1.05)
pic(s, "eq_mwpm.png", 0.65, 3.42, h=0.78)
text(s, 0.5, 4.5, 5.1, 1.1, [[("Pair up lit detectors by cheapest total path: ", {}),
                               ("polynomial time", {"bold": True}),
                               (" via blossom algorithm [20]; ~10⁶ errors per core-second [10]", {})]], size=14,
     color=INK)
pic(s, "spacetime_mwpm.png", 5.75, 1.75, h=3.75)
caption(s, 5.75, 5.45, 3.8, "Real d = 3 Stim shot, decoded by PyMatching [4]. X/Z: data errors, M: misread", h=0.35)
slides_order.append(s)

# ---- 9 Gap
s = new_slide("MWPM makes 19–26% more logical errors than optimal at d = 3; lost correlations cause ~95% of the gap",
              "Suggested presenter: Carson (~60 s).\n"
              "Graph-like isn't the whole story. At d = 3, 113 of the 219 fault mechanisms flip three or four "
              "detectors; they're hyperedges, not edges. The clearest case is a Y error: it lights two Z-checks and "
              "two X-checks at once. MWPM splits it into two edges and decodes them as if independent, so it throws "
              "away the fact that they happen together.\n"
              "We measured how much that costs, exactly and without sampling, at d = 3 with 3 rounds. The chart shows "
              "each decoder's logical error rate divided by the optimal one. MWPM is 1.19 to 1.26 times optimal. If we "
              "keep MWPM's max but give it the true correlated model, it's within 2% of optimal. If we instead fix the "
              "max-vs-sum issue (degeneracy) but keep the graph model, it barely moves. So 92 to 96% of the gap is "
              "correlation [S].\n"
              "That's the opening for ML: a model trained directly on syndromes can learn these correlations, and "
              "that's where we expect it to beat MWPM.")
pic(s, "hyperedge.png", 0.35, 1.85, h=2.95)
text(s, 0.45, 4.8, 3.2, 0.85, [[("Y on d4 ", {"bold": True}), ("= X + Z together. MWPM splits it into 2 "
                                                               "independent edges.", {"color": GRAY})]], size=13)
cd = CategoryChartData()
cd.categories = ["p = 0.1%", "0.2%", "0.3%", "0.5%"]
cd.add_series("MWPM", (1.262, 1.241, 1.222, 1.190))
cd.add_series("Fix degeneracy (sum)", (1.253, 1.231, 1.211, 1.178))
cd.add_series("Fix correlations (hyperedges)", (1.013, 1.014, 1.016, 1.019))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(3.75), Inches(1.85), Inches(5.8), Inches(3.45), cd)
ch = gf.chart
chart_fonts(ch, 11)
ch.has_title = True
ch.chart_title.text_frame.text = "Logical error rate ÷ optimal   (d = 3, 3 rounds, exact)"
tr = ch.chart_title.text_frame.paragraphs[0].runs[0]
tr.font.size, tr.font.bold, tr.font.color.rgb, tr.font.name = Pt(12), True, INK, "Arial"
va = ch.value_axis
va.minimum_scale, va.maximum_scale, va.major_unit = 1.0, 1.3, 0.1
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE6, 0xE6)
va.tick_labels.number_format = '0.0"×"'
va.tick_labels.number_format_is_linked = False
va.format.line.fill.background()
ch.category_axis.format.line.color.rgb = MID
plot = ch.plots[0]
plot.gap_width = 60
plot.overlap = -10
for ser, col in zip(plot.series, [SCARLET, RGBColor(0xE8, 0x8A, 0x8A), GRAY]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format = '0.00'
dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(9)
dl.font.color.rgb = INK
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11)
b = box(s, 3.75, 5.3, 5.8, 0.4, fill=SCARLET)
text(s, 3.9, 5.3, 5.6, 0.4, "Target: an ML decoder that learns these correlations", size=14,
     bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
slides_order.append(s)

# ---- 10 Project overview (replaces Problem Definition + Scope)
s = new_slide("Our objective: two ML decoders that beat MWPM on the same data, within size and latency limits",
              "Suggested presenter: Daniel (~45 s).\n"
              "That gap is our project. ML decoders like Google DeepMind's AlphaQubit already beat matching on real "
              "hardware data, cutting logical errors from 4.0% to 2.9% on Sycamore, but they are far too slow and "
              "large for real-time use [6].\n"
              "Our sponsor, Dr. Kim, asked for two ML decoders that beat MWPM: a feed-forward or convolutional one, "
              "and a recurrent or transformer one. The work runs in four steps: generate labeled syndromes with "
              "Stim [8], build the MWPM baseline, train the two decoders, and compare on identical test shots. We "
              "train on simulation and test on public Google hardware data. Running a decoder on an FPGA at about 1 "
              "microsecond per round is a stretch goal.")
steps = [("data", "1. Generate data", "Labeled syndromes from Stim [8]"),
         ("graph", "2. MWPM baseline", "Reproduced ✓"),
         ("brain", "3. ML decoders", "FF / CNN, then RNN / Transformer"),
         ("scale", "4. Compare", "Same shots, ε per round")]
for k, (ic, a, b_) in enumerate(steps):
    x = 0.5 + k * 2.31
    box(s, x, 1.95, 2.07, 2.0)
    icon_circle(s, x + 0.7, 2.08, 0.66, icon=ic if ic in ICON else None)
    text(s, x + 0.05, 2.85, 1.97, 0.35, a, size=15, bold=True, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 3.2, 1.87, 0.7, b_, size=13, color=GRAY, align=PP_ALIGN.CENTER)
    if k < 3:
        text(s, x + 2.07, 2.25, 0.24, 0.4, "›", size=20, color=GRAY, align=PP_ALIGN.CENTER)
box(s, 0.5, 4.15, 4.4, 0.62)
text(s, 0.7, 4.15, 4.1, 0.62, [[("Train: ", {"bold": True}), ("Stim simulations", {})]], size=14,
     anchor=MSO_ANCHOR.MIDDLE)
box(s, 5.1, 4.15, 4.4, 0.62)
text(s, 5.3, 4.15, 4.1, 0.62, [[("Test: ", {"bold": True}), ("Google Sycamore / Willow data [14], [7]", {})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.5, 4.95, 9.0, 0.62, fill=WHITE, line=GRAY, dash=4)
icon_circle(s, 0.68, 5.03, 0.46, icon="chip")
text(s, 1.2, 4.95, 8.2, 0.62, [[("Stretch goal: ", {"bold": True}),
                                 ("compress and run on an FPGA at ~1 µs per round [6], [7]", {"color": GRAY})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE)
slides_order.append(s)

# ---- 11-14 reused requirement slides
retitle(S_STAKE, "Stakeholders range from our sponsor to QEC researchers")
set_notes(S_STAKE, "Suggested presenter: Nate (~30 s).\n" + S_STAKE.notes_slide.notes_text_frame.text)
retitle(S_NEEDS, "Needs Assessment: beating MWPM ranks first")
set_notes(S_NEEDS, "Suggested presenter: Nate (~30 s).\n" + S_NEEDS.notes_slide.notes_text_frame.text)
retitle(S_REQ, "Requirements Definition")
set_notes(S_REQ, "Suggested presenter: Nate (~30 s).\n" + S_REQ.notes_slide.notes_text_frame.text +
          "\nUpdate since the problem statement: decoders will be compared on logical error per round (ε), fitted "
          "over 3–100 rounds, because per-shot error tends to 50% on long runs for every decoder. The exact d = 3 "
          "ceiling says at most a 16–21% reduction in logical errors is possible there.")
retitle(S_CONS, "No quantum hardware is our biggest constraint")
set_notes(S_CONS, "Suggested presenter: Nate (~25 s).\n" + S_CONS.notes_slide.notes_text_frame.text)
slides_order += [s for s in (S_STAKE, S_NEEDS, S_REQ, S_CONS)]

# ---- 15 System design block diagram
s = new_slide("Design: one Stim pipeline feeds MWPM and both ML decoders the same shots, so the comparison is fair",
              "Suggested presenter: Ali (~45 s).\n"
              "This is the system. Stim builds a noisy surface-code circuit from a config: distance, rounds, noise "
              "level, basis. Each sample, or shot, gives a row of detection events and one label bit, whether the "
              "logical qubit flipped. Google's public hardware data comes in the same format, so it slots in at the "
              "same point.\n"
              "From there, two paths. The ML path preprocesses each shot into a space-time grid and feeds the "
              "decoders. The baseline path builds a matching graph from the same circuit's detector error model and "
              "runs PyMatching. Both predict the same bit on the same test shots, and we compare logical error per "
              "round with confidence intervals, plus how much of the MWPM-to-optimal gap the model closes.")
W, G = 1.6, 0.27
X = [0.5 + k * (W + G) for k in range(5)]
TOP, BOT, MIDY, H = 1.95, 3.85, 2.9, 1.25
nodes = {
    "stim": (X[0], TOP, "Simulation", "d, r, p, basis\nseeded, SI1000 noise", SCARLET),
    "hw": (X[0], BOT, "Hardware data", "Google Sycamore / Willow [14], [7]", GRAY),
    "ds": (X[1], MIDY, "Shots", "detection events\n+ 1 label bit each", INK),
    "pre": (X[2], TOP, "Preprocessing", "space-time grid\n+ 5 channels", INK),
    "dem": (X[2], BOT, "Error graph", "edges from the same circuit's error model", INK),
    "ml": (X[3], TOP, "ML decoders", "#1 FF / CNN\n#2 RNN / Transformer", SCARLET),
    "mw": (X[3], BOT, "MWPM", "PyMatching [4]", GRAY),
    "ev": (X[4], MIDY, "Compare", "ε per round ± CI\n% of gap closed", INK),
}
for k, (x, y, a, b_, col) in nodes.items():
    box(s, x, y, W, H, fill=LIGHT if col == INK else col)
    fc = WHITE if col != INK else INK
    text(s, x + 0.08, y + 0.1, W - 0.16, 0.35, a, size=13, bold=True, color=fc, align=PP_ALIGN.CENTER)
    text(s, x + 0.08, y + 0.47, W - 0.16, 0.75, b_, size=12, color=WHITE if col != INK else GRAY,
         align=PP_ALIGN.CENTER)
cy = lambda y: y + H / 2
arrow(s, X[0] + W, cy(TOP), X[1], cy(MIDY) - 0.25)
arrow(s, X[0] + W, cy(BOT), X[1], cy(MIDY) + 0.25)
arrow(s, X[1] + W, cy(MIDY) - 0.25, X[2], cy(TOP))
arrow(s, X[1] + W, cy(MIDY) + 0.25, X[2] + 0.2, BOT + 0.02)
arrow(s, X[2] + W, cy(TOP), X[3], cy(TOP))
arrow(s, X[2] + W, cy(BOT), X[3], cy(BOT))
arrow(s, X[3] + W, cy(TOP), X[4], cy(MIDY) - 0.25)
arrow(s, X[3] + W, cy(BOT), X[4], cy(MIDY) + 0.25)
text(s, 0.5, 5.25, 9.0, 0.4, [[("Owners: ", {"bold": True}),
                                ("Ali – pipeline   ·   Daniel – preprocessing & masks   ·   Nate – input shapes   ·   Carson – MWPM", {"color": GRAY})]], size=12)
slides_order.append(s)

# ---- 16 Stim
s = new_slide("Stim simulates realistic circuit noise at scale, but not every hardware effect",
              "Suggested presenter: Ali (~45 s).\n"
              "Stim [8] simulates the full measurement circuit with random Pauli faults after every gate, reset, "
              "measurement and idle step, and it's fast: our full 12-experiment sweep took about 40 seconds. Its "
              "output has exactly the hardware format: detection events plus the logical flip label.\n"
              "What it can't model: leakage out of the qubit states, coherent errors, slow drift, and analog readout "
              "values. Those are what separate real data from simulation, so we follow the standard recipe: train "
              "on simulation, then test, and possibly fine-tune, on Google's public hardware data [6], [12].\n"
              "Storage: we store compact bit-packed events and the generating config and seed, not float tensors. A "
              "million d = 5 shots is 15 MB packed versus 2.9 GB as a float32 model input, so batches are built on "
              "the fly. Train, validation and test sets use different seeds, so test shots are never seen in "
              "training.")
box(s, 0.5, 1.95, 4.35, 2.55)
text(s, 0.7, 2.05, 4.0, 0.35, "Models", size=16, bold=True, color=SCARLET)
for k, t_ in enumerate(["Random Pauli faults on every gate, reset, measurement, idle",
                        "Fast: 12-experiment sweep in ~40 s [S]",
                        "Same output format as hardware"]):
    s.shapes.add_picture(ICON["check"], Inches(0.7), Inches(2.52 + k * 0.62), Inches(0.24), Inches(0.24))
    text(s, 1.05, 2.47 + k * 0.62, 3.7, 0.6, t_, size=14)
box(s, 5.15, 1.95, 4.35, 2.55)
text(s, 5.35, 2.05, 4.0, 0.35, "Does not model", size=16, bold=True, color=GRAY)
for k, t_ in enumerate(["Leakage out of the qubit states", "Coherent errors and drift", "Analog (soft) readout"]):
    s.shapes.add_picture(ICON["cross"], Inches(5.35), Inches(2.52 + k * 0.62), Inches(0.24), Inches(0.24))
    text(s, 5.7, 2.47 + k * 0.62, 3.7, 0.6, t_, size=14)
text(s, 5.35, 4.05, 4.0, 0.4, "→ test and fine-tune on hardware data", size=13, bold=True, color=SCARLET)
# storage callout
box(s, 0.5, 4.7, 9.0, 0.95)
text(s, 0.7, 4.7, 1.6, 0.95, "15 MB", size=30, bold=True, color=SCARLET, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.3, 4.7, 2.2, 0.95, "1M shots, d = 5, stored as packed bits", size=13, color=GRAY,
     anchor=MSO_ANCHOR.MIDDLE)
text(s, 4.6, 4.7, 1.5, 0.95, "2.9 GB", size=30, bold=True, color=GRAY, anchor=MSO_ANCHOR.MIDDLE)
text(s, 6.2, 4.7, 3.2, 0.95, "as float32 model input → store bits + seed; batch on the fly",
     size=13, color=GRAY, anchor=MSO_ANCHOR.MIDDLE)
slides_order.append(s)

# ---- 17 Preprocessing
s = new_slide("Preprocessing puts each check's history in one column, so a model can reuse weights across rounds",
              "Suggested presenter: Daniel (~50 s).\n"
              "Stim gives one flat row of bits per shot, ordered however the circuit declared its detectors. For d = 3 "
              "with 3 rounds that's 24 bits. The order is irregular: Z1 is the third bit of round 1 but the seventh "
              "of round 2, and the first and last rounds only have Z-checks.\n"
              "Step one maps every bit, using its space-time coordinates, into a grid of rounds by checks, so a column "
              "always means the same check. Missing slots are masked. Then we add channels: the events, a mask of "
              "which slots exist, the check type X or Z, a flag for the final row (which comes from reading the data "
              "qubits, not from a check), and a running XOR, which is the check's current value. AlphaQubit uses "
              "the same kinds of features [6].\n"
              "Why bother? A recurrent or convolutional model reuses the same weights on every round, so column j has "
              "to mean the same thing in every row. A plain feed-forward network can use the flat row directly.")
pic(s, "preprocess.png", 0.75, 1.9, w=8.5)
slides_order.append(s)

# ---- 18 Model selection
s = new_slide("Each candidate architecture exploits a different structure in the syndrome data",
              "Suggested presenter: Nate (~60 s).\n"
              "The data has two kinds of structure: space, meaning the lattice, and time, meaning the rounds. Each "
              "architecture uses them differently.\n"
              "A feed-forward network (MLP) sees one flat vector: simple and fast, but it must relearn every pattern at "
              "every position, and its input size is fixed. A CNN slides one small kernel over the lattice, so a "
              "local pattern like two neighboring checks firing is learned once; that matches the surface code's "
              "locality. These two are the candidates for decoder #1.\n"
              "A recurrent network (GRU or LSTM) reads one round at a time and carries a memory, so it handles any "
              "number of rounds; that matters because our runs are 20 to 100+ rounds [16]. A transformer lets every "
              "check-round attend to every other, so it can catch long-range correlations; AlphaQubit combines "
              "attention with a recurrent core [6], [12]. These are the candidates for decoder #2.\n"
              "A graph network (GNN) works only on the detectors that fired, which suits sparse syndromes, and "
              "is a strong alternative [16].\n"
              "We will pick one model for a December proof of concept against MWPM.")
models = [("mlp", "MLP", "flat [N, 24]", "Every bit gets its own weight", "#1"),
          ("cnn", "CNN", "lattice image per round", "Local patterns, anywhere on lattice", "#1"),
          ("rnn", "RNN / GRU", "one round at a time", "Memory across any number of rounds", "#2"),
          ("transformer", "Transformer", "token per check-round", "Long-range correlations via attention", "#2"),
          ("gnn", "GNN", "graph of fired detectors", "Sparse; uses only what fired", "alt")]
CW, CG = 1.66, 0.175
for k, (g, a, inp, learn, tag) in enumerate(models):
    x = 0.5 + k * (CW + CG)
    box(s, x, 1.95, CW, 3.3)
    tagcol = SCARLET if tag != "alt" else GRAY
    tg = box(s, x + 0.33, 1.83, CW - 0.66, 0.26, fill=tagcol, radius=0.5)
    text(s, x + 0.33, 1.83, CW - 0.66, 0.26, "Decoder " + tag if tag != "alt" else "Alternative", size=10,
         bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    pic(s, f"glyph_{g}.png", x + 0.2, 2.2, h=0.8)
    text(s, x + 0.08, 3.05, CW - 0.16, 0.35, a, size=15, bold=True, align=PP_ALIGN.CENTER)
    text(s, x + 0.08, 3.42, CW - 0.16, 0.6, inp, size=12, color=GRAY, align=PP_ALIGN.CENTER)
    text(s, x + 0.08, 4.1, CW - 0.16, 1.1, learn, size=13, align=PP_ALIGN.CENTER)
text(s, 0.5, 5.35, 9.0, 0.35, [[("Next: ", {"bold": True, "color": SCARLET}),
                                ("choose one model for a proof of concept against MWPM by 12/4", {})]], size=14)
slides_order.append(s)

# ---- 19 Progress
s = new_slide("Progress: the MWPM baseline is reproduced, and its error per round holds steady from 3 to 100 rounds",
              "Suggested presenter: Carson (~50 s).\n"
              "Done so far: the MWPM baseline is reproduced, and the exact optimal ceiling and gap analysis from the "
              "earlier slide are finished. Version one of the Stim pipeline generates labeled data from a config, and "
              "the preprocessing design is written up.\n"
              "The chart is the baseline we have to beat. Per-shot error grows with the number of rounds for every "
              "decoder, so on long runs it heads toward 50% and a single number says nothing. The fair metric is ε, "
              "the logical error per round. For MWPM it is flat from 3 to 100 rounds: 2.4 × 10^-4 per round at "
              "p = 0.1%. It scales as p squared, as expected at d = 3 [S].\n"
              "In progress: freezing the data format and input shape for each model on 10/30, batching for memory, "
              "and the math of masking.")
cd = CategoryChartData()
cd.categories = ["3", "5", "10", "25", "50", "100"]
eps = {"p = 0.1%": (2.48e-4, 2.60e-4, 2.54e-4, 2.51e-4, 2.34e-4, 2.36e-4),
       "p = 0.2%": (9.59e-4, 9.88e-4, 9.45e-4, 9.45e-4, 9.23e-4, 9.25e-4),
       "p = 0.3%": (2.17e-3, 2.13e-3, 2.02e-3, 2.02e-3, 2.03e-3, 2.02e-3),
       "p = 0.5%": (5.77e-3, 5.63e-3, 5.52e-3, 5.47e-3, 5.37e-3, 5.31e-3)}
for k_, v in eps.items():
    cd.add_series(k_, v)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4), Inches(1.85), Inches(5.0), Inches(3.85), cd)
ch = gf.chart
chart_fonts(ch, 11)
ch.has_title = True
ch.chart_title.text_frame.text = "MWPM error per round ε   (d = 3)"
tr = ch.chart_title.text_frame.paragraphs[0].runs[0]
tr.font.size, tr.font.bold, tr.font.color.rgb, tr.font.name = Pt(12), True, INK, "Arial"
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 1e-4, 1e-2
scaling = va._element.find(qn("c:scaling"))
lb = etree.SubElement(scaling, qn("c:logBase"))
lb.set("val", "10")
scaling.remove(lb)
scaling.insert(0, lb)
va.tick_labels.number_format = '0E+0'
va.tick_labels.number_format_is_linked = False
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE6, 0xE6)
va.format.line.fill.background()
ca = ch.category_axis
ca.has_title = True
ca.axis_title.text_frame.text = "rounds per shot"
ar = ca.axis_title.text_frame.paragraphs[0].runs[0]
ar.font.size, ar.font.bold, ar.font.color.rgb, ar.font.name = Pt(11), False, GRAY, "Arial"
ca.format.line.color.rgb = MID
for ser, col in zip(ch.plots[0].series, ["F0B3B3", "E06666", "BB0000", "6E0000"]):
    ser.format.line.color.rgb = RGBColor.from_string(col)
    ser.format.line.width = Pt(2.25)
    ser.smooth = False
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = RGBColor.from_string(col)
    ser.marker.format.line.color.rgb = RGBColor.from_string(col)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(10)
done = ["MWPM baseline reproduced", "Exact optimal ceiling + gap split", "Stim pipeline v1 (config-driven)",
        "Preprocessing design documented"]
prog = ["Data format & input shapes (10/30)", "Batching for memory", "Masking math", "AlphaQubit deep dive"]
text(s, 5.75, 1.9, 3.8, 0.3, "Done", size=15, bold=True, color=SCARLET)
for k, t_ in enumerate(done):
    s.shapes.add_picture(ICON["check"], Inches(5.75), Inches(2.3 + k * 0.4), Inches(0.22), Inches(0.22))
    text(s, 6.1, 2.25 + k * 0.4, 3.45, 0.35, t_, size=13)
text(s, 5.75, 3.95, 3.8, 0.3, "In progress", size=15, bold=True, color=GRAY)
for k, t_ in enumerate(prog):
    box(s, 5.78, 4.38 + k * 0.33, 0.16, 0.16, fill=None, line=GRAY, shape=MSO_SHAPE.OVAL, lw=1.5)
    text(s, 6.1, 4.3 + k * 0.33, 3.45, 0.33, t_, size=13, color=GRAY)
slides_order.append(s)

# ---- 20 Risks
s = new_slide("Our main risks are statistical and sim-to-real, and each has a planned mitigation",
              "Suggested presenter: Ali (~45 s).\n"
              "Top risk: at d = 3 the most any decoder can gain is 16 to 21% fewer logical errors, so a real win "
              "needs very tight error bars. We size datasets for that early, report confidence intervals, and also "
              "report the fraction of the gap closed.\n"
              "Second: Stim doesn't model leakage or drift, so a model that wins in simulation may not win on "
              "hardware; that's why hardware data is in the test plan.\n"
              "Third: compute. We don't have high-power GPUs, so we start at d = 3 and request Ohio Supercomputer "
              "Center time if needed.\n"
              "Fourth: data-format churn; the format freezes before any prototype code is written.\n"
              "Last: the FPGA stretch goal depends on getting a board; there's a go or no-go decision on 3/12.")
risks = [["", "Risk", "Impact", "Mitigation"],
         [("H", {"bold": True, "color": SCARLET, "align": PP_ALIGN.CENTER}),
          "Small headroom at d = 3 (≤ 16–21%)", "Wins hidden by noise", "Large test sets, 95% CIs, report % of gap closed"],
         [("H", {"bold": True, "color": SCARLET, "align": PP_ALIGN.CENTER}),
          "Simulation ≠ hardware (leakage, drift)", "Model fails on real data", "Test and fine-tune on Google hardware data"],
         [("M", {"bold": True, "color": GRAY, "align": PP_ALIGN.CENTER}),
          "No high-power GPUs", "Slow training", "Start at d = 3; request OSC compute"],
         [("M", {"bold": True, "color": GRAY, "align": PP_ALIGN.CENTER}),
          "Data format keeps changing", "Rework across team", "Freeze format 10/30, before prototype code"],
         [("L", {"bold": True, "color": GRAY, "align": PP_ALIGN.CENTER}),
          "No FPGA board in time", "Stretch goal dropped", "Go / no-go decision 3/12"]]
table(s, 0.5, 2.0, 9.0, risks, [0.45, 2.9, 2.15, 3.5], size=13, header_size=13, row_h=0.56)
caption(s, 0.5, 5.45, 9.0, "Likelihood: H = high, M = medium, L = low")
slides_order.append(s)

# ---- 21 Gantt
s = new_slide("Next: freeze the data format, then test a prototype decoder against MWPM by December",
              "Suggested presenter: Daniel (~45 s).\n"
              "Here's the plan by phase; the red line is today. This semester: freeze the data format on 10/30, "
              "finish the pipeline by 11/13 and the datasets by 11/30, then a prototype decoder against MWPM by 12/4, "
              "shown at our semester-one presentation on 12/8.\n"
              "Next semester: decoder one, feed-forward or CNN, benchmarked by mid-February. Decoder two, recurrent or "
              "transformer, pretrained on simulation and fine-tuned on hardware data by late March. Then compression "
              "and the FPGA stretch goal, with the go or no-go on 3/12, and the design expo on 4/15. Full work "
              "breakdown and dates are in the backup slides and our schedule workbook.")
import datetime as dt

D0, D1 = dt.date(2026, 9, 21), dt.date(2027, 4, 30)
GX0, GX1 = 2.85, 9.5


def gx(d):
    return GX0 + (d - D0).days / (D1 - D0).days * (GX1 - GX0)


months = [dt.date(2026, 10, 1), dt.date(2026, 11, 1), dt.date(2026, 12, 1), dt.date(2027, 1, 1),
          dt.date(2027, 2, 1), dt.date(2027, 3, 1), dt.date(2027, 4, 1)]
for m in months:
    x = gx(m)
    ln = nostyle(s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(2.15), Inches(x), Inches(5.5)))
    ln.line.color.rgb = RGBColor(0xE3, 0xE3, 0xE3)
    ln.line.width = Pt(0.75)
    text(s, x, 1.9, 0.9, 0.25, m.strftime("%b"), size=11, color=GRAY)
rows = [
    ("Research & baselines", dt.date(2026, 9, 24), dt.date(2027, 1, 29), SCARLET, []),
    ("Data pipeline", dt.date(2026, 9, 24), dt.date(2026, 11, 30), SCARLET,
     [(dt.date(2026, 10, 30), ""), (dt.date(2026, 11, 30), "format 10/30, data 11/30")]),
    ("ML prototype", dt.date(2026, 11, 2), dt.date(2026, 12, 15), SCARLET, [(dt.date(2026, 12, 4), ""), (dt.date(2026, 12, 15), "PoC vs MWPM 12/4")]),
    ("Decoder #1  FF / CNN", dt.date(2027, 1, 11), dt.date(2027, 2, 12), INK, [(dt.date(2027, 2, 12), "2/12")]),
    ("Decoder #2  RNN / Transf.", dt.date(2027, 1, 18), dt.date(2027, 3, 26), INK, [(dt.date(2027, 3, 26), "3/26")]),
    ("Benchmark & latency", dt.date(2027, 2, 8), dt.date(2027, 4, 9), INK, []),
    ("Optimize & quantize", dt.date(2027, 3, 22), dt.date(2027, 4, 9), INK, []),
    ("FPGA (stretch)", dt.date(2027, 2, 15), dt.date(2027, 4, 21), None, [(dt.date(2027, 3, 12), "go/no-go 3/12")]),
]
RH = 0.36
for k, (lab, a, b_, col, ms) in enumerate(rows):
    y = 2.22 + k * RH
    text(s, 0.5, y, 2.3, RH, lab, size=12, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    if col is None:
        box(s, gx(a), y + 0.08, gx(b_) - gx(a), RH - 0.16, fill=WHITE, line=GRAY, radius=0.3, dash=4)
    else:
        box(s, gx(a), y + 0.08, gx(b_) - gx(a), RH - 0.16, fill=col if col == SCARLET else GRAY, radius=0.3)
    for d, ml in ms:
        box(s, gx(d) - 0.08, y + 0.1, 0.16, 0.16, fill=INK, shape=MSO_SHAPE.DIAMOND)
        if ml:
            lx = max(gx(d), gx(b_)) + 0.14
            if lx > 8.6:
                text(s, gx(a) - 1.55, y, 1.45, RH, ml, size=10, color=INK, anchor=MSO_ANCHOR.MIDDLE,
                     align=PP_ALIGN.RIGHT)
            else:
                text(s, lx, y, 1.8, RH, ml, size=10, color=INK, anchor=MSO_ANCHOR.MIDDLE)
# course milestones row
y = 2.22 + len(rows) * RH + 0.05
text(s, 0.5, y, 2.3, RH, "Course milestones", size=12, bold=True, color=SCARLET, anchor=MSO_ANCHOR.MIDDLE)
for d, ml in [(dt.date(2026, 10, 29), "PDR (today)"), (dt.date(2026, 12, 8), "Sem 1"), (dt.date(2027, 2, 24), "Mid"),
              (dt.date(2027, 4, 15), "Expo")]:
    box(s, gx(d) - 0.09, y + 0.1, 0.18, 0.18, fill=SCARLET, shape=MSO_SHAPE.DIAMOND)
    text(s, gx(d) + 0.12, y, 1.0, RH, ml, size=10, color=SCARLET, bold=True, anchor=MSO_ANCHOR.MIDDLE)
today = gx(dt.date(2026, 10, 29))
ln = nostyle(s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(today), Inches(2.15), Inches(today), Inches(y + 0.1)))
ln.line.color.rgb = SCARLET
ln.line.width = Pt(1.75)

slides_order.append(s)

# ---- 22 Questions
s = new_slide(None, layout=LAYOUT["Big Phrase-Word Slide RED"],
              notes="Suggested presenter: Carson.\n"
                    "Thank the audience. Repeat each question before answering. Backup slides cover: the decoding "
                    "bottleneck, exact gap numbers, complexity, the per-round metric, preprocessing steps, input "
                    "shapes per model, dataset statistics, and the full work breakdown.")
for ph in list(s.placeholders):
    ph._element.getparent().remove(ph._element)
text(s, 0.7, 1.7, 8.6, 1.0, "Questions?", size=44, bold=True, color=WHITE)
text(s, 0.7, 2.75, 8.6, 1.6, [["MWPM is fast but blind to correlated errors, which cause ~95% of its gap to optimal "
                               "at d = 3."],
                              ["We will train two ML decoders on Stim data to learn those correlations and beat MWPM "
                               "on the same shots."]], size=18, color=WHITE, spacing=10)
text(s, 0.7, 4.7, 8.6, 0.5, "Team 22  ·  Ali Moallin  ·  Carson Brice  ·  Daniel Kowalski  ·  Nate Herman",
     size=14, color=WHITE)
slides_order.append(s)

# ---- 23-24 References (reuse + continuation)
REFS2 = [
    "[10]\tO. Higgott and C. Gidney, “Sparse blossom: Correcting a million errors per core second with "
    "minimum-weight matching,” Quantum, vol. 9, p. 1600, Jan. 2025, doi: 10.22331/q-2025-01-20-1600.",
    "[11]\tE. Dennis, A. Kitaev, A. Landahl, and J. Preskill, “Topological quantum memory,” J. Math. Phys., vol. 43, "
    "no. 9, pp. 4452–4505, 2002, doi: 10.1063/1.1499754.",
    "[12]\tA. W. Senior et al., “A scalable and real-time neural decoder for topological quantum codes,” "
    "arXiv:2512.07737, 2026.",
    "[13]\tC. Lee, T. Hur, J. Jae, and D. K. Park, “Machine learning approaches to decoding topological quantum "
    "codes,” arXiv:2608.15760, 2026.",
    "[14]\tGoogle Quantum AI, “Suppressing quantum errors by scaling a surface code logical qubit,” Nature, vol. 614, "
    "no. 7949, pp. 676–681, Feb. 2023, doi: 10.1038/s41586-022-05434-1.",
    "[15]\tR. Blume-Kohout and K. Young, “Estimating detector error models from syndrome data,” arXiv:2504.14643, "
    "2025.",
    "[16]\tM. Lange et al., “Data-driven decoding of quantum error correcting codes using graph neural networks,” "
    "Phys. Rev. Research, vol. 7, Art. no. 023181, 2025.",
    "[17]\tB. M. Varbanov, M. Serra-Peralta, D. Byfield, and B. M. Terhal, “Neural network decoder for near-term "
    "surface-code experiments,” Phys. Rev. Research, vol. 7, Art. no. 013029, 2025.",
    "[18]\tC. Gidney, M. Newman, and M. McEwen, “Benchmarking the planar honeycomb code,” Quantum, vol. 6, p. 813, "
    "2022, doi: 10.22331/q-2022-09-21-813.",
    "[19]\tP. Iyer and D. Poulin, “Hardness of decoding quantum stabilizer codes,” IEEE Trans. Inf. Theory, vol. 61, "
    "no. 9, pp. 5209–5223, Sep. 2015, doi: 10.1109/TIT.2015.2422294.",
    "[20]\tJ. Edmonds, “Paths, trees, and flowers,” Can. J. Math., vol. 17, pp. 449–467, 1965, "
    "doi: 10.4153/CJM-1965-045-4.",
    "[21]\tE. R. Berlekamp, R. J. McEliece, and H. C. A. van Tilborg, “On the inherent intractability of certain "
    "coding problems,” IEEE Trans. Inf. Theory, vol. 24, no. 3, pp. 384–386, May 1978, doi: 10.1109/TIT.1978.1055873.",
    "[S]\tTeam 22, Stim 1.16.0 / PyMatching 2.4.0 simulations, mwpm.ipynb and stim_pipeline, unpublished, "
    "Sep.–Oct. 2026.",
]
retitle(S_REFS, "References")
slides_order.append(S_REFS)
s = new_slide("References (continued)", notes="Not presented aloud.")
ref_src = next(sh for sh in S_REFS.shapes if sh.has_text_frame and "[1]" in sh.text_frame.text)
text(s, ref_src.left / 914400, 1.75, ref_src.width / 914400, 3.95, [[(r_, {})] for r_ in REFS2], size=9,
     color=INK, spacing=3)
slides_order.append(s)

# ================================================================== backup
s = new_slide(None, layout=LAYOUT["Big Phrase-Word Slide RED"], notes="Backup slides for Q&A.")
for ph in list(s.placeholders):
    ph._element.getparent().remove(ph._element)
text(s, 0.7, 2.4, 8.6, 1.0, "Backup slides", size=40, bold=True, color=WHITE)
slides_order.append(s)

retitle(S_BG, "Backup: decoding must keep up with ~1 µs rounds")
slides_order.append(S_BG)

# B: exact gap table
s = new_slide("Backup: exact logical error rates of four decoders at d = 3, 3 rounds",
              "Exact, no sampling: the full joint distribution P(s, L) over all 2^25 outcomes, folded mechanism by "
              "mechanism. 'Graph' = MWPM's model (product of edge marginals); 'true' = full DEM with hyperedges. "
              "'max' = single most likely error; 'sum' = whole logical class (optimal). Graph-max reproduces "
              "PyMatching on every syndrome (mismatch 0). The split depends on which fix is applied first, so the "
              "share column is the average of both orders. Source: mwpm.ipynb, research_log/mwpm_gap_derivation.md.")
rows_ = [["p", "Optimal", "True DEM, max", "Graph, sum", "MWPM", "MWPM ÷ opt", "Correlation share"],
         ["0.1%", "6.10e-4", "6.18e-4", "7.65e-4", "7.70e-4", "1.26×", "96%"],
         ["0.2%", "2.42e-3", "2.45e-3", "2.97e-3", "3.00e-3", "1.24×", "95%"],
         ["0.3%", "5.37e-3", "5.45e-3", "6.50e-3", "6.56e-3", "1.22×", "94%"],
         ["0.5%", "1.44e-2", "1.47e-2", "1.70e-2", "1.72e-2", "1.19×", "92%"]]
table(s, 0.5, 2.0, 9.0, rows_, [0.8, 1.2, 1.4, 1.3, 1.2, 1.35, 1.75], size=13, row_h=0.45)
text(s, 0.5, 4.45, 9.0, 1.1, [["Where MWPM's call differs from optimal, fixing correlations alone recovers 84–89% of "
                               "the weight; fixing degeneracy alone recovers ≤ 1%."],
                              [("Uniform p on 4 Stim noise channels; rotated memory-Z [S].", {"color": GRAY})]],
     size=14, spacing=6)
slides_order.append(s)

# B: complexity
s = new_slide("Backup: graph-like errors are the line between tractable and intractable decoding",
              "Min-weight decoding with column weight ≤ 2 reduces to perfect matching (in P). With weight-3 columns it "
              "is NP-complete, by reduction from 3-dimensional matching [21]. Exact maximum-likelihood decoding is "
              "#P-hard for stabilizer codes in general [19]. These are worst-case results; for the surface code the "
              "exact cost grows exponentially in the lattice cross-section, d^2, and linearly in rounds. MWPM runtime "
              "is near-linear in the number of defects on random noise and O(k^3) on adversarial syndromes [10]. "
              "Source: research_log/decoder_complexity.md.")
rows_ = [["Problem", "Graph-like H (≤ 2 detectors per fault)", "General H (≥ 3 detectors per fault)"],
         ["Most likely error (MWPM)", "Polynomial: perfect matching, blossom [20]", "NP-complete [21]"],
         ["Most likely logical class (optimal)", "Hard in general", "#P-hard [19]"],
         ["Belief propagation", "Cheap, inexact (loops, degeneracy)", "Cheap, inexact"]]
table(s, 0.5, 2.0, 9.0, rows_, [2.8, 3.3, 2.9], size=13, row_h=0.55, bold_first_col=True)
rows_ = [["d = 3 DEM", "Detectors", "Mechanisms", "Hyperedges (3–4 detectors)"],
         ["3 rounds", "24", "219", "113 (52%)"],
         ["25 rounds", "200", "3,056", "1,669 (55%)"]]
table(s, 0.5, 4.4, 9.0, rows_, [2.8, 1.6, 1.7, 2.9], size=13, row_h=0.4, bold_first_col=True)
slides_order.append(s)

# B: per-round metric
s = new_slide("Backup: we compare decoders on error per round ε, because per-shot error compounds with rounds",
              "LER(r) = ½(1 − (1 − 2ε)^r) for every decoder, optimal included, so per-shot LER tends to ½ on long runs. "
              "We fit ln(1 − 2·LER) = ln A + r·ln(1 − 2ε) by weighted least squares over r ∈ {3, 5, 10, 25, 50, 100}; "
              "A absorbs the preparation and readout boundaries. Win condition: lower fitted ε at the same (d, p) with "
              "non-overlapping error bars, same r list, same fit. Sampling: 1000 failures or 2M shots per point [S].")
text(s, 0.5, 2.0, 9.0, 0.6, [[("LER(r) = ½ [1 − (1 − 2ε)ʳ]", {}),
                               ("      → ½ for every decoder as r grows", {"color": GRAY})]], size=20)
rows_ = [["p", "ε per round (MWPM)", "LER at r = 25", "LER at r = 100"],
         ["0.1%", "2.40e-4 ± 0.04e-4", "0.62%", "2.3%"],
         ["0.2%", "9.26e-4 ± 0.08e-4", "2.3%", "8.5%"],
         ["0.3%", "2.02e-3 ± 0.01e-3", "4.8%", "16.7%"],
         ["0.5%", "5.34e-3 ± 0.03e-3", "12.0%", "32.8%"]]
table(s, 0.5, 2.85, 9.0, rows_, [1.5, 3.0, 2.25, 2.25], size=13, row_h=0.42)
caption(s, 0.5, 5.1, 9.0, "d = 3 rotated memory-Z, uniform p on 4 channels, PyMatching; ε ∝ p² as expected at d = 3 [S]")
slides_order.append(s)

# B: preprocessing steps
s = new_slide("Backup: twelve preprocessing steps, by priority",
              "From the team's QEC Decoder Data Notes, Part III. Skipped on purpose: normalization (inputs are 0/1), "
              "deduplication (repeats are the real distribution), class re-weighting, and lattice-symmetry "
              "augmentation (Stim's CNOT order breaks some symmetries).")
rows_ = [["#", "Step", "Priority", "Applies to"],
         ["1", "Reorder into a [round, check] grid", "MUST", "RNN, CNN, transformer"],
         ["2", "Mask channel (which slots exist)", "MUST", "any grid model"],
         ["3", "Flag the final (data-readout) row", "MUST", "all"],
         ["4", "float32; labels kept separate", "MUST", "all"],
         ["5", "Split by seed; shuffle whole shots", "MUST", "all"],
         ["6", "Check-type channel (X / Z)", "SHOULD", "all; essential across sizes"],
         ["7", "Lay each round out as a 2D image", "SHOULD", "CNN only"],
         ["8", "Train on a mix of noise levels", "SHOULD", "training set"],
         ["9", "Basis flag", "SHOULD", "if one model sees both bases"],
         ["10", "Cumulative-XOR channel", "OPTIONAL", "all"],
         ["11", "Drop empty shots from training", "OPTIONAL", "training set only"],
         ["12", "Vary the number of rounds", "OPTIONAL", "recurrent models"]]
table(s, 0.5, 1.9, 9.0, rows_, [0.5, 4.0, 1.3, 3.2], size=11, header_size=11, row_h=0.29)
slides_order.append(s)

# B: input shapes per model
s = new_slide("Backup: input shape per architecture (d = 3, 3 rounds)",
              "From Model_input_shape.pdf. Constant channels (mask, type, final flag) are stored once and broadcast "
              "per batch.")
rows_ = [["Model", "Tensor into first layer", "PyTorch layer", "Weights shared across"],
         ["MLP", "dets [B, 24]", "nn.Linear", "nothing: one weight per input"],
         ["CNN", "[B, 5, 4, 4, 4]", "nn.Conv3d", "lattice positions and rounds"],
         ["RNN (GRU)", "[B, 4, 40]", "nn.GRU", "rounds"],
         ["Transformer", "[B, 32, 5] + slot/round embeddings", "nn.TransformerEncoderLayer", "check-rounds (tokens)"],
         ["GNN", "nodes [k, 4] (x, y, t, type), edges [2, E]", "PyTorch Geometric", "nodes and edges"]]
table(s, 0.5, 2.0, 9.0, rows_, [1.4, 3.1, 2.3, 2.2], size=12, row_h=0.5, bold_first_col=True)
slides_order.append(s)

# B: dataset stats
s = new_slide("Backup: logical errors are rare, so accuracy is the wrong metric and datasets must be large",
              "Per 100,000 shots, d = 3, Z basis, 3 rounds, uniform circuit noise [S]. At p = 0.1%, only 77 shots per "
              "100,000 are ones MWPM gets wrong: those are the informative examples. Training at elevated noise is a "
              "form of importance sampling. At an error rate of 0.05%, 100,000 shots hold about 50 failures, so the "
              "statistical uncertainty is about 14%; we report Wilson score intervals.")
rows_ = [["Noise p", "Label = 1", "MWPM wrong", "Empty shots"],
         ["0.1%", "2.3%", "77", "84.0%"],
         ["0.2%", "4.5%", "333", "70.8%"],
         ["0.3%", "6.7%", "610", "59.6%"],
         ["0.5%", "10.5%", "1,763", "42.4%"],
         ["1%", "18.7%", "5,830", "18.2%"]]
table(s, 0.5, 2.0, 5.2, rows_, [1.2, 1.3, 1.35, 1.35], size=13, row_h=0.42)
box(s, 6.0, 2.0, 3.5, 2.52)
text(s, 6.2, 2.1, 3.1, 2.4, [[("A model that always says “no flip” is 90% accurate at d = 5, p = 0.1%", {})],
                              [("Report LER beside MWPM's, on the same shots, with confidence intervals",
                                {"bold": True, "color": SCARLET})]], size=14, spacing=10)
caption(s, 0.5, 4.65, 9.0, "Per 100,000 shots; d = 3, Z basis, 3 rounds, uniform circuit noise [S]")
slides_order.append(s)

# B: WBS
s = new_slide("Backup: work breakdown structure",
              "Seven phases plus schedule buffers. Owners, done criteria and dates are in "
              "deliverables/QEC_Project_Schedule.xlsx (WBS and Gantt tabs).")
box(s, 3.5, 1.85, 3.0, 0.48, fill=SCARLET)
text(s, 3.5, 1.85, 3.0, 0.48, "ML Decoders for QEC", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
     anchor=MSO_ANCHOR.MIDDLE)
wbs = [("1 Project mgmt", ["PDR", "Sem 1 slides", "Final report, expo"]),
       ("2 Research & baselines", ["Lit. survey", "MWPM ✓", "Gap analysis ✓"]),
       ("3 Data pipeline", ["Format & shapes", "Stim generator", "Datasets"]),
       ("4 ML prototype", ["Pick model", "PoC vs MWPM", "2 prelim models"]),
       ("5 Decoders", ["#1 FF / CNN", "#2 RNN / Transf.", "Benchmarks"]),
       ("6 Optimize", ["Pruning", "Quantization", "Re-benchmark"]),
       ("7 FPGA (stretch)", ["Board & HLS", "Synthesis", "On-board test"])]
WW, WG = 1.2, 0.1
bar = nostyle(s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.5 + WW / 2), Inches(2.55),
                                     Inches(0.5 + 6 * (WW + WG) + WW / 2), Inches(2.55)))
bar.line.color.rgb = MID
v = nostyle(s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(5.0), Inches(2.33), Inches(5.0), Inches(2.55)))
v.line.color.rgb = MID
for k, (head, kids) in enumerate(wbs):
    x = 0.5 + k * (WW + WG)
    v = nostyle(s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + WW / 2), Inches(2.55),
                                       Inches(x + WW / 2), Inches(2.7)))
    v.line.color.rgb = MID
    box(s, x, 2.7, WW, 0.72, fill=INK if k < 4 else GRAY)
    text(s, x + 0.04, 2.7, WW - 0.08, 0.72, head, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    for j, kid in enumerate(kids):
        box(s, x, 3.55 + j * 0.62, WW, 0.52)
        text(s, x + 0.04, 3.55 + j * 0.62, WW - 0.08, 0.52, kid, size=11, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
slides_order.append(s)

# ================================================================== order + cleanup
sldIdLst = prs.slides._sldIdLst
id_by_slide = {}
for sid in list(sldIdLst):
    rid = sid.get(qn("r:id"))
    id_by_slide[prs.part.related_part(rid)] = sid
keep = [id_by_slide[sl.part] for sl in slides_order]
for sid in list(sldIdLst):
    sldIdLst.remove(sid)
for sid in keep:
    sldIdLst.append(sid)
# drop the two replaced slides from the package
for sl in (S_PDEF, S_SCOPE):
    for rel_id, rel in list(prs.part.rels.items()):
        if rel.target_part is sl.part:
            prs.part.drop_rel(rel_id)
for sl in slides_order:
    if sl.has_notes_slide:
        tf = sl.notes_slide.notes_text_frame
        if "[Team Member]" in tf.text:
            tf.text = tf.text.replace("Presenter: [Team Member].\n", "").replace("Presenter: [Team Member].", "")
prs.core_properties.title = "Team 22 PDR: ML Decoders for Quantum Error Correction"
prs.save(OUT)
print("saved", OUT, len(slides_order), "slides")
