import sys
from datetime import date, timedelta
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter as L
from openpyxl.worksheet.datavalidation import DataValidation

OUT = sys.argv[1]
D = lambda m, d: date(2027 if m < 7 else 2026, m, d)

# (id, name, type, owner, start, end, preds, status, linear, desc, date_src)
# type: Phase | Summary | Task | Milestone | Buffer ; date_src: Linear | Done | Est | TBD
R = []
def P(*a): R.append(a)

P("1", "Project management & course deliverables", "Phase", "All", None, None, "", "", "P-QEC-3", "Course-facing work, coordination, and documentation.", "")
P("1.1", "Team & tooling setup (Linear, GitHub repo, Obsidian vault)", "Task", "All", D(9,24), D(9,30), "", "Done", "", "Shared workspace, issue tracker, and research log in place.", "Done")
P("1.2", "Weekly PI meetings & Linear status upkeep (recurring)", "Task", "All", D(9,24), D(4,30), "", "In Progress", "", "Weekly meeting with Dr. Kim; keep Linear issues/dates current; update this schedule after each meeting.", "Est")
P("1.3", "Project schedule assignment", "Summary", "All", None, None, "", "", "", "", "")
P("1.3.1", "Build WBS, milestone table and Gantt chart", "Task", "All", D(10,13), D(10,15), "1.1", "In Progress", "", "This document. Baseline schedule; re-baselined at 1.6.1.", "Est")
P("1.3.2", "Project schedule submitted", "Milestone", "All", D(10,16), D(10,16), "1.3.1", "Planned", "", "Course milestone (placeholder date).", "TBD")
P("1.4", "Preliminary Design Review (PDR)", "Summary", "All", None, None, "", "", "", "", "")
P("1.4.1", "Build PDR slideshow", "Task", "All", D(10,14), D(10,22), "2.5.3", "Todo", "QEC-16", "Problem statement, MWPM baseline + optimal-decoder headroom (2.5), pipeline design (3.2), Sem 1/Sem 2 plan.", "Linear")
P("1.4.2", "PI review & rehearsal", "Task", "All", D(10,23), D(10,28), "1.4.1", "Planned", "", "Send deck to Dr. Kim; incorporate feedback; timed run-through.", "Est")
P("1.4.3", "PDR presentation (12:00 pm)", "Milestone", "All", D(10,29), D(10,29), "1.4.2", "Todo", "QEC-17", "Course milestone.", "Linear")
P("1.5", "Semester 1 deliverables", "Summary", "All", None, None, "", "", "", "", "")
P("1.5.1", "Code-base analysis report", "Task", "All", D(11,16), D(12,10), "2.3.2SS", "Planned", "", "PROJECT.md deliverable: structure, data formats and design choices of Stim, PyMatching and published neural decoders.", "Est")
P("1.5.2", "Semester 1 presentation slides", "Task", "All", D(11,30), D(12,7), "4.1.4SS", "Planned", "QEC-20", "Pipeline results, prototype PoC vs MWPM, Sem 2 plan.", "Est")
P("1.5.3", "Semester 1 presentation", "Milestone", "All", D(12,8), D(12,8), "1.5.2", "Todo", "QEC-20", "Course milestone. Linear says 'December'; exact date TBD.", "TBD")
P("1.5.4", "Code-base analysis report submitted", "Milestone", "All", D(12,11), D(12,11), "1.5.1", "Planned", "", "Course/project deliverable.", "TBD")
P("1.6", "Semester 2 course deliverables", "Summary", "All", None, None, "", "", "", "", "")
P("1.6.1", "Sem 2 kickoff: re-baseline schedule with PI", "Task", "All", D(1,11), D(1,15), "4.5", "Planned", "", "Update this schedule with actuals; confirm Sem 2 course dates; move Sem 2 items into Linear.", "TBD")
P("1.6.2", "Mid-semester design review prep", "Task", "All", D(2,15), D(2,23), "5.1.3", "Planned", "", "Decoder #1 results, decoder #2 status, FPGA go/no-go outlook.", "Est")
P("1.6.3", "Mid-semester design review", "Milestone", "All", D(2,24), D(2,24), "1.6.2", "Planned", "", "Course milestone (placeholder date).", "TBD")
P("1.6.4", "Final poster, slides & demo prep", "Task", "All", D(4,5), D(4,14), "5.6", "Planned", "", "Poster/slides for the design expo.", "Est")
P("1.6.5", "Final presentation / Design Expo", "Milestone", "All", D(4,15), D(4,15), "1.6.4", "Planned", "", "Course milestone (placeholder date).", "TBD")
P("1.6.6", "Final report", "Task", "All", D(4,5), D(4,22), "5.6", "Planned", "", "Code-base analysis, decoder implementations, benchmarks, accuracy-vs-latency study, hardware results if reached.", "Est")
P("1.6.7", "Open-source repository release", "Task", "All", D(4,12), D(4,22), "5.6", "Planned", "", "Cleaned repo with docs, tests and reproducible benchmark scripts.", "Est")
P("1.6.8", "Final report & repository submitted", "Milestone", "All", D(4,23), D(4,23), "1.6.6, 1.6.7", "Planned", "", "Course milestone (placeholder date).", "TBD")

P("2", "QEC research & classical baselines", "Phase", "All", None, None, "", "", "", "Step 1: understand QEC, current ML strategies, and the MWPM baseline we must beat.", "")
P("2.1", "QEC background from PI (surface code, stabilizers, DEMs)", "Task", "All", D(9,24), D(10,9), "", "Done", "", "PI-provided context in project_docs_context/.", "Done")
P("2.2", "Literature survey by architecture family (Lee et al., arXiv:2608.15760)", "Summary", "All", None, None, "", "", "", "", "")
P("2.2.1", "Feedforward / CNN and recurrent decoders", "Task", "TBD", D(10,5), D(10,23), "2.1SS", "In Progress", "", "Summarize inputs, labels, results and latency of each.", "Est")
P("2.2.2", "Transformer decoders: AlphaQubit 1 & 2 deep dive", "Task", "TBD", D(10,12), D(10,30), "2.1", "Planned", "", "Target algorithm for reproduction (Step 2).", "Est")
P("2.2.3", "GNN and state-space-model decoders", "Task", "TBD", D(10,19), D(11,6), "2.1", "Planned", "", "", "Est")
P("2.2.4", "Architecture shortlist for Sem 2 (decoder #1 & #2)", "Milestone", "All", D(11,20), D(11,20), "2.2.2, 2.2.3, 2.3.2", "Planned", "", "Decision point.", "Est")
P("2.3", "Code-base analysis of open-source repos", "Summary", "All", None, None, "", "", "", "", "")
P("2.3.1", "Stim & PyMatching internals (DEM, hyperedge decomposition, matching graph)", "Task", "Carson Brice", D(10,1), D(10,30), "2.1SS", "In Progress", "", "Partly done via research_log/mwpm_gap_derivation.md.", "Est")
P("2.3.2", "Published neural-decoder repos (structure, data formats, design choices)", "Task", "TBD", D(10,26), D(11,20), "2.2.1", "Planned", "", "Feeds code-base analysis report (1.5.1).", "Est")
P("2.4", "MWPM baseline", "Summary", "All", None, None, "", "", "", "", "")
P("2.4.1", "Reproduce MWPM baseline in PyMatching (mwpm.ipynb)", "Task", "TBD", D(9,29), D(10,7), "2.1SS", "Done", "", "LER vs p, d, r on Stim surface-code circuits.", "Done")
P("2.4.2", "MWPM baseline reproduced", "Milestone", "All", D(10,8), D(10,8), "2.4.1", "Done", "", "", "Done")
P("2.4.3", "Correlated matching (two-pass reweighting) as stronger baseline", "Task", "TBD", D(1,11), D(1,29), "2.4.2", "Planned", "", "From research_log open questions.", "Est")
P("2.5", "MWPM vs optimal-decoder gap analysis", "Summary", "Carson Brice", None, None, "", "", "QEC-11", "", "")
P("2.5.1", "Exact optimal-decoder ceiling at d=3", "Task", "Carson Brice", D(10,7), D(10,8), "2.4.1", "Done", "QEC-11", "MWPM LER is 1.19-1.26x optimal: 16-21% headroom.", "Done")
P("2.5.2", "Derivation of why MWPM picks the wrong logical class", "Task", "Carson Brice", D(10,8), D(10,13), "2.5.1", "Done", "QEC-11", "Correlations lost when hyperedges are decomposed.", "Done")
P("2.5.3", "Optimal ceiling & MWPM gap derived", "Milestone", "Carson Brice", D(10,13), D(10,13), "2.5.2", "Done", "QEC-11", "", "Done")
P("2.5.4", "Theory write-up for Dr. Kim", "Task", "Carson Brice", D(10,14), D(10,30), "2.5.3", "Todo", "QEC-11", "Sub-issue of QEC-11; readable account of the gap theory.", "Linear")
P("2.5.5", "Split the d=3 gap into correlation vs degeneracy over all syndromes", "Task", "Carson Brice", D(10,19), D(11,6), "2.5.3", "Planned", "", "From research_log open questions.", "Est")
P("2.5.6", "Success metric agreed with PI", "Milestone", "All", D(10,23), D(10,23), "2.5.3", "Planned", "", "Relative LER at fixed (d,p,r) with error bars, plus fraction of the MWPM-to-optimal gap closed at d=3.", "Est")

P("3", "Data pipeline (Stim)", "Phase", "All", None, None, "", "", "P-QEC-1", "Generates labeled syndrome datasets and formats them for each architecture.", "")
P("3.1", "Determine method of generating training/test data", "Task", "TBD", D(9,24), D(10,7), "", "Done", "QEC-5", "", "Done")
P("3.2", "Data format & model-input design", "Summary", "All", None, None, "", "", "", "", "")
P("3.2.1", "Decide on formatting for generated data", "Task", "TBD", D(10,7), D(10,29), "3.1", "In Progress", "QEC-7", "", "Linear")
P("3.2.2", "Profile optimal data structures per architecture (CNN, RNN, GNN, transformer, JEPA)", "Task", "Nate Herman", D(10,7), D(10,30), "3.1", "In Progress", "QEC-9", "", "Linear")
P("3.2.3", "Mathematically define data masking (effect on training & inference)", "Task", "Daniel Kowalski", D(10,7), D(10,30), "3.1", "In Progress", "QEC-10", "", "Linear")
P("3.2.4", "Determine input shape per model", "Task", "TBD", D(10,19), D(10,30), "3.2.2SS, 3.2.3SS", "Todo", "QEC-14", "Shape of syndromes, feature transformations and masks for each model.", "Linear")
P("3.2.5", "Data format & input shape frozen", "Milestone", "All", D(10,30), D(10,30), "3.2.1, 3.2.4", "Planned", "", "Decision point (Linear milestone 'Initial Presentation 10/29').", "Linear")
P("3.3", "Pipeline implementation", "Summary", "All", None, None, "", "", "", "", "")
P("3.3.1", "Parametrized Stim circuit generator (d, r, p, basis)", "Task", "TBD", D(10,13), D(10,31), "3.1", "Todo", "QEC-18", "", "Linear")
P("3.3.2", "Feature transformation steps", "Task", "TBD", D(10,19), D(10,31), "3.2.3SS", "Todo", "QEC-19", "Turn any syndrome output + circuit into model-ready input.", "Linear")
P("3.3.3", "Batching simulated data during training to manage memory", "Task", "Ali Moallin", D(10,7), D(10,30), "3.1", "In Progress", "QEC-8", "", "Linear")
P("3.3.4", "Adapter code: Stim to per-architecture data packages", "Task", "TBD", D(10,13), D(11,13), "3.3.1SS, 3.2.5FF", "In Progress", "QEC-13", "Formatting chosen by a parameter.", "Linear")
P("3.3.5", "Data pipeline code complete", "Milestone", "All", D(11,13), D(11,13), "3.3.2, 3.3.3, 3.3.4", "Planned", "", "Linear milestone 'Write code for data pipeline'.", "Linear")
P("3.4", "Validation & dataset production", "Summary", "All", None, None, "", "", "", "", "")
P("3.4.1", "Pipeline tests: label correctness, MWPM LER reproduced through pipeline", "Task", "TBD", D(11,16), D(11,20), "3.3.4", "Planned", "", "Unit tests; cross-check against mwpm.ipynb.", "Est")
P("3.4.2", "Generate train/val/test sets (d=3,5; p sweep; multiple rounds)", "Task", "TBD", D(11,23), D(11,30), "3.4.1", "Planned", "", "Sized for rare-event statistics.", "Est")
P("3.4.3", "Datasets generated & pipeline validated", "Milestone", "All", D(11,30), D(11,30), "3.4.2", "Planned", "", "Linear project target date.", "Linear")
P("3.5", "Google Sycamore dataset (Zenodo 6804040): ingest & explore", "Task", "TBD", D(1,11), D(1,22), "3.3.5", "Planned", "", "Experimental data for Sem 2 finetuning & benchmarking.", "Est")

P("4", "ML decoder prototype (Semester 1)", "Phase", "All", None, None, "", "", "P-QEC-2", "Proof of concept that the pipeline trains a model end to end.", "")
P("4.1", "Minimum-feature-transformation proof of concept", "Summary", "All", None, None, "", "", "QEC-12", "Train on raw syndromes; learn which features and masks are needed.", "")
P("4.1.1", "Choose one model for prototype", "Task", "TBD", D(11,2), D(11,6), "3.2.5", "Todo", "QEC-21", "", "Est")
P("4.1.2", "Write prototype model code connected to the data pipeline", "Task", "TBD", D(11,16), D(11,20), "4.1.1, 3.3.5", "Todo", "QEC-23", "", "Est")
P("4.1.3", "Generalize prototype code to be model-agnostic", "Task", "TBD", D(11,23), D(12,1), "4.1.2", "Todo", "QEC-24", "", "Est")
P("4.1.4", "Basis experiments: RNN on Z only, X only, and both bases (small training set)", "Task", "TBD", D(11,23), D(12,3), "4.1.2", "Planned", "QEC-12", "", "Est")
P("4.1.5", "Proof-of-concept results vs MWPM", "Milestone", "All", D(12,4), D(12,4), "4.1.4", "Todo", "QEC-12", "Linear due date.", "Linear")
P("4.2", "Decide: one adaptive model vs multiple specialized models", "Task", "TBD", D(11,23), D(12,1), "4.1.2", "Todo", "QEC-15", "Across rounds, distances and bases.", "Linear")
P("4.3", "ML prototype complete", "Milestone", "All", D(12,7), D(12,7), "4.1.5, 4.2", "Planned", "", "Linear project target date.", "Linear")
P("4.4", "Train 2 preliminary models", "Task", "TBD", D(12,2), D(12,15), "4.1.3, 3.4.3", "Todo", "QEC-22", "Gives a starting point on the ML side for Sem 2.", "Linear")
P("4.5", "Two preliminary models trained", "Milestone", "All", D(12,15), D(12,15), "4.4", "Planned", "QEC-22", "", "Linear")

P("5", "Decoder implementation & benchmarking (Semester 2)", "Phase", "All", None, None, "", "", "", "Step 2: implement and reproduce; benchmark against MWPM.", "")
P("5.1", "Decoder #1: feedforward / CNN", "Summary", "TBD", None, None, "", "", "", "", "")
P("5.1.1", "Scale training to full datasets (d=3,5; p sweep)", "Task", "TBD", D(1,11), D(1,29), "4.5, 1.6.1SS", "Planned", "", "", "Est")
P("5.1.2", "Hyperparameter tuning & ablations (masks, features)", "Task", "TBD", D(2,1), D(2,11), "5.1.1", "Planned", "", "", "Est")
P("5.1.3", "Decoder #1 benchmarked vs MWPM", "Milestone", "All", D(2,12), D(2,12), "5.1.2", "Planned", "", "", "Est")
P("5.2", "Decoder #2: recurrent / transformer (AlphaQubit-style)", "Summary", "TBD", None, None, "", "", "", "", "")
P("5.2.1", "Architecture design & implementation", "Task", "TBD", D(1,18), D(2,12), "2.2.4", "Planned", "", "", "Est")
P("5.2.2", "Pretrain on simulated data (SI1000 or DEM fitted to device)", "Task", "TBD", D(2,15), D(3,5), "5.2.1, 3.4.3", "Planned", "", "", "Est")
P("5.2.3", "Finetune on Sycamore experimental data", "Task", "TBD", D(3,8), D(3,19), "5.2.2, 3.5", "Planned", "", "Spans spring break (B3); extra calendar time built in.", "Est")
P("5.3", "Benchmarking", "Summary", "TBD", None, None, "", "", "", "", "")
P("5.3.1", "LER vs MWPM & correlated matching on simulated data", "Task", "TBD", D(2,8), D(3,19), "5.1.1, 2.4.3", "Planned", "", "Error bars; fraction of MWPM-to-optimal gap closed at d=3.", "Est")
P("5.3.2", "LER on Google experimental dataset", "Task", "TBD", D(3,22), D(3,26), "5.2.3", "Planned", "", "", "Est")
P("5.3.3", "Generalization tests (unseen d, p, rounds)", "Task", "TBD", D(3,8), D(3,26), "5.2.2", "Planned", "", "", "Est")
P("5.4", "Latency profiling vs ~1 us/round budget (CPU/GPU)", "Task", "TBD", D(2,22), D(3,26), "5.1.3", "Planned", "", "", "Est")
P("5.5", "Accuracy-vs-latency study across architectures", "Task", "TBD", D(3,29), D(4,9), "5.3.1, 5.4", "Planned", "", "Core deliverable from PROJECT.md.", "Est")
P("5.6", "Decoder #2 benchmarked on simulated & experimental data", "Milestone", "All", D(3,26), D(3,26), "5.3.2, 5.3.3", "Planned", "", "", "Est")

P("6", "Optimization & quantization", "Phase", "TBD", None, None, "", "", "", "Step 3: compress the best model for hardware.", "")
P("6.1", "Select best model & set compression targets", "Task", "TBD", D(3,22), D(3,24), "5.3.1", "Planned", "", "", "Est")
P("6.2", "Pruning & quantization (PTQ, then QAT; INT8 and below)", "Task", "TBD", D(3,25), D(4,2), "6.1", "Planned", "", "", "Est")
P("6.3", "Re-benchmark compressed model (LER & latency)", "Task", "TBD", D(4,5), D(4,8), "6.2", "Planned", "", "", "Est")
P("6.4", "Compressed model within accuracy budget", "Milestone", "All", D(4,9), D(4,9), "6.3", "Planned", "", "LER loss vs float model within tolerance agreed at 2.5.6.", "Est")

P("7", "Stretch: FPGA deployment", "Phase", "TBD", None, None, "", "", "", "Step 4 (stretch): real-time inference in hardware on unseen data.", "")
P("7.1", "Select board & set up HLS toolchain (e.g. hls4ml / Vitis HLS)", "Task", "TBD", D(2,15), D(3,11), "5.1.3", "Planned", "", "Started early to cover procurement lead time.", "Est")
P("7.2", "FPGA go / no-go decision", "Milestone", "All", D(3,12), D(3,12), "7.1", "Planned", "", "Decision point: board in hand + toolchain runs a toy model.", "Est")
P("7.3", "HLS conversion & synthesis of compressed model", "Task", "TBD", D(4,5), D(4,14), "6.2, 7.2", "Planned", "", "", "Est")
P("7.4", "On-board latency measurement & live test on unseen data", "Task", "TBD", D(4,15), D(4,20), "7.3", "Planned", "", "", "Est")
P("7.5", "Hardware real-time decoding demo", "Milestone", "All", D(4,21), D(4,21), "7.4", "Planned", "", "Stretch milestone.", "Est")

P("8", "Schedule reserve (buffers)", "Phase", "All", None, None, "", "", "", "Planned slack for slips and term breaks.", "")
P("8.1", "B1: Sem 1 finals buffer", "Buffer", "All", D(12,9), D(12,15), "", "Planned", "", "Absorbs slips in data pipeline / prototype.", "TBD")
P("8.2", "B2: Winter break (catch-up only)", "Buffer", "All", D(12,16), D(1,10), "", "Planned", "", "No planned work.", "TBD")
P("8.3", "B3: Spring break (reduced capacity)", "Buffer", "All", D(3,15), D(3,19), "", "Planned", "", "Placeholder date.", "TBD")
P("8.4", "B4: End-of-project buffer", "Buffer", "All", D(4,24), D(4,30), "", "Planned", "", "Final slack before finals week.", "TBD")

MILESTONES = [  # (wbs id, type, associated tasks)
    ("2.4.2", "Project", "2.4.1"),
    ("2.5.3", "Project", "2.5.1, 2.5.2"),
    ("1.3.2", "Course", "1.3.1"),
    ("2.5.6", "Decision", "2.5.3, 2.5.4"),
    ("1.4.3", "Course", "1.4.1, 1.4.2"),
    ("3.2.5", "Decision", "3.2.1-3.2.4"),
    ("3.3.5", "Project", "3.3.1-3.3.4"),
    ("2.2.4", "Decision", "2.2.1-2.2.3, 2.3.2"),
    ("3.4.3", "Project", "3.4.1, 3.4.2"),
    ("4.1.5", "Project", "4.1.1-4.1.4"),
    ("4.3", "Project", "4.1, 4.2"),
    ("1.5.3", "Course", "1.5.2"),
    ("1.5.4", "Course", "1.5.1, 2.3"),
    ("4.5", "Project", "4.4"),
    ("5.1.3", "Project", "5.1.1, 5.1.2"),
    ("1.6.3", "Course", "1.6.2"),
    ("7.2", "Decision", "7.1"),
    ("5.6", "Project", "5.2, 5.3"),
    ("6.4", "Project", "6.1-6.3"),
    ("1.6.5", "Course", "1.6.4, 5.5"),
    ("7.5", "Project (stretch)", "7.3, 7.4"),
    ("1.6.8", "Course", "1.6.6, 1.6.7"),
]
MS_OVERRIDE = {}

# ---------------- styles ----------------
F = "Arial"
font = lambda **k: Font(name=F, size=k.pop("size", 10), **k)
fill = lambda c: PatternFill("solid", start_color=c, end_color=c)
thin = Side(style="thin", color="D0D4DA")
box = Border(left=thin, right=thin, top=thin, bottom=thin)
HDR = fill("1F2A44")
PHASE_BG = fill("E8ECF2")
YEL = fill("FFF59D")
PHASE_COL = {"1": "5B6B7F", "2": "7E57C2", "3": "00897B", "4": "1E88E5",
             "5": "F4511E", "6": "43A047", "7": "D81B60", "8": "B0BEC5"}

wb = Workbook()

# ---------------- WBS: what the work is and who owns it ----------------
ws = wb.active
ws.title = "WBS"
ws["A1"] = "Machine Learning Decoders for Quantum Error Correction: Work Breakdown Structure"
ws["A1"].font = font(size=14, bold=True)
ws["A2"] = "Hierarchy of all project work, owners and done criteria. Dates, durations and dependencies are on the Gantt tab."
ws["A2"].font = font(italic=True, color="555555")
cols = ["WBS ID", "Task / deliverable", "Type", "Owner", "Status", "Linear", "Description / done criteria"]
widths = [8, 66, 11, 16, 12, 10, 90]
HR = 4
for i, (c, w) in enumerate(zip(cols, widths), 1):
    cell = ws.cell(HR, i, c)
    cell.font = font(bold=True, color="FFFFFF"); cell.fill = HDR
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.column_dimensions[L(i)].width = w
ws.row_dimensions[HR].height = 30
first = HR + 1
last = first + len(R) - 1
for k, (wid, name, typ, owner, s_, e_, preds, status, lin, desc, src) in enumerate(R):
    r = first + k
    for c, v in enumerate([wid, name, typ, owner, status, lin, desc], 1):
        cell = ws.cell(r, c, v)
        cell.font = font(bold=typ in ("Phase", "Summary"))
        cell.border = box
        cell.alignment = Alignment(vertical="top", wrap_text=c in (2, 7),
                                   horizontal="center" if c in (3, 5, 6) else None)
    ws.cell(r, 2).alignment = Alignment(indent=wid.count(".") * 2, vertical="top", wrap_text=True)
    if typ == "Phase":
        for c in range(1, len(cols) + 1):
            ws.cell(r, c).fill = PHASE_BG
        ws.cell(r, 2).font = font(bold=True, color=PHASE_COL[wid])
dv = DataValidation(type="list", formula1='"Planned,Todo,In Progress,Done,Canceled"', allow_blank=True)
ws.add_data_validation(dv); dv.add(f"E{first}:E{last}")
ws.conditional_formatting.add(f"E{first}:E{last}", FormulaRule(formula=[f'$E{first}="Done"'], fill=fill("C8E6C9")))
ws.conditional_formatting.add(f"E{first}:E{last}", FormulaRule(formula=[f'$E{first}="In Progress"'], fill=fill("BBDEFB")))
ws.freeze_panes = ws.cell(HR + 1, 3)
ws.auto_filter.ref = f"A{HR}:G{last}"
ws.page_setup.orientation = "landscape"; ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0
ws.sheet_properties.pageSetUpPr.fitToPage = True
ws.print_title_rows = f"{HR}:{HR}"

# ---------------- Milestone table ----------------
GH = 5                      # Gantt week-date header row
g0 = GH + 1                 # first Gantt task row
glast = g0 + len(R) - 1
ms = wb.create_sheet("Milestones")
ms["A1"] = "Milestone Table"; ms["A1"].font = font(size=14, bold=True)
ms["A2"] = "Name, status and description come from the WBS tab; dates come from the Gantt tab (matched by WBS ID). Yellow = placeholder course date to confirm."
ms["A2"].font = font(italic=True, color="555555")
mcols = ["#", "Milestone", "Type", "WBS ID", "Associated task(s)", "Expected date", "Status", "Description", "Date source"]
mw = [6, 50, 16, 9, 20, 13, 12, 70, 11]
for i, (c, w) in enumerate(zip(mcols, mw), 1):
    cell = ms.cell(4, i, c); cell.font = font(bold=True, color="FFFFFF"); cell.fill = HDR
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ms.column_dimensions[L(i)].width = w
wr = lambda col: f"WBS!${col}${first}:${col}${last}"
gr = lambda col: f"Gantt!${col}${g0}:${col}${glast}"
src_of = {r[0]: r[10] for r in R}
for n, (wid, mtype, assoc) in enumerate(MILESTONES, 1):
    r = 4 + n
    mw_ = f"MATCH($D{r},{wr('A')},0)"
    mg_ = f"MATCH($D{r},{gr('A')},0)"
    vals = [f"M{n:02d}",
            MS_OVERRIDE.get(wid) or f"=INDEX({wr('B')},{mw_})",
            mtype, wid, assoc,
            f"=INDEX({gr('D')},{mg_})",
            f"=INDEX({wr('E')},{mw_})",
            f'=IF(INDEX({wr("G")},{mw_})="","",INDEX({wr("G")},{mw_}))',
            src_of[wid]]
    for c, v in enumerate(vals, 1):
        cell = ms.cell(r, c, v); cell.font = font(color="008000" if isinstance(v, str) and v.startswith("=") else "000000")
        cell.border = box
        cell.alignment = Alignment(vertical="top", wrap_text=c in (2, 8),
                                   horizontal="center" if c in (1, 3, 4, 5, 6, 7, 9) else None)
    ms.cell(r, 6).number_format = "ddd m/d/yy"
    if src_of[wid] == "TBD":
        ms.cell(r, 6).fill = YEL
ml = 4 + len(MILESTONES)
ms.conditional_formatting.add(f"G5:G{ml}", FormulaRule(formula=['$G5="Done"'], fill=fill("C8E6C9")))
ms.freeze_panes = "C5"
ms.page_setup.orientation = "landscape"; ms.sheet_properties.pageSetUpPr.fitToPage = True
ms.page_setup.fitToWidth = 1; ms.page_setup.fitToHeight = 0

# ---------------- Gantt (static fills: renders identically in Excel, Numbers, Sheets, PDF) ----------------
T0 = date(2026, 9, 21)          # first Monday on the timeline
STATUS = date(2026, 10, 13)     # baseline status date
NW = 33
spans = {}
for rr in R:
    if rr[4]:
        spans[rr[0]] = (rr[4], rr[5])
for rr in R:
    if not rr[4]:
        kids = [x for x in R if x[0].startswith(rr[0] + ".") and x[4]]
        spans[rr[0]] = (min(x[4] for x in kids), max(x[5] for x in kids))

g = wb.create_sheet("Gantt")
g["A1"] = "Project Schedule: Gantt Chart"; g["A1"].font = font(size=14, bold=True)
g["A2"] = f"Baseline as of {STATUS:%-m/%-d/%y}. Bars = task span by phase color; ◆ = milestone; grey = done; hatched = buffer; red column = status week. Predecessors: FS default, SS = start-to-start, FF = finish-to-finish."
g["A2"].font = font(italic=True, color="555555", size=9)
gcols = ["WBS ID", "Task / milestone", "Start", "End", "Days", "Predecessors"]
gw = [7, 66, 9, 9, 6, 16]
W0 = len(gcols) + 1
for i, (c, w) in enumerate(zip(gcols, gw), 1):
    cell = g.cell(GH, i, c); cell.font = font(bold=True, color="FFFFFF"); cell.fill = HDR
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    g.cell(GH - 1, i).fill = HDR
    g.column_dimensions[L(i)].width = w
g.row_dimensions[GH].height = 34
weeks = [T0 + timedelta(days=7 * j) for j in range(NW)]
cur = next(j for j, w in enumerate(weeks) if w <= STATUS < w + timedelta(days=7))
RED = "E53935"
for j, wk in enumerate(weeks):
    c = W0 + j
    g.column_dimensions[L(c)].width = 3.8
    hc = g.cell(GH, c, f"{wk.month}/{wk.day}")
    hc.font = font(size=7, color="FFFFFF", bold=True); hc.fill = fill(RED) if j == cur else HDR
    hc.alignment = Alignment(text_rotation=90, horizontal="center", vertical="center")
    g.cell(GH - 1, c).fill = HDR
# month labels: merged across that month's weeks
j = 0
while j < NW:
    k = j
    while k + 1 < NW and weeks[k + 1].month == weeks[j].month:
        k += 1
    lab = weeks[j].strftime("%b") + (f" '{weeks[j]:%y}" if j == 0 or weeks[j].month == 1 else "")
    mc = g.cell(GH - 1, W0 + j, lab)
    mc.font = font(size=8, color="FFFFFF", bold=True)
    mc.alignment = Alignment(horizontal="left", vertical="center")
    if k > j:
        g.merge_cells(start_row=GH - 1, start_column=W0 + j, end_row=GH - 1, end_column=W0 + k)
    j = k + 1

hair = Side(style="hair", color="E0E0E0"); redline = Side(style="thin", color=RED)
for k, rr in enumerate(R):
    r = g0 + k
    wid, name, typ, owner, _, _, preds, status = rr[:8]
    s, e = spans[wid]
    days = 0 if typ == "Milestone" else (e - s).days + 1
    vals = [wid, name, s, e, days, preds]
    for c, v in enumerate(vals, 1):
        cell = g.cell(r, c, v); cell.border = box
        cell.font = font(size=9, bold=typ in ("Phase", "Summary"))
        cell.alignment = Alignment(horizontal="left" if c == 2 else "center", vertical="center",
                                   indent=wid.count(".") * 2 if c == 2 else 0)
    for c in (3, 4):
        g.cell(r, c).number_format = "m/d/yy"
        if rr[10] == "TBD":
            g.cell(r, c).fill = YEL
    if typ == "Phase":
        for c in range(1, W0):
            g.cell(r, c).fill = PHASE_BG
        g.cell(r, 2).font = font(size=9, bold=True, color=PHASE_COL[wid])
    for j, wk in enumerate(weeks):
        cell = g.cell(r, W0 + j)
        cell.border = Border(left=redline if j == cur else hair, bottom=hair)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        inwk = s <= wk + timedelta(days=6) and e >= wk
        if not inwk:
            continue
        if typ == "Milestone":
            cell.value = "◆"
            cell.font = font(size=12, bold=True, color="9E9E9E" if status == "Done" else "1F2A44")
        elif typ == "Buffer":
            cell.fill = PatternFill(patternType="lightUp", fgColor="90A4AE", bgColor="ECEFF1")
        elif typ == "Phase":
            cell.fill = fill("1F2A44")
        elif status == "Done":
            cell.fill = fill("BDBDBD")
        elif typ == "Summary":
            cell.fill = fill("546E7A")
        else:
            cell.fill = fill(PHASE_COL[wid[0]])
g.freeze_panes = g.cell(g0, W0)
g.page_setup.orientation = "landscape"; g.sheet_properties.pageSetUpPr.fitToPage = True
g.page_setup.fitToWidth = 1; g.page_setup.fitToHeight = 0
g.print_title_rows = f"{GH-1}:{GH}"

# ---------------- Notes ----------------
nt = wb.create_sheet("Notes")
notes = [
    ("Notes, assumptions & legend", True),
    ("", False),
    ("Sources", True),
    ("• Linear workspace 'QEC Capstone' (projects P-QEC-1 STIM data pipeline, P-QEC-2 ML prototype, P-QEC-3 deliverables), pulled 10/13/26. Linear issue IDs are in the WBS 'Linear' column.", False),
    ("• PROJECT.md (sponsor: Dr. Youngseok Kim): Sem 1 survey/reproduce/baseline, Sem 2 implement/benchmark/profile, stretch FPGA.", False),
    ("• research_log/: completed MWPM baseline and optimal-decoder gap analysis; open questions turned into tasks 2.4.3, 2.5.5, 3.5.", False),
    ("", False),
    ("Date source (Milestones tab; yellow dates on the Gantt and Milestones tabs)", True),
    ("• Linear: date or due date taken from Linear. Start dates are estimated when Linear only has a due date.", False),
    ("• Done: completed work, dated from Linear completion or git history.", False),
    ("• Est: planning estimate made for this schedule.", False),
    ("• TBD (yellow): placeholder course date (OSU calendar assumed: Autumn finals ~Dec 9-15, Spring term Jan 11-Apr 30, spring break ~Mar 15-19, expo ~Apr 15). Confirm and edit.", False),
    ("", False),
    ("Editing", True),
    ("• WBS tab = what the work is, who owns it, status and done criteria. Gantt tab = dates, durations and dependencies. Milestones tab pulls names/status from the WBS and dates from the Gantt by WBS ID.", False),
    ("• The Gantt is a static baseline (plain cell fills, so it renders the same in Excel, Numbers, Google Sheets and PDF). After changing dates, regenerate it with code/claude/build_schedule.py.", False),
    ("• Owner 'TBD' = unassigned in Linear; assign at the next team meeting.", False),
    ("• Predecessors are finish-to-start (FS) by default. SS = start-to-start (can begin once the predecessor has started, working from partial results); FF = finish-to-finish (cannot finish before the predecessor finishes).", False),
    ("", False),
    ("Risks & buffers", True),
    ("• Rare-event statistics: large datasets needed for LER error bars. Mitigation: 3.4.2 sized early; OSC compute if needed.", False),
    ("• Data format churn: 3.2.5 freeze milestone before prototype coding (4.1.2).", False),
    ("• Decoder #2 training cost / Sycamore finetune: 5.2.3 spans spring break; B4 holds end slack.", False),
    ("• FPGA procurement: board selection starts 2/15 with a go/no-go on 3/12 (7.2). If no-go, Step 4 is dropped and 6.x results go in the final report.", False),
    ("• Buffers: B1 (Sem 1 finals), B2 (winter break catch-up), B3 (spring break), B4 (end of project).", False),
]
for i, (t, b) in enumerate(notes, 1):
    c = nt.cell(i, 1, t); c.font = font(size=14 if i == 1 else 10, bold=b)
    c.alignment = Alignment(wrap_text=True, vertical="top")
nt.column_dimensions["A"].width = 130
r0 = len(notes) + 2
nt.cell(r0, 1, "Gantt colors by phase").font = font(bold=True)
names = {r[0]: r[1] for r in R if r[2] == "Phase"}
for i, (ph, colr) in enumerate(PHASE_COL.items(), 1):
    c = nt.cell(r0 + i, 1, f"{ph}  {names[ph]}"); c.font = font(color="FFFFFF", bold=True); c.fill = fill(colr)

wb._sheets = [wb["WBS"], wb["Milestones"], wb["Gantt"], wb["Notes"]]
ends = {r[0]: r[5] for r in R if r[5]}
starts = {r[0]: r[4] for r in R if r[4]}
import re
def span(p):
    if p in ends: return starts[p], ends[p]
    kids = [k for k in ends if k.startswith(p + ".")]
    return min(starts[k] for k in kids), max(ends[k] for k in kids)
for r in R:
    for tok in [x.strip() for x in r[6].split(",") if x.strip()]:
        p, kind = re.match(r"([\d.]+)(SS|FF)?$", tok).groups()
        ps, pe = span(p)
        bad = {"SS": ps > r[4], "FF": pe > r[5]}.get(kind, pe > r[4])
        if bad: print("VIOLATION", r[0], tok)
wb.save(OUT)
print("rows", len(R), "milestones", len(MILESTONES))
