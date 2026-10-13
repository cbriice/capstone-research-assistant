"""Anatomy of the syndrome where MWPM loses the most probability to the optimal decoder
(d=3, rounds=3, p=1e-3): exact P(s, L), the leading true explanations with their true cost
-ln(prod odds) vs MWPM's graph cost sum(w'), and which DEM lines feed MWPM's matched edges.

Run from code/:  .venv/bin/python claude/worst_syndrome.py   (~30 s, ~1 GB)
See research_log/mwpm_gap_derivation.md section 4.
"""
import itertools

import numpy as np
import pymatching

from exact_decoding import exact_joint, surface_circuit
from kernel_dims import dem_lines

c = surface_circuit(1e-3)
dec_dem = c.detector_error_model(decompose_errors=True)
P, n = exact_joint(c.detector_error_model(decompose_errors=False))
m = pymatching.Matching.from_detector_error_model(dec_dem)
coords = c.get_detector_coordinates()


def edge(ds):
    ds = sorted(ds)
    return m.get_edge_data(ds[0], ds[1]) if len(ds) == 2 else m.get_boundary_edge_data(ds[0])


def bits(s):
    return np.array([s >> i & 1 for i in range(n)], np.uint8)


# true mechanisms = decomposed DEM lines (each keeps its pieces, so its graph image is known)
mechs = []
for line, inst in zip(dem_lines(dec_dem), [i for i in dec_dem.flattened() if i.type == "error"]):
    sig, L = frozenset(), 0
    for d, l in line:
        sig, L = sig ^ d, L ^ l
    mechs.append((inst.args_copy()[0], sig, L, line))

# worst syndrome: largest |P(s,0) - P(s,1)| among syndromes where MWPM disagrees with ML
ml = (P[1] > P[0]).astype(int)
keep = np.nonzero(P.sum(0) > 1e-14)[0]
pred = m.decode_batch(((keep[:, None] >> np.arange(n)) & 1).astype(np.uint8))[:, 0]
dis = keep[pred != ml[keep]]
gain = np.abs(P[0, dis] - P[1, dis])
s = int(dis[np.argmax(gain)])
sset = frozenset(i for i in range(n) if s >> i & 1)
print(f"total gap {gain.sum():.4e} over {len(dis)} syndromes; worst s carries {gain.max():.3e}")
print("s =", {f"D{i}": tuple(int(x) for x in coords[i]) for i in sorted(sset)})
print(f"P(s,0)={P[0, s]:.3e}  P(s,1)={P[1, s]:.3e}  Lambda*={np.log(P[0, s] / P[1, s]):+.2f}  "
      f"ML={ml[s]}  MWPM={m.decode(bits(s))[0]}")

# leading explanations with 1-2 mechanisms
print("\nexplanation   L   true cost   graph cost")
exps = []
for k in (1, 2):
    for combo in itertools.combinations(range(len(mechs)), k):
        sig = frozenset()
        for j in combo:
            sig = sig ^ mechs[j][1]
        if sig == sset:
            exps.append(combo)
rows = []
for combo in exps:
    L = 0
    true_cost = 0.0
    pieces = {}
    for j in combo:
        p, _, l, line = mechs[j]
        L ^= l
        true_cost -= np.log(p / (1 - p))
        for pc in line:
            pieces[pc] = pieces.get(pc, 0) ^ 1  # pieces used twice cancel
    graph_cost = sum(edge(pc[0])["weight"] for pc, used in pieces.items() if used)
    rows.append((true_cost, L, graph_cost, combo))
for true_cost, L, graph_cost, combo in sorted(rows)[:6]:
    desc = " + ".join(f"[{' '.join(f'D{d}' for d in sorted(mechs[j][1]))}{' L0' if mechs[j][2] else ''}]" for j in combo)
    print(f"{desc:<34} {L}   {true_cost:6.2f}     {graph_cost:6.2f}")

# MWPM's matching and where its edge probabilities come from
print("\nMWPM matched edges and the DEM lines feeding each:")
for a, b in m.decode_to_edges_array(bits(s)):
    ds = frozenset([a] if b == -1 else [a, b])
    e = edge(ds)
    print(f"  {sorted(ds)}{'->boundary' if b == -1 else ''}  w'={e['weight']:.2f}  p'={e['error_probability']:.3e}")
    for p, sig, L, line in mechs:
        if any(pc[0] == ds for pc in line):
            kind = "alone" if len(line) == 1 else "piece of " + " ^ ".join(" ".join(f"D{d}" for d in sorted(pc[0])) for pc in line)
            print(f"      {p:.3e}  {kind}")
