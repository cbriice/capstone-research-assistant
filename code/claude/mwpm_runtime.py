"""PyMatching runtime: typical case (random circuit noise) vs adversarial case (contiguous defect block).

Run from code/:  .venv/bin/python claude/mwpm_runtime.py   (~3 min, dominated by the k=8001 block)
See research_log/decoder_complexity.md section 3.
"""
import time

import numpy as np
import stim
from pymatching import Matching


def circ(d, r, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=r,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p,
    )


print("== random circuit noise: time per shot, binned by number of defects k ==")
for d in (3, 11):
    for p in (1e-3, 3e-3, 6e-3, 1e-2):
        c = circ(d, d, p)
        m = Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        det, _ = c.compile_detector_sampler(seed=1).sample(200_000 if d == 3 else 40_000, separate_observables=True)
        k = det.sum(1)
        m.decode_batch(det[:100])
        t = time.perf_counter()
        m.decode_batch(det)
        T = time.perf_counter() - t
        bins = np.quantile(k, [0, .5, .9, .99, 1.0]).astype(int)
        rows = []
        for lo, hi, name in [(bins[0], bins[1], "<=median"), (bins[1] + 1, bins[2], "median-p90"),
                             (bins[2] + 1, bins[3], "p90-p99"), (bins[3] + 1, bins[4], ">p99")]:
            sel = det[(k >= lo) & (k <= hi)]
            if len(sel) < 20:
                continue
            t = time.perf_counter()
            m.decode_batch(sel)
            dt = (time.perf_counter() - t) / len(sel)
            rows.append(f"{name} k={lo}-{hi}: {dt * 1e6:.2f}us ({dt * 1e6 / max((lo + hi) / 2, 1):.3f}us/defect)")
        print(f"d={d} p={p:.0e} mean k={k.mean():.1f} mean {T / len(det) * 1e6:.2f}us/shot | " + " | ".join(rows), flush=True)

print("\n== adversarial: line graph, unit weights, contiguous block of k defects in the middle ==")
n = 20001
m = Matching()
for i in range(n - 1):
    m.add_edge(i, i + 1, weight=1.0)
m.add_boundary_edge(0, weight=1.0)
m.add_boundary_edge(n - 1, weight=1.0)
mid = n // 2
for k in (1000, 1001, 2000, 2001, 4000, 4001, 8000, 8001):
    s = np.zeros(n, np.uint8)
    s[mid - k // 2: mid - k // 2 + k] = 1
    m.decode(s)
    reps = 1 if k > 4001 else 5
    t = time.perf_counter()
    for _ in range(reps):
        m.decode(s)
    dt = (time.perf_counter() - t) / reps
    print(f"block k={k:5d} ({'odd ' if k % 2 else 'even'}): {dt * 1e3:8.2f} ms", flush=True)
