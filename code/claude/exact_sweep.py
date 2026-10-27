"""Exact per-shot ML (sum) or most-likely-error (max) decoding by sweeping the DEM in time order.

exact_decoding.py builds the table over all 2^(n+1) (syndrome, L) outcomes, so it is exponential in
n = r(d^2-1) and stops at d=3, r=3. Here the syndrome is fixed, so a detector can be checked against s
and dropped as soon as the last mechanism touching it has been folded in. The state is only the
parity of the currently open detectors plus L: 2^(w+1) entries, where the frontier width w does not
grow with r (d=3: 11, d=5: 28, d=7: 53). Cost O(M * 2^(w+1)) per shot, so d=3 is exact at any r.

Run from code/:  .venv/bin/python claude/exact_sweep.py   (~5 min)
See research_log/decoder_complexity.md section 4.
"""
import sys
import time

import numpy as np
import pymatching

sys.path.insert(0, "claude")
from exact_decoding import exact_joint, surface_circuit


def frontier_order(dem):
    """Mechanisms as (p, detectors, flips_L), sorted by time; last[x] = index of the last mechanism touching x."""
    mechs = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        L = int(any(t.is_logical_observable_id() for t in inst.targets_copy()))
        mechs.append((inst.args_copy()[0], ds, L))
    mechs.sort(key=lambda m: (min(m[1], default=-1), max(m[1], default=-1)))
    last = {x: i for i, (_, ds, _) in enumerate(mechs) for x in ds}
    return mechs, last


def sweep_decode(dem, dets, rule="sum", W=12):
    """Predicted L for each row of dets. rule='sum' is ML (optimal), 'max' is most-likely error. W >= frontier width."""
    mechs, last = frontier_order(dem)
    size = 1 << (W + 1)
    idx = np.arange(size)
    P = np.zeros((dets.shape[0], size))
    P[:, 0] = 1.0
    slot, free = {}, list(range(W))[::-1]
    for i, (p, ds, L) in enumerate(mechs):
        for x in ds:
            if x not in slot:
                slot[x] = free.pop()
        mask = sum(1 << slot[x] for x in ds) | (L << W)
        Q = P[:, idx ^ mask]
        P = (1 - p) * P + p * Q if rule == "sum" else np.maximum((1 - p) * P, p * Q)
        for x in ds:
            if last[x] == i:  # detector x is final: keep only states whose parity matches s[x], then free its slot
                k = slot.pop(x)
                free.append(k)
                lo = idx[(idx >> k) & 1 == 0]
                hi = lo | (1 << k)
                P[:, lo] = np.where(dets[:, [x]].astype(bool), P[:, hi], P[:, lo])
                P[:, hi] = 0
        P /= P.max(1, keepdims=True)
    return (P[:, 1 << W] > P[:, 0]).astype(np.uint8)


def batched(dem, dets, chunk=250, **kw):
    return np.concatenate([sweep_decode(dem, dets[i:i + chunk], **kw) for i in range(0, len(dets), chunk)])


if __name__ == "__main__":
    c = surface_circuit(1e-3, 3, 3)
    dem = c.detector_error_model()
    P, n = exact_joint(dem)
    det, _ = c.compile_detector_sampler(seed=7).sample(2000, separate_observables=True)
    s = (det.astype(np.int64) << np.arange(n)).sum(1)
    table = (P[1, s] > P[0, s]).astype(np.uint8)
    t = time.perf_counter()
    sw = batched(dem, det)
    print(f"r=3 check vs full joint table: {np.sum(sw != table)} / {len(sw)} disagree; "
          f"{(time.perf_counter() - t) / len(sw) * 1e3:.1f} ms/shot")

    for p, r, shots in [(3e-3, 25, 3000)]:
        c = surface_circuit(p, 3, r)
        dem = c.detector_error_model()
        det, obs = c.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
        t = time.perf_counter()
        ml = batched(dem, det)
        dt = (time.perf_counter() - t) / shots
        mw = pymatching.Matching.from_detector_error_model(
            c.detector_error_model(decompose_errors=True)).decode_batch(det)[:, 0]
        f_ml, f_mw = np.sum(ml != obs[:, 0]), np.sum(mw != obs[:, 0])
        print(f"p={p} r={r}: {dt * 1e3:.0f} ms/shot; failures ML {f_ml}, MWPM {f_mw} of {shots}")
