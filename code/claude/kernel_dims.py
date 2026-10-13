"""GF(2) kernel dimensions of the true signature matrix H', the graph matrix H_G, and the
decomposition map D (H' = H_G D, l' = l_G D), for the d=3 circuit-level DEM.

Run from code/:  .venv/bin/python claude/kernel_dims.py
See research_log/mwpm_gap_derivation.md section 3.
"""
import numpy as np

from exact_decoding import surface_circuit


def rank2(M):
    M = M.copy() % 2
    r = 0
    for c in range(M.shape[1]):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        M[[r, r + piv[0]]] = M[[r + piv[0], r]]
        rows = np.nonzero(M[:, c])[0]
        rows = rows[rows != r]
        M[rows] ^= M[r]
        r += 1
        if r == M.shape[0]:
            break
    return r


def dem_lines(dem):
    """Each error line as a list of pieces (frozenset of detectors, L bit); undecomposed lines have one piece."""
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        pieces = [[set(), 0]]
        for t in inst.targets_copy():
            if t.is_separator():
                pieces.append([set(), 0])
            elif t.is_relative_detector_id():
                pieces[-1][0] ^= {t.val}
            elif t.is_logical_observable_id():
                pieces[-1][1] ^= 1
        out.append([(frozenset(d), L) for d, L in pieces])
    return out


def report(name, H, n):
    m = H.shape[1]
    print(f"  {name:<34} cols={m:4d}  dim ker H={m - rank2(H[:n]):4d}  dim ker [H; l]={m - rank2(H):4d}")


if __name__ == "__main__":
    for rounds in (1, 3):
        c = surface_circuit(1e-3, rounds=rounds)
        undec = c.detector_error_model(decompose_errors=False)
        n = undec.num_detectors
        full = dem_lines(undec)
        dec = dem_lines(c.detector_error_model(decompose_errors=True))

        def vec(d, L):
            v = np.zeros(n + 1, np.uint8)
            v[list(d)] = 1
            v[n] = L
            return v

        edges = sorted({p for line in dec for p in line}, key=lambda e: (sorted(e[0]), e[1]))
        eidx = {e: i for i, e in enumerate(edges)}
        HG = np.array([vec(*e) for e in edges]).T
        D = np.zeros((len(edges), len(dec)), np.uint8)
        for j, line in enumerate(dec):
            for p in line:
                D[eidx[p], j] ^= 1
        Hd = (HG @ D) % 2  # true signature of each decomposed line (rows: detectors, then L)
        for j, line in enumerate(dec):
            sig = np.zeros(n + 1, np.uint8)
            for p in line:
                sig ^= vec(*p)
            assert (Hd[:, j] == sig).all()

        print(f"rounds={rounds}  detectors={n}")
        report("H  (undecomposed DEM)", np.array([vec(*line[0]) for line in full]).T, n)
        report("H' (decomposed lines, full sig)", Hd, n)
        report("H_G (graph edges)", HG, n)
        rD = rank2(D)
        print(f"  D: rank {rD} of {len(edges)} edges (surjective: {rD == len(edges)}), dim ker D = {len(dec) - rD}")
