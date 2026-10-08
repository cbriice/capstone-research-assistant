"""Exact (sampling-free) decoder evaluation for small Stim circuits.

Builds the full joint distribution P(syndrome, logical) from a detector error model by
XOR-convolving every error mechanism, then reads off:
  - the optimal (maximum-likelihood) decoder's logical error rate: sum_s min(P(s,0), P(s,1))
  - any matcher's exact logical error rate: sum_s P(s, L != prediction(s))

Feasible only while num_detectors + 1 <= ~30 bits (memory is 8 * 2^bits bytes):
d=3 surface code with rounds <= 3 (24 detectors) uses ~1 GB peak and takes ~15 s.
Assumes a single logical observable.
"""
import numpy as np
import pymatching
import stim


def surface_circuit(p: float, distance: int = 3, rounds: int = 3) -> stim.Circuit:
    """Same noise convention as data_pipe.ipynb: one p on all four built-in channels."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def exact_joint(dem: stim.DetectorErrorModel) -> tuple[np.ndarray, int]:
    """Return (P, n) with P[L, s] = probability of logical flip L and syndrome s (as an int bitmask)."""
    dem = dem.flattened()
    n = dem.num_detectors
    if dem.num_observables != 1:
        raise ValueError("expects exactly one logical observable")
    if n + 1 > 30:
        raise ValueError(f"{n + 1} bits is too large to enumerate")
    P = np.zeros(2 ** (n + 1))
    P[0] = 1.0
    idx = np.arange(2 ** (n + 1), dtype=np.uint32)
    for inst in dem:
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        mask = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                mask ^= 1 << (n + t.val)
        # this mechanism either didn't fire (keep P) or fired (shift every outcome by its signature)
        P = (1 - p) * P + p * P[idx ^ mask]
    return P.reshape(2, 2**n), n


def optimal_ler(P: np.ndarray) -> float:
    return float(np.minimum(P[0], P[1]).sum())


def mwpm_ler(P: np.ndarray, n: int, decoder_dem: stim.DetectorErrorModel, min_prob: float = 1e-14) -> float:
    """Exact LER of PyMatching built from decoder_dem, on data distributed as P.

    Syndromes with P(s) < min_prob are skipped; the skipped mass is negligible at d=3."""
    keep = np.nonzero(P.sum(0) > min_prob)[0]
    bits = ((keep[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    pred = pymatching.Matching.from_detector_error_model(decoder_dem).decode_batch(bits)[:, 0]
    return float(P[1 - pred, keep].sum())


def per_qubit_noise(circ: stim.Circuit, mult) -> stim.Circuit:
    """Rescale every noise channel by a per-qubit multiplier mult[q].

    Two-qubit channels use the mean multiplier of the pair."""
    out = stim.Circuit()
    for inst in circ.flattened():
        gd = stim.gate_data(inst.name)
        if not gd.is_noisy_gate or gd.produces_measurements:
            out.append(inst)
            continue
        p = inst.gate_args_copy()[0]
        ts = inst.targets_copy()
        k = 2 if gd.is_two_qubit_gate else 1
        for i in range(0, len(ts), k):
            grp = ts[i : i + k]
            m = np.mean([mult[t.value] for t in grp])
            out.append(inst.name, grp, min(p * m, 0.5 if k == 1 else 15 / 16))
    return out


def with_crosstalk(circ: stim.Circuit, p_ct: float) -> stim.Circuit:
    """Add correlated ZZ and XX errors (each with prob p_ct) on every nearest-neighbour data-qubit pair,
    once per round, right after the before-round data depolarization.

    This is an invented toy model, not a calibrated device model."""
    coords = circ.get_final_qubit_coordinates()
    data = sorted(q for q, (x, y) in coords.items() if x % 2 == 1 and y % 2 == 1)
    pairs = [
        (a, b)
        for a in data
        for b in data
        if a < b and abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) == 2
    ]
    out = stim.Circuit()
    for inst in circ.flattened():
        out.append(inst)
        if inst.name == "DEPOLARIZE1" and sorted(t.value for t in inst.targets_copy()) == data:
            for a, b in pairs:
                out.append("CORRELATED_ERROR", [stim.target_z(a), stim.target_z(b)], p_ct)
                out.append("CORRELATED_ERROR", [stim.target_x(a), stim.target_x(b)], p_ct)
    return out
