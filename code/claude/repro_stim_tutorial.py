# Reproduce Stim getting_started §9 (PyMatching, rotated_memory_z, uniform p on 4 channels, r = 3d)
import numpy as np, stim
from pymatching import Matching

def circ(d, r, p):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=r,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)

def run(d, r, p, max_shots, max_err):
    c = circ(d, r, p)
    m = Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    s = c.compile_detector_sampler(seed=1)
    shots = fails = 0
    while shots < max_shots and fails < max_err:
        det, obs = s.sample(100_000, separate_observables=True)
        fails += int(np.sum(m.decode_batch(det)[:, 0] != obs[:, 0])); shots += 100_000
    ler = fails / shots
    per_round = 0.5 * (1 - (1 - 2 * ler) ** (1 / r))   # same as sinter.shot_error_rate_to_piece_error_rate
    return per_round, fails, shots

print("threshold sweep, per-round eps (r=3d):")
for p in [0.008, 0.009, 0.010, 0.011, 0.012]:
    row = [run(d, 3 * d, p, 1_000_000, 5_000)[0] for d in (3, 5, 7)]
    print(f"  p={p:.3f}  d3={row[0]:.3e}  d5={row[1]:.3e}  d7={row[2]:.3e}")
print("p=1e-3, r=3d:")
for d in (3, 5):
    e, f, n = run(d, 3 * d, 1e-3, 5_000_000, 1_000)
    print(f"  d={d}  eps={e:.3e}  ({f} fails / {n} shots)")
print("tutorial fit ln(eps) = -4.615 - 1.1896 d  ->  d3 =", f"{np.exp(-4.615-1.1896*3):.3e}", " d5 =", f"{np.exp(-4.615-1.1896*5):.3e}")
