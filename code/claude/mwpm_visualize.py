"""3D step-through of blossom MWPM on a d=3, r=5 surface-code decoding graph.

Uses the Fusion Blossom package (`pip install fusion-blossom`) and its built-in
three.js viewer. Axes are (i, j) = stabilizer position, t = measurement round.
Every snapshot is a real solver state: dual radii growing, edges going tight,
alternating trees, blossoms, and the final matching.

Writes two files:
  blossom.json  - SolverSerial: the classic primal-dual blossom on the whole graph
  fusion.json   - SolverParallel: graph cut into per-round blocks, each solved
                  alone, then fused with a linear (streaming-style) fusion tree
micro_*.json come from micro_blossom_viz.rs (Rust, see its header); the same server shows them.

Noise model is phenomenological (data + measurement errors, uniform p), not the
full Stim circuit model. That's enough to see how the algorithm moves.

Usage:
  code/.venv/bin/python code/claude/mwpm_visualize.py            # generate + open browser
  code/.venv/bin/python code/claude/mwpm_visualize.py --no-serve # just write the JSON
Viewer: the snapshot slider in the right panel steps through the solver; drag to rotate.
"""

import argparse
import os

import fusion_blossom as fb

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz_data")


def build_code(d, rounds, p):
    # noisy_measurements = rounds - 1 gives `rounds` layers of detectors
    return fb.PhenomenologicalRotatedCode(d=d, noisy_measurements=rounds - 1, p=p, max_half_weight=500)


def run_serial(code, syndrome, path):
    vis = fb.Visualizer(filepath=path, positions=code.get_positions())
    solver = fb.SolverSerial(code.get_initializer())
    solver.solve(syndrome, vis)
    return solver.subgraph(vis), solver.sum_dual_variables()


def layer_partition(code, rounds):
    """Even layers are blocks, odd layers are the fusion interfaces between them.

    rounds=5: blocks L0, L2, L4; fuse (L0,L2) over L1, then (that, L4) over L3.
    The linear tree is the streaming shape: each new block fuses onto the past.
    """
    init = code.get_initializer()
    per_layer = init.vertex_num // rounds  # vertices are ordered by t
    pc = fb.PartitionConfig(init.vertex_num)
    blocks = list(range(0, rounds, 2))
    pc.partitions = [fb.VertexRange(t * per_layer, (t + 1) * per_layer) for t in blocks]
    fusions, acc = [], 0
    for k in range(1, len(blocks)):
        fusions.append((acc, k))
        acc = len(blocks) + k - 1  # index of the unit this fusion creates
    pc.fusions = fusions
    return pc.info()


def run_fusion(code, syndrome, rounds, path):
    cfg = {
        "dual": {"thread_pool_size": 1, "enable_parallel_execution": False},
        "primal": {"thread_pool_size": 1, "debug_sequential": True, "pin_threads_to_cores": False},
    }
    vis = fb.Visualizer(filepath=path, positions=code.get_positions())
    solver = fb.SolverParallel(code.get_initializer(), layer_partition(code, rounds), cfg)
    solver.solve(syndrome, vis)
    return solver.subgraph(vis), solver.sum_dual_variables()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.03)
    # seed 292 at d=3, r=5, p=0.03: 8 defects, two blossoms form
    ap.add_argument("--seed", type=int, default=292)
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    code = build_code(args.d, args.rounds, args.p)
    syndrome = code.generate_random_errors(seed=args.seed)
    print(f"defects (vertex ids): {syndrome.defect_vertices}")

    for name, run in [
        ("blossom", lambda f: run_serial(code, syndrome, f)),
        ("fusion", lambda f: run_fusion(code, syndrome, args.rounds, f)),
    ]:
        sub, dual_sum = run(os.path.join(OUT_DIR, f"{name}.json"))
        # sum of radii = matching weight (LP duality); both solvers must agree
        print(f"{name:8s} correction edges {sub}, sum of radii {dual_sum}")

    if not args.no_serve:
        port = 51666
        print(f"blossom: http://localhost:{port}/?filename=blossom.json")
        print(f"fusion:  http://localhost:{port}/?filename=fusion.json")
        for f in sorted(os.listdir(OUT_DIR)):
            if f.startswith("micro_"):
                print(f"micro:   http://localhost:{port}/?filename={f}")
        print("press Enter here to stop the server")
        fb.helper.open_visualizer("blossom.json", port=port, data_folder=OUT_DIR, open_browser=True)


if __name__ == "__main__":
    main()
