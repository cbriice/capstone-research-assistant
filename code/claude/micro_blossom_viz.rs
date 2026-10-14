// Micro Blossom visualizer runner. Pairs with mwpm_visualize.py; writes micro_*.json into viz_data/.
// Not built from this folder. To rebuild:
//   git clone --depth 1 https://github.com/yuewuo/micro-blossom.git && cd micro-blossom/src/cpu/blossom
//   in Cargo.toml set crate-type = ["lib"]  (the cdylib needs an embedded print_char symbol and fails to link on macOS)
//   cp <this file> examples/viz_d3r5.rs
//   cargo build --release --example viz_d3r5   (rustup fetches the pinned nightly-2023-11-16 itself)
//   ./target/release/examples/viz_d3r5 <repo>/code/claude/viz_data
// Uses the comb dual module: a Rust behavioral model of the FPGA logic, not real hardware.
// d=3, r=5 phenomenological rotated code, same syndrome as code/claude/mwpm_visualize.py (seed 292)
// writes micro_<variant>.json for the fusion-blossom viewer
use fusion_blossom::example_codes::*;
use fusion_blossom::mwpm_solver::*;
use fusion_blossom::visualize::*;
use micro_blossom::mwpm_solver::*;
use micro_blossom::resources::*;
use serde_json::json;

fn main() {
    let out = std::env::args().nth(1).expect("usage: viz_d3r5 <out_dir>");
    let dense = vec![4, 11, 12, 15, 23, 28, 31, 39];
    // two isolated measurement-error pairs (same stabilizer, consecutive rounds): pre-matching handles both on "FPGA"
    let sparse = vec![8, 16, 31, 39];
    for (name, defects, offload, layer) in [
        ("micro_plain", &dense, false, false),
        ("micro_stream", &dense, true, true),
        ("micro_prematch", &sparse, true, false),
    ] {
        let mut code = PhenomenologicalRotatedCode::new(3, 4, 0.03, 500);
        let initializer = code.get_initializer();
        code.set_defect_vertices(defects);
        let syndrome = code.get_syndrome();
        let path = format!("{out}/{name}.json");
        let mut vis = Some(Visualizer::new(Some(path.clone()), code.get_positions(), true).unwrap());
        let graph = MicroBlossomSingle::new(&initializer, &code.get_positions());
        let mut solver = SolverEmbeddedComb::new(
            graph,
            json!({"dual": {"sim_config": {"support_offloading": offload, "support_layer_fusion": layer}}}),
        );
        solver.solve_visualizer(&syndrome, vis.as_mut());
        let sub = solver.subgraph_visualizer(vis.as_mut());
        println!("{name}: defects {defects:?} subgraph {sub:?} offloaded {}", solver.offloaded);
    }
}
