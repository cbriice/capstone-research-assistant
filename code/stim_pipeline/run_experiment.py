"""Run the first-stage experiment: Stim labels, saved splits, validation MWPM."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from baseline import build_matching, evaluate_saved_split
from config import ExperimentConfig
from data_pipeline import (build_circuit, make_metadata, sample_labeled_data,
                           save_dataset, split_indices)


def run_experiment(config: ExperimentConfig, output_dir: Path = Path("data")):
    output_dir = Path(output_dir)
    # Choose another output directory for each experiment. Never silently
    # regenerate a held-out test set or mix files from different circuits.
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a new --output-dir")

    circuit = build_circuit(config)
    _, dem = build_matching(circuit)
    X, y = sample_labeled_data(circuit, config.shots, config.seed)
    indices = split_indices(config)
    metadata = make_metadata(config, circuit)

    print(f"Circuit: {config.circuit_type}")
    print(f"Distance: {config.distance}; rounds: {config.rounds}")
    print(f"Physical error rate: {config.physical_error_rate}")
    print(f"Detectors: {circuit.num_detectors}; observables: {circuit.num_observables}")
    print(f"Generated shots: {config.shots}; Stim seed: {config.seed}")
    print(f"X.shape = {X.shape}; y.shape = {y.shape}")
    print("Labels are supplied by Stim, before any decoder predictions.")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "circuit.stim").write_text(str(circuit))
    (output_dir / "detector_error_model.dem").write_text(str(dem))
    metadata["split_counts"] = {name: len(ids) for name, ids in indices.items()}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, ids in indices.items():
        save_dataset(output_dir / f"{name}.npz", X[ids], y[ids], ids,
                     {**metadata, "split": name})
        print(f"Saved {name}: {len(ids)} shots")

    # Show only training examples, so inspecting the run does not expose test labels.
    examples = list(indices["train"][:3])
    flipped = indices["train"][np.any(y[indices["train"]], axis=1)]
    if len(flipped) and flipped[0] not in examples:
        examples.append(flipped[0])
    print("\nTraining examples (includes a logical flip if one was sampled):")
    for shot_id in examples:
        print(f"Shot {shot_id}")
        print(f"  Detectors: {X[shot_id].astype(np.uint8)}")
        print(f"  Logical label (Stim): {y[shot_id].astype(np.uint8)}")

    # Reopen validation.npz: MWPM sees the very same rows future ML can use.
    report = evaluate_saved_split(output_dir, "validation")
    print(f"\nMWPM validation predictions.shape = "
          f"({report['num_samples']}, {report['num_observables']})")
    print(f"Logical failures: {report['logical_failures']} / {report['num_samples']}")
    print(f"Logical error rate per shot: {report['logical_error_rate']:.6g}")
    print(f"Logical accuracy: {report['logical_accuracy']:.6%}")
    print("Test data saved and reserved; no test evaluation performed.")
    return report


def main():
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=defaults.distance)
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--physical-error-rate", type=float, default=defaults.physical_error_rate)
    parser.add_argument("--shots", type=int, default=defaults.shots)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = replace(defaults, distance=args.distance, rounds=args.rounds,
                     physical_error_rate=args.physical_error_rate,
                     shots=args.shots, seed=args.seed, split_seed=args.split_seed)
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
