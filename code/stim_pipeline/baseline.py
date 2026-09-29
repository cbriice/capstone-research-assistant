"""MWPM predictions and logical failure metrics, independent of label generation."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pymatching
import stim

from data_pipeline import load_dataset


def build_matching(circuit: stim.Circuit):
    """Derive the graph from the exact circuit that generated the dataset."""
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    # At p=0 there may be no error edges carrying observable IDs. Preserve
    # all observable output columns even in that useful noiseless sanity check.
    matching.ensure_num_fault_ids(circuit.num_observables)
    return matching, dem


def observable_matrix(values: np.ndarray) -> np.ndarray:
    """Accept (shots,) for one observable, otherwise require (shots, observables)."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Expected a nonempty observable axis")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Observable arrays must contain only 0/1")
    return values


def decode_detectors(matching, X: np.ndarray, num_observables: int):
    """Only detector events enter the decoder; it never receives ground truth."""
    if X.ndim != 2 or X.shape[1] != matching.num_detectors:
        raise ValueError("X must have shape (shots, matching.num_detectors)")
    predicted = observable_matrix(matching.decode_batch(X))
    if predicted.shape != (len(X), num_observables):
        raise ValueError(f"Unexpected prediction shape: {predicted.shape}")
    return predicted


def logical_metrics(y: np.ndarray, predicted: np.ndarray):
    """A shot fails when ANY predicted logical bit differs from Stim's label.

    Check shapes BEFORE comparison: (N, 1) != (N,) otherwise broadcasts
    to (N, N), silently producing a completely incorrect failure count.
    The rate here is per complete memory experiment, not per QEC round.
    """
    y, predicted = observable_matrix(y), observable_matrix(predicted)
    if y.shape != predicted.shape or len(y) == 0:
        raise ValueError("Labels and predictions must have the same nonempty shape")
    failed = np.any(predicted != y, axis=1)
    failures = int(np.count_nonzero(failed))
    return {
        "num_samples": len(y),
        "num_observables": y.shape[1],
        "logical_failures": failures,
        "logical_error_rate": failures / len(y),
        "logical_accuracy": 1 - failures / len(y),
        "rate_unit": "per complete memory experiment (shot)",
    }


def evaluate_saved_split(data_dir: Path, split: str = "validation"):
    """Decode the saved rows directly, without resampling or editing the dataset."""
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    data_dir = Path(data_dir)
    X, y, sample_ids, metadata = load_dataset(data_dir / f"{split}.npz")
    circuit_text = (data_dir / "circuit.stim").read_text()
    circuit_hash = hashlib.sha256(circuit_text.encode()).hexdigest()
    if circuit_hash != metadata["circuit_sha256"]:
        raise ValueError("The saved circuit does not match this dataset")
    circuit = stim.Circuit(circuit_text)
    if y.shape != (len(X), circuit.num_observables):
        raise ValueError("Saved labels do not match the circuit's observables")
    matching, _ = build_matching(circuit)
    predicted = decode_detectors(matching, X, circuit.num_observables)
    report = {
        **logical_metrics(y, predicted),
        "split": split,
        "decoder": "PyMatching MWPM (default uncorrelated matching)",
        "circuit_sha256": circuit_hash,
        "versions": {name: version(name) for name in ("stim", "pymatching", "numpy")},
    }
    # Results are separate artifacts: never replace y with decoder predictions.
    np.savez_compressed(data_dir / f"mwpm_{split}_predictions.npz",
                        predicted_obs=predicted, sample_ids=sample_ids)
    (data_dir / f"mwpm_{split}_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation", help="Use test only for the final comparison")
    args = parser.parse_args()
    print(json.dumps(evaluate_saved_split(args.data_dir, args.split), indent=2))


if __name__ == "__main__":
    main()
