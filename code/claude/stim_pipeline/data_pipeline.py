"""Generate Stim labels and persist disjoint datasets without using a decoder."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import stim

from config import ExperimentConfig


def build_circuit(config: ExperimentConfig) -> stim.Circuit:
    """Use the same p at each of the three requested circuit noise locations."""
    p = config.physical_error_rate
    return stim.Circuit.generated(
        config.circuit_type,
        distance=config.distance,
        rounds=config.rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def sample_labeled_data(circuit: stim.Circuit, shots: int, seed: int):
    """Stim supplies BOTH arrays from the same simulated shots.

    X[i] is a detector-event vector. y[i] is the observable flip relative to
    the noiseless reference, before decoding. It is NOT an MWPM prediction
    or the indicator that a decoder failed. Keep the observable axis even
    for a single observable: y has shape (shots, 1), not (shots,).
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    X, y = sampler.sample(shots=shots, separate_observables=True)
    if X.shape != (shots, circuit.num_detectors):
        raise ValueError(f"Unexpected detector shape: {X.shape}")
    if y.shape != (shots, circuit.num_observables):
        raise ValueError(f"Unexpected label shape: {y.shape}")
    return X, y


def split_indices(config: ExperimentConfig):
    """Partition shot IDs once; each sampled shot belongs to exactly one split.

    Identical bit patterns from independent shots are legitimate, especially
    all-zero syndromes. Split by shot identity, not by unique bit patterns.
    Round training/validation counts down and assign the remainder to test.
    """
    ids = np.random.default_rng(config.split_seed).permutation(config.shots)
    train_end = int(config.shots * config.train_fraction)
    validation_end = train_end + int(config.shots * config.validation_fraction)
    return dict(zip(("train", "validation", "test"),
                    np.split(ids, [train_end, validation_end])))


def make_metadata(config: ExperimentConfig, circuit: stim.Circuit):
    """Record settings, software, and the exact circuit's identity for reuse."""
    return {
        **asdict(config),
        "schema_version": 1,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "label_source": "Stim circuit detector sampler, separate_observables=True",
        "noise_model": {
            "after_clifford_depolarization": config.physical_error_rate,
            "before_measure_flip_probability": config.physical_error_rate,
            "after_reset_flip_probability": config.physical_error_rate,
            "before_round_data_depolarization": 0.0,
        },
        "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
        "versions": {name: version(name) for name in ("stim", "pymatching", "numpy")},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sampling": "One sample call for all shots, unpacked boolean arrays",
        "split_method": "NumPy default_rng permutation; floor train/validation, remainder test",
    }


def save_dataset(path: Path, X: np.ndarray, y: np.ndarray,
                 sample_ids: np.ndarray, metadata: dict):
    """Save aligned features/labels plus JSON metadata; never overwrite a dataset.

    Boolean arrays already store 0/1 compactly (one byte per entry before
    compression). JSON is a string array, so loading never needs pickle.
    """
    if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X and y must be 2D arrays with equal numbers of shots")
    if sample_ids.shape != (len(X),) or len(np.unique(sample_ids)) != len(X):
        raise ValueError("Each row must have one unique source shot ID")
    if not all(np.all((a == 0) | (a == 1)) for a in (X, y)):
        raise ValueError("Detector data and labels must contain only 0/1")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "num_samples": len(X)}
    # Exclusive creation protects an existing train/validation/test partition.
    with path.open("xb") as stream:
        np.savez_compressed(stream, X=X.astype(bool), y=y.astype(bool),
                            sample_ids=sample_ids,
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))


def load_dataset(path: Path):
    """Return X, y, original shot IDs, and decoded metadata."""
    with np.load(path, allow_pickle=False) as data:
        return (data["X"], data["y"], data["sample_ids"],
                json.loads(data["metadata"].item()))
