"""Small, independent checks of the research invariants (no neural network)."""

from dataclasses import replace
import json

import numpy as np
import pytest
import stim

from baseline import (build_matching, decode_detectors, evaluate_saved_split,
                      logical_metrics)
from config import ExperimentConfig
from data_pipeline import (build_circuit, load_dataset, make_metadata,
                           sample_labeled_data, save_dataset, split_indices)
from run_experiment import run_experiment


def test_labels_come_directly_from_stim_and_repeat_with_seed():
    config = ExperimentConfig(shots=256, physical_error_rate=0.02)
    circuit = build_circuit(config)
    X, y = sample_labeled_data(circuit, config.shots, config.seed)
    direct_X, direct_y = circuit.compile_detector_sampler(seed=config.seed).sample(
        shots=config.shots, separate_observables=True)
    np.testing.assert_array_equal(X, direct_X)
    np.testing.assert_array_equal(y, direct_y)
    assert X.shape == (256, 24)
    assert y.shape == (256, 1)
    assert X.dtype == y.dtype == np.bool_


def test_stim_produces_known_logical_label_without_any_decoder():
    # A certain stochastic X fault flips a Z measurement relative to the
    # noiseless reference. Both detector and logical parity are known to be 1.
    circuit = stim.Circuit("""
        R 0
        X_ERROR(1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    X, y = sample_labeled_data(circuit, shots=8, seed=42)
    assert X.all()
    assert y.all()


@pytest.mark.parametrize("p", [0.0, 0.01])
def test_real_decoder_accepts_same_samples_and_matches_single_shot_api(p):
    config = ExperimentConfig(shots=64, physical_error_rate=p)
    circuit = build_circuit(config)
    X, y = sample_labeled_data(circuit, config.shots, config.seed)
    matching, _ = build_matching(circuit)
    X_before, y_before = X.copy(), y.copy()
    predicted = decode_detectors(matching, X, circuit.num_observables)
    individual = np.array([matching.decode(row) for row in X])
    np.testing.assert_array_equal(predicted, individual)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    assert predicted.shape == y.shape
    if p == 0:
        assert not X.any() and not y.any()
        assert logical_metrics(y, predicted)["logical_failures"] == 0


def test_shape_normalization_prevents_broadcasting_and_counts_failed_shots():
    y = np.array([[0], [1], [1]])
    predicted = np.array([0, 0, 1])
    report = logical_metrics(y, predicted)
    assert report["logical_failures"] == 1
    assert report["logical_error_rate"] == pytest.approx(1 / 3)
    # A two-bit mismatch in ONE shot is still only ONE logical failure.
    assert logical_metrics(np.zeros((3, 2)),
                           np.array([[1, 1], [0, 0], [0, 1]]))["logical_failures"] == 2
    with pytest.raises(ValueError):
        logical_metrics(y, np.zeros((3, 2)))
    with pytest.raises(ValueError):
        logical_metrics(y, np.zeros((1, 3)))


def test_splits_are_reproducible_disjoint_and_exhaustive():
    config = ExperimentConfig()
    splits = split_indices(config)
    assert [len(ids) for ids in splits.values()] == [7000, 1500, 1500]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))),
                                  np.arange(config.shots))
    for name, ids in splits.items():
        np.testing.assert_array_equal(ids, split_indices(config)[name])
    changed = split_indices(replace(config, split_seed=44))
    assert not np.array_equal(splits["train"], changed["train"])


def test_saved_splits_preserve_exact_row_label_pairs(tmp_path):
    config = ExperimentConfig(shots=100)
    circuit = build_circuit(config)
    X, y = sample_labeled_data(circuit, config.shots, config.seed)
    metadata = make_metadata(config, circuit)
    for name, ids in split_indices(config).items():
        path = tmp_path / f"{name}.npz"
        save_dataset(path, X[ids], y[ids], ids, {**metadata, "split": name})
        saved_X, saved_y, saved_ids, saved_metadata = load_dataset(path)
        np.testing.assert_array_equal(saved_X, X[saved_ids])
        np.testing.assert_array_equal(saved_y, y[saved_ids])
        np.testing.assert_array_equal(ids, saved_ids)
        assert saved_metadata["num_samples"] == len(ids)
        assert saved_metadata["circuit_type"] == config.circuit_type
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            save_dataset(path, X[ids], y[ids], ids, metadata)
        assert path.read_bytes() == before


def test_end_to_end_reserves_test_set_and_supports_later_evaluation(tmp_path):
    config = ExperimentConfig(shots=200, physical_error_rate=0.01)
    output = tmp_path / "experiment"
    report = run_experiment(config, output)
    assert report["num_samples"] == 30
    assert report["split"] == "validation"
    assert not (output / "mwpm_test_metrics.json").exists()
    with np.load(output / "mwpm_validation_predictions.npz", allow_pickle=False) as data:
        X, y, ids, _ = load_dataset(output / "validation.npz")
        np.testing.assert_array_equal(ids, data["sample_ids"])
        assert logical_metrics(y, data["predicted_obs"])["logical_failures"] == report["logical_failures"]
    # Exercise final evaluation only on this disposable test fixture.
    before = (output / "test.npz").read_bytes()
    final = evaluate_saved_split(output, "test")
    assert final["split"] == "test"
    assert before == (output / "test.npz").read_bytes()
    assert json.loads((output / "metadata.json").read_text())["split_counts"] == {
        "train": 140, "validation": 30, "test": 30}
    with pytest.raises(FileExistsError):
        run_experiment(config, output)
    # Prevent accidentally pairing a dataset with a different circuit.
    (output / "circuit.stim").write_text(str(build_circuit(replace(config, distance=5))))
    with pytest.raises(ValueError, match="does not match"):
        evaluate_saved_split(output)


@pytest.mark.parametrize("changes", [
    {"physical_error_rate": -0.1}, {"physical_error_rate": float("nan")},
    {"shots": 2}, {"rounds": 0}, {"distance": 1}, {"seed": -1},
    {"test_fraction": 0.5},
])
def test_invalid_configuration_fails_early(changes):
    with pytest.raises(ValueError):
        ExperimentConfig(**changes)
