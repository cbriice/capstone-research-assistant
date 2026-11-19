import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import stim
import torch

ALL_COLUMNS = ("event", "x", "y", "t", "is_z", "is_final", "is_first")                          #make_tokens builds these, then the sin/cos columns


@lru_cache(maxsize=None)                                                                        #remembers the answer per run folder
def read_manifest(run_dir: str) -> dict:                                                        #so manifest.json is read once, not once per r
    return json.loads((Path(run_dir) / "manifest.json").read_text())                            #["config"] = the config the data was made with


def feature_columns(features: list[str], periods: list[float]) -> list[str]:                    #token column names after the event
    cols = []
    for f in features:
        if f == "sincos":                                                                       #2 columns per period
            cols += [f"{fn}_t{p:g}" for p in periods for fn in ("sin", "cos")]
        else:
            cols.append(f)
    return cols


def load_syndromes(run_dir, split: str, r: int) -> tuple[np.ndarray, np.ndarray, stim.Circuit]:
    run_dir = Path(run_dir)
    circuit = stim.Circuit.from_file(run_dir / "circuits" / f"r{r:03d}.stim")
    with np.load(run_dir / split / f"r{r:03d}.npz") as f:                                       #with closes the file when done
        dets, obs, num_detectors = f["dets"], f["obs"], int(f["num_detectors"])
    if num_detectors != circuit.num_detectors or dets.shape[1] != num_detectors:                #if instead of assert: python -O removes asserts
        raise ValueError(f"{split}/r{r:03d}.npz does not match circuits/r{r:03d}.stim")
    return dets.astype(np.uint8, copy=False), obs[:, 0].astype(np.uint8), circuit               #uint8 0/1 [shots, stabilizers], [shots]


def load_tokens(run_dir, split: str, r: int, tokens_cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    basis = read_manifest(str(run_dir))["config"]["circuit"]["basis"]                           #all tokens of one file at once: fine for a look,
    dets, labels, circuit = load_syndromes(run_dir, split, r)                                   #training builds them per batch (training_data.py)
    tokens = make_tokens(circuit, dets.astype(bool), basis, tokens_cfg["features"], tokens_cfg["sincos_periods"])
    return torch.from_numpy(tokens), torch.from_numpy(labels).float()


def static_features(circuit: stim.Circuit, basis: str, features: list[str], periods: list[float]) -> np.ndarray:
    empty = np.zeros((1, circuit.num_detectors), dtype=bool)                                    #one shot where nothing fired
    return make_tokens(circuit, empty, basis, features, periods)[0, :, 1:]                      #[stabilizers, columns]: same in every shot


def create_position_features(circuit: stim.Circuit, syndrome: np.ndarray) -> np.ndarray:
    stab_coords = circuit.get_detector_coordinates()                                            #gets stabilizer coordinates from stim circuit
    xyt = np.array([stab_coords[k] for k in range(circuit.num_detectors)], dtype=np.float32)    #creates numpy array of coords
    two_d = max(max(c[0], c[1]) for c in circuit.get_final_qubit_coordinates().values())        #gets maximum coord position
    x = xyt[:, 0] / two_d - 0.5                                                                 #normalizes x to [0,1] and centers on 0
    y = xyt[:, 1] / two_d - 0.5                                                                 #normalizes y to [0,1] and centers on 0
    t = xyt[:, 2] / xyt[:, 2].max()                                                             #round index / r: t in [0,1] (relative time)
    position = np.stack([x, y, t], axis=1)                                                      #combines x,y,t on columns
    shots = syndrome.shape[0]                                                                   #gets number of shots
    return np.concatenate([syndrome[:, :, None].astype(np.float32),                             #adds axis of length 1 to syndrome
                                                                                                #creating [shots, stabilizers, 1]
                           np.broadcast_to(position, (shots, *position.shape))], axis=2)        #(shots, *position.shape) unpacks position tuple (stabilizers,features)
                                                                                                #into tuple (syndrome.shape[0], position.shape[0], position.shape[1])
                                                                                                #broadcast_to then attaches the position to every shot
                                                                                                #np.broadcast_to(array, shape), axis=2 concatenates them along the right axis


def add_stabilizer_type(circuit: stim.Circuit, tokens: np.ndarray, basis: str = "z") -> np.ndarray:
    coords = circuit.get_detector_coordinates()                                                 #gets stabilizer coordinates from stim circuit
    xyt = np.array([coords[k] for k in range(circuit.num_detectors)])                           #creates numpy array of coords
    first_slice = {(x, y) for x, y, t in xyt if t == 0}                                         #gets the coords of the round (no x-readout in z-basis)
    is_basis = np.array([(x, y) in first_slice for x, y, _ in xyt])                             #gives boolean 1 if coords match initial readout coords {0 otherwise}
    is_z = is_basis if basis == "z" else ~is_basis                                              #if x basis invert all 1s and 0s
    return _append(tokens, is_z)


def add_final_round(circuit: stim.Circuit, tokens: np.ndarray) -> np.ndarray:
    coords = circuit.get_detector_coordinates()                                                 #gets stabilizer coordinates from stim circuit
    t = np.array([coords[k][2] for k in range(circuit.num_detectors)])                          #creates and array of t
    return _append(tokens, t == t.max())                                                        #appends array of if t == t.max()


def add_first_round(circuit: stim.Circuit, tokens: np.ndarray) -> np.ndarray:
    coords = circuit.get_detector_coordinates()
    t = np.array([coords[k][2] for k in range(circuit.num_detectors)])
    return _append(tokens, t == 0)                                                              #1 on the first time slice


def add_sincos_time(circuit: stim.Circuit, tokens: np.ndarray, periods: list[float]) -> np.ndarray:
    coords = circuit.get_detector_coordinates()
    k = np.array([coords[i][2] for i in range(circuit.num_detectors)])                          #round index, not divided by r (absolute time)
    for p in periods:
        tokens = _append(tokens, np.sin(2 * np.pi * k / p))                                     #one full turn every p rounds:
        tokens = _append(tokens, np.cos(2 * np.pi * k / p))                                     #sin and cos together tell every round of a turn apart
    return tokens


def _append(tokens: np.ndarray, column: np.ndarray) -> np.ndarray:
    col = np.broadcast_to(column.astype(np.float32)[None, :, None]                              #turns column into a (1, stabilizer, 1) array
                          , (tokens.shape[0], tokens.shape[1], 1))                              #sets broacast size to (shots, stabilizers, 1)
    return np.concatenate([tokens, col], axis=2)                                                #concatenates tokens and column along right axis


def make_tokens(circuit: stim.Circuit, dets: np.ndarray, basis: str = "z",
                features: list[str] = ("x", "y", "t", "is_z", "is_final"),
                periods: list[float] = (8, 64)) -> np.ndarray:
    tokens = create_position_features(circuit, dets)                                            #event, x, y, t
    tokens = add_stabilizer_type(circuit, tokens, basis)                                        #is_z
    tokens = add_final_round(circuit, tokens)                                                   #is_final
    tokens = add_first_round(circuit, tokens)                                                   #is_first
    tokens = add_sincos_time(circuit, tokens, periods)                                          #sin_t8, cos_t8, sin_t64, cos_t64, ...
    names = [*ALL_COLUMNS, *feature_columns(["sincos"], periods)]
    keep = [names.index(c) for c in ["event", *feature_columns(features, periods)]]             #the columns asked for, in token order
    return np.ascontiguousarray(tokens[:, :, keep])                                             #[shots, stabilizers, 1 + columns]
