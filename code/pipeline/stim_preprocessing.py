import numpy as np
import stim
from pathlib import Path
import json
 
 
def load_tokens(run_dir, split, r):
    run_dir = Path(run_dir)
    basis = json.loads((run_dir / "manifest.json").read_text())["config"]["circuit"]["basis"]
    circuit = stim.Circuit.from_file(run_dir / "circuits" / f"r{r:03d}.stim")
    f = np.load(run_dir / split / f"r{r:03d}.npz")
    assert int(f["num_detectors"]) == circuit.num_detectors
    dets = f["dets"]
    if dets.shape[1] != circuit.num_detectors:      # older bit-packed runs
        dets = np.unpackbits(dets, axis=1, bitorder="little", count=circuit.num_detectors)
    labels = f["obs"][:, 0]
    return make_tokens(circuit, dets.astype(bool), basis), labels
 
def create_position_features(circuit: stim.Circuit, syndrome: np.ndarray) -> np.ndarray:
    stab_coords = circuit.get_detector_coordinates()                                            #gets stabilizer coordinates from stim circuit
    xyt = np.array([stab_coords[k] for k in range(circuit.num_detectors)], dtype=np.float32)    #creates numpy array of coords 
    two_d = max(max(c[0], c[1]) for c in circuit.get_final_qubit_coordinates().values())        #gets maximum coord position
    x = xyt[:, 0] / two_d - 0.5                                                                 #normalizes x to [0,1] and centers on 0
    y = xyt[:, 1] / two_d - 0.5                                                                 #normalizes y to [0,1] and centers on 0
    t = xyt[:, 2] / xyt[:, 2].max()                                                             #normalizes t to [0,1]
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
 
 
def _append(tokens: np.ndarray, column: np.ndarray) -> np.ndarray:
    col = np.broadcast_to(column.astype(np.float32)[None, :, None]                              #turns column into a (1, stabilizer, 1) array
                          , (tokens.shape[0], tokens.shape[1], 1))                              #sets broacast size to (shots, stabilizers, 1)
    return np.concatenate([tokens, col], axis=2)                                                #concatenates tokens and column along right axis
 
 
def make_tokens(circuit: stim.Circuit, dets: np.ndarray, basis: str = "z") -> np.ndarray:
    tokens = create_position_features(circuit, dets)
    tokens = add_stabilizer_type(circuit, tokens, basis)
    return add_final_round(circuit, tokens)