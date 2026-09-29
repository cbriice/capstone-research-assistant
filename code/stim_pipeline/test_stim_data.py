import stim

# Create a small noisy surface-code memory experiment
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.001,
    before_measure_flip_probability=0.001,
    after_reset_flip_probability=0.001,
)

# Compile fast detector sampler
sampler = circuit.compile_detector_sampler()

# Generate labeled data
dets, obs = sampler.sample(
    shots=10,
    separate_observables=True
)

print("Detector data:")
print(dets)

print("\nLogical observable labels:")
print(obs)

print("\nShapes:")
print("X:", dets.shape)
print("y:", obs.shape)