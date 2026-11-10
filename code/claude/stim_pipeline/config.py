"""Edit these defaults to choose the experiment; the CLI can override them."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExperimentConfig:
    distance: int = 3
    rounds: int = 3
    physical_error_rate: float = 0.001
    shots: int = 10_000
    seed: int = 42
    # Separate random streams: changing the split does not change the simulation.
    split_seed: int = 43
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    circuit_type: str = "surface_code:rotated_memory_z"

    def __post_init__(self):
        for name, minimum in (("distance", 2), ("rounds", 1), ("shots", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("seed", "split_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError(f"{name} must be an integer in [0, 2**64)")
        if not 0 <= self.physical_error_rate <= 1:
            raise ValueError("physical_error_rate must be in [0, 1]")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if not all(0 < f < 1 for f in fractions) or not math.isclose(sum(fractions), 1):
            raise ValueError("Split fractions must be positive and sum to 1")
        train = int(self.shots * self.train_fraction)
        validation = int(self.shots * self.validation_fraction)
        if min(train, validation, self.shots - train - validation) < 1:
            raise ValueError("Increase shots so every split contains at least one sample")
        if self.circuit_type != "surface_code:rotated_memory_z":
            raise ValueError("This first-stage experiment supports rotated_memory_z")
