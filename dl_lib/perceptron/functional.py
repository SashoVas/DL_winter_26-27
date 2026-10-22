import numpy as np


def initialize_weights(
    low: float, high: float, size: int, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(low=low, high=high, size=size)
