import numpy as np


def apply_threshold(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(dtype=int)
