import numpy as np


def apply_threshold(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(dtype=int)


def confusion_matrix(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "tp": ((pred == 1) & (true == 1)).sum(),
        "tn": ((pred == 0) & (true == 0)).sum(),
        "fp": ((pred == 0) & (true == 1)).sum(),
        "fn": ((pred == 1) & (true == 0)).sum(),
    }
