import numpy as np


def apply_threshold(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(dtype=int)


def confusion_matrix(true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "tp": ((pred == 1) & (true == 1)).sum(),
        "tn": ((pred == 0) & (true == 0)).sum(),
        "fp": ((pred == 1) & (true == 0)).sum(),
        "fn": ((pred == 0) & (true == 1)).sum(),
    }


def accuracy(true: np.ndarray, pred: np.ndarray) -> np.float64:
    res = confusion_matrix(true, pred)
    denominator = res["tp"] + res["tn"] + res["fp"] + res["fn"]
    if denominator == 0:
        return np.float64(0.0)

    return (res["tp"] + res["tn"]) / denominator


def precision(true: np.ndarray, pred: np.ndarray) -> np.float64:
    res = confusion_matrix(true, pred)
    denominator = res["tp"] + res["fp"]
    if denominator == 0:
        return np.float64(0.0)
    return res["tp"] / denominator


def recall(true: np.ndarray, pred: np.ndarray) -> np.float64:
    res = confusion_matrix(true, pred)
    denominator = res["tp"] + res["fn"]
    if denominator == 0:
        return np.float64(0.0)
    return res["tp"] / denominator


def f1_score(true: np.ndarray, pred: np.ndarray) -> np.float64:
    res = confusion_matrix(true, pred)
    denominator = 2 * res["tp"] + res["fp"] + res["fn"]
    if denominator == 0:
        return np.float64(0.0)
    return 2 * res["tp"] / denominator
