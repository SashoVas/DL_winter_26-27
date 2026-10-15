import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> np.float64:
    return np.abs(true - pred).mean()


def mse(pred: np.ndarray, true: np.ndarray) -> np.float64:
    return np.pow(true - pred, 2).mean()
