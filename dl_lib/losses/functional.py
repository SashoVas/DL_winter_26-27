import numpy as np


def mae(pred: np.ndarray, true: np.ndarray, axis: int | None = None) -> np.float64:
    return np.abs(true - pred).mean(axis=axis)


def mse(pred: np.ndarray, true: np.ndarray, axis: int | None = None) -> np.float64:
    return np.pow(true - pred, 2).mean(axis=axis)


def rmse(pred: np.ndarray, true: np.ndarray, axis: int | None = None) -> np.float64:
    return np.sqrt(mse(pred, true, axis=axis))
