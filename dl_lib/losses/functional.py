import numpy as np

EPSILON = 0.00001


def mae(true: np.ndarray, pred: np.ndarray, axis: int | None = None) -> np.float64:
    return np.abs(true - pred).mean(axis=axis)


def mse(true: np.ndarray, pred: np.ndarray, axis: int | None = None) -> np.float64:
    return np.pow(true - pred, 2).mean(axis=axis)


def rmse(true: np.ndarray, pred: np.ndarray, axis: int | None = None) -> np.float64:
    return np.sqrt(mse(pred, true, axis=axis))


def binary_cross_entropy(true: np.ndarray, pred: np.ndarray):
    pred = np.clip(pred, EPSILON, 1 - EPSILON)

    return -(true * np.log(pred) + (1 - true) * np.log(1 - pred)).mean()


def categorical_cross_entropy(true: np.ndarray, pred: np.ndarray):
    pred = np.clip(pred, EPSILON, 1 - EPSILON)

    return -np.log(pred[np.arange(len(true)), true]).mean()
