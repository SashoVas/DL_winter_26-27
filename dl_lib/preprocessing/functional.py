import numpy as np


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    min = array.min()
    max = array.max()
    if np.isclose(min, max):
        return np.zeros(len(array))

    return (array - min) / (max - min)


def standardize(array: np.ndarray) -> np.ndarray:
    mu = array.mean()
    sigma = array.std()
    if np.isclose(sigma, 0):
        return np.zeros(len(array))
    return (array - mu) / sigma
