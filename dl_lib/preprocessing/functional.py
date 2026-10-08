import numpy as np


def min_max_normalize(array: np.array):
    min = array.min()
    max = array.max()
    if np.isclose(min, max):
        return np.zeros(len(array))

    return (array - min) / (max - min)


def standardize(array: np.array):
    mu = array.mean()
    sigma = array.std()
    if np.isclose(sigma, 0):
        return np.zeros(len(array))
    return (array - mu) / sigma
