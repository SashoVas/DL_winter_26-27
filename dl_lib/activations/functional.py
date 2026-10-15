import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    maximum = x.max(axis=-1, keepdims=True)
    exps = np.exp(x - maximum)
    return exps / exps.sum(axis=-1, keepdims=True)
