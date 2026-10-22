from collections.abc import Callable

import numpy as np


def initialize_weights(
    low: float, high: float, size: int, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(low=low, high=high, size=size)


def numerical_gradient(
    loss: Callable, params: np.ndarray, epsilon: float = 0.0001
) -> np.ndarray:
    res = np.zeros_like(params)
    original_loss = loss(params)

    for index in np.ndindex(params.shape):
        original = params[index]
        params[index] = params[index] + epsilon
        new_loss = loss(params)
        params[index] = original
        res[index] = (new_loss - original_loss) / epsilon

    return res


def gradient_descent_step(params: np.ndarray, grad: np.ndarray, lr: float):
    return params - lr * grad
