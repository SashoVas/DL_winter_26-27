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


def analytical_gradient(
    dataset: list,
    weights: np.ndarray,
    bias: np.ndarray | float = 0,
    activation: Callable = lambda x: x,
    activation_derivative: Callable = lambda x: np.ones_like(x),
) -> tuple:

    x = np.array([i[0] for i in dataset])
    if x.ndim == 1:
        x = x.reshape(len(x), 1)
    y = np.array([i[1] for i in dataset])

    logits = (x @ weights + bias).squeeze()
    probs = activation(logits)
    anti_der = activation_derivative(logits)
    w_grad = (2 / len(dataset)) * ((probs - y) * anti_der * x.squeeze()).sum()
    bias_grad = (2 / len(dataset)) * ((probs - y) * anti_der).sum()
    return w_grad, bias_grad


def gradient_descent_step(params: np.ndarray, grad: np.ndarray, lr: float):
    return params - lr * grad
