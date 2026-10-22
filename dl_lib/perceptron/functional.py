from collections.abc import Callable

import numpy as np

from dl_lib.losses.functional import mse


def initialize_weights(
    low: float, high: float, size: int, seed: int | None = None
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
    loss="mse",
) -> tuple:

    x = np.array([i[0] for i in dataset])
    if x.ndim == 1:
        x = x.reshape(len(x), 1)
    y = np.array([i[1] for i in dataset])

    logits = x @ weights + bias
    probs = activation(logits)
    w_grad = None
    bias_grad = None
    if loss == "mse":
        anti_der = activation_derivative(logits)
        w_grad = (2 / len(dataset)) * (x.T @ ((probs - y) * anti_der))
        bias_grad = (2 / len(dataset)) * ((probs - y) * anti_der).sum()
    elif loss == "bce":
        w_grad = x.T @ ((probs - y) / len(y))
        bias_grad = (probs - y).sum() / len(y)
    return w_grad, bias_grad


def gradient_descent_step(params: np.ndarray, grad: np.ndarray, lr: float):
    return params - lr * grad


def forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: float | np.ndarray = 0,
    activation: Callable = lambda x: x,
) -> np.ndarray:
    return activation(x @ weights + bias)


def dataset_loss(
    dataset,
    targets,
    weights,
    bias,
    activation: Callable | None = lambda x: x,
    loss: Callable = mse,
) -> float:
    if activation == None:
        activation = lambda x: x
    x = np.array([i[0] for i in dataset])
    res = forward(x, weights, bias, activation)
    return loss(targets, res)


def dataset_loss_given_bias(
    dataset, targets, weights, activation, bias_array, loss: Callable = mse
):
    if activation == None:
        activation = lambda x: x
    x = np.array([i[0] for i in dataset])
    res = forward(x, weights, bias_array, activation)
    return loss(targets, res)
