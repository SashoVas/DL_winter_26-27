import numpy as np

from dl_lib.perceptron.datasets import and_dataset, or_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train
from dl_lib.losses.functional import binary_cross_entropy, mse
from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.functional import numerical_gradient, analytical_gradient
import matplotlib.pyplot as plt
from typing import Callable


def get_loss_of_only_one_param_weight(
    x: np.ndarray, y: np.ndarray, model: Perceptron
) -> Callable:
    def mse_pached(w: np.ndarray) -> float:
        return binary_cross_entropy(y, model.activation(x @ w + model.bias))

    return mse_pached


def get_loss_of_only_one_param_bias(
    x: np.ndarray, y: np.ndarray, model: Perceptron
) -> Callable:
    def mse_pached(b: np.ndarray) -> float:
        return binary_cross_entropy(y, model.activation(x @ model.weights + b))

    return mse_pached


def main() -> None:
    dataset = and_dataset()
    x = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])
    model = Perceptron(2, use_bias=True, activation=sigmoid, seed=42)
    loss_weight = get_loss_of_only_one_param_weight(x, y, model)
    loss_bias = get_loss_of_only_one_param_bias(x, y, model)
    grad_w_num = numerical_gradient(loss_weight, model.weights, epsilon=1e-7)
    grad_b_num = numerical_gradient(loss_bias, model.bias, epsilon=1e-7)
    print(f"AND gate (sigmoid) - weights: {model.weights}, bias={model.bias}")
    print(
        "Numerical weight gradient:",
        grad_w_num,
    )
    grad_w, grad_b = analytical_gradient(
        dataset, model.weights, model.bias, model.activation, loss="bce"
    )
    print(
        "Analytical weight gradient:",
        grad_w,
    )
    print("Weight gradient difference:", grad_w - grad_w_num)
    print("Numerical bias gradient:", grad_b_num)
    print("Analytical bias gradient:", grad_b)
    print("Bias gradient difference:", grad_b - grad_b_num)
