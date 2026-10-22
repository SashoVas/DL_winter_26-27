from collections.abc import Callable

import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    analytical_gradient,
    initialize_weights,
    numerical_gradient,
)


def get_loss_of_only_one_param(x: np.ndarray, y: np.ndarray) -> Callable:
    def mse_pached(w: np.ndarray) -> float:
        return mse(y, x * w)

    return mse_pached


def main() -> None:
    dataset = create_dataset(6)
    x = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])
    w = initialize_weights(0, 10, 1, 42)
    loss = get_loss_of_only_one_param(x, y)
    numerical = numerical_gradient(loss, w, epsilon=1e-7)
    analytical_w, _ = analytical_gradient(dataset, w)
    print("Loss:", loss(w))
    print("Numerical gradient:", numerical)
    print("Analytical gradient:", analytical_w)
    print("Difference:", numerical - analytical_w)
