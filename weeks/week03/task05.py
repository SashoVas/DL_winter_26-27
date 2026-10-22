from collections.abc import Callable

import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    analytical_gradient,
    gradient_descent_step,
    initialize_weights,
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

    print(f"Seed 42, initial w={w}")
    l = loss(w)
    for _ in range(500):
        w = gradient_descent_step(w, analytical_gradient(dataset, w)[0], 0.001)
        l = loss(w)
    print(f"After 500 epochs: w={w}, loss={l} (target: w=2.0)")
    print()
    print("Training again without a seed:")
    w = initialize_weights(0, 10, 1)

    print(f"No seed, initial w={w}")
    for _ in range(500):
        w = gradient_descent_step(w, analytical_gradient(dataset, w)[0], 0.001)
        l = loss(w)
    print(f"After 500 epochs: w={w}, loss={l} (target: w=2.0)")
