from collections.abc import Callable

import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    gradient_descent_step,
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

    print("Loss before update: ", loss(w))
    grad = numerical_gradient(loss, w, epsilon=1e-7)
    print("Numerical gradient:", grad)
    print("w after subtracting the raw gradient:")
    print(f"w={w - grad}, loss={loss(w - grad)}")
    print("w after one step with learning_rate=0.001:")
    print(f"w={w - 0.001 * grad}, loss={loss(w - 0.001 * grad)}")

    print()
    l = loss(w)
    print("Loss before update:", l)
    for epoch in range(10):
        w = gradient_descent_step(w, numerical_gradient(loss, w, epsilon=1e-7), 0.001)
        l = loss(w)
        print(f"epoch  {epoch + 1}: w={w}, loss={l}")
