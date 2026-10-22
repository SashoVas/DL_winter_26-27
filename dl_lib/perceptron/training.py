from collections.abc import Callable

import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.functional import analytical_gradient, gradient_descent_step
from dl_lib.perceptron.perceptron import Perceptron


def train(
    model: Perceptron,
    dataset: list,
    learning_rate: float,
    epochs: int,
    log_every: int = 1,
    loss: Callable = mse,
    enable_prints: bool = True,
) -> list:
    x = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])
    result = []
    for epoch in range(1, epochs + 1):
        grad_w, grad_b = analytical_gradient(
            dataset, model.weights, bias=model.bias, activation=model.activation
        )
        model.weights = gradient_descent_step(model.weights, grad_w, lr=learning_rate)
        if model.use_bias:
            model.bias = gradient_descent_step(model.bias, grad_b, lr=learning_rate)
        l = loss(y, model.forward(x))
        if epoch % log_every == 0:
            if enable_prints:
                print(f"epoch {epoch}: loss={l}")
            result.append((epoch, l))

    return result
