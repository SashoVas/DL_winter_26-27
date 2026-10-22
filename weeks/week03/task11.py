from collections.abc import Callable
from functools import partial

import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron.datasets import xor_dataset
from dl_lib.perceptron.functional import gradient_descent_step, numerical_gradient
from dl_lib.perceptron.xor import Xor


def _loss_given(
    model: Xor,
    x: np.ndarray,
    targets: np.ndarray,
    param_name: str,
    param: np.ndarray | float,
) -> float:
    original = getattr(model, param_name)
    setattr(model, param_name, param)
    predictions = model.forward(x)
    setattr(model, param_name, original)
    return losses_functional.binary_cross_entropy(targets, predictions)


def train(
    model: Xor,
    dataset: list,
    learning_rate: float,
    epochs: int,
    log_every: int = 1,
    loss: Callable = losses_functional.binary_cross_entropy,
    enable_prints: bool = True,
) -> list:
    x = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])
    hidden_weights_loss = partial(_loss_given, model, x, y, "hidden_weights")
    output_weights_loss = partial(_loss_given, model, x, y, "output_weights")
    hidden_bias_loss = partial(_loss_given, model, x, y, "hidden_bias")
    output_bias_loss = partial(_loss_given, model, x, y, "output_bias")
    result = []

    # if bias is just a float, the numerical_gradient wont work as is, because i asume it is np.array but the tests want it to be float
    # and i am too lazy to fix it :)
    model.output_bias = np.array([model.output_bias])
    for epoch in range(1, epochs + 1):
        grad_hw = numerical_gradient(hidden_weights_loss, model.hidden_weights)
        grad_ow = numerical_gradient(output_weights_loss, model.output_weights)
        grad_hb = numerical_gradient(hidden_bias_loss, model.hidden_bias)
        grad_ob = numerical_gradient(output_bias_loss, model.output_bias)

        model.hidden_weights = gradient_descent_step(
            model.hidden_weights, grad_hw, lr=learning_rate
        )
        model.output_weights = gradient_descent_step(
            model.output_weights, grad_ow, lr=learning_rate
        )
        model.hidden_bias = gradient_descent_step(
            model.hidden_bias, grad_hb, lr=learning_rate
        )
        model.output_bias = gradient_descent_step(
            model.output_bias, grad_ob, lr=learning_rate
        )

        l = loss(y, model.forward(x))
        if epoch % log_every == 0:
            if enable_prints:
                print(f"epoch {epoch}: loss={l}")
            result.append((epoch, l))

    # fixing the bias to be a float
    model.output_bias = model.output_bias[0]
    return result


def main() -> None:
    dataset = xor_dataset()
    model = Xor(seed=1)
    print("First attempt - seed=1:")
    train(
        model,
        dataset,
        learning_rate=0.1,
        epochs=10_000,
        log_every=1_000,
    )
    print(f"XOR(0, 0) = 0 -> predicted {model.forward(np.array([0, 0]))}")
    print(f"XOR(0, 1) = 0 -> predicted {model.forward(np.array([0, 1]))}")
    print(f"XOR(1, 0) = 0 -> predicted {model.forward(np.array([1, 0]))}")
    print(f"XOR(1, 1) = 1 -> predicted {model.forward(np.array([1, 1]))}")
    print()
    print("Second attempt - seed=42 (a different random start):")
    model = Xor(seed=42)
    train(
        model,
        dataset,
        learning_rate=0.1,
        epochs=10_000,
        log_every=1_000,
    )
    print(f"XOR(0, 0) = 0 -> predicted {model.forward(np.array([0, 0]))}")
    print(f"XOR(0, 1) = 0 -> predicted {model.forward(np.array([0, 1]))}")
    print(f"XOR(1, 0) = 0 -> predicted {model.forward(np.array([1, 0]))}")
    print(f"XOR(1, 1) = 1 -> predicted {model.forward(np.array([1, 1]))}")
