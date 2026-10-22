import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.losses.functional import binary_cross_entropy
from dl_lib.perceptron.datasets import nand_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train


def main() -> None:
    dataset = nand_dataset()

    model = Perceptron(2, use_bias=True, activation=sigmoid)
    train(
        model,
        dataset,
        learning_rate=0.1,
        epochs=100_000,
        log_every=10_000,
        loss=binary_cross_entropy,
        enable_prints=True,
    )
    print(f"NAND gate - final weights: {model.weights}, bias: {model.bias}")
    print(f"NAND(0, 0) = 0 -> predicted {model.forward(np.array([0, 0]))}")
    print(f"NAND(0, 1) = 0 -> predicted {model.forward(np.array([0, 1]))}")
    print(f"NAND(1, 0) = 0 -> predicted {model.forward(np.array([1, 0]))}")
    print(f"NAND(1, 1) = 1 -> predicted {model.forward(np.array([1, 1]))}")
