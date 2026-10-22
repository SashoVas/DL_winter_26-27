import numpy as np

from dl_lib.perceptron.datasets import and_dataset, or_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train
from dl_lib.losses.functional import binary_cross_entropy


def main() -> None:
    dataset_and = and_dataset()
    dataset_or = or_dataset()

    model = Perceptron(2, use_bias=True)
    print("AND gate - initial weights:", model.weights)
    train(
        model,
        dataset_and,
        learning_rate=0.001,
        epochs=100_000,
        log_every=10_000,
        loss=binary_cross_entropy,
    )
    print("AND gate - final weights:", model.weights)
    print("AND gate - predictions:")
    print(f"AND(0, 0) = 0 -> predicted {model.forward(np.array([0, 0]))}")
    print(f"AND(0, 1) = 0 -> predicted {model.forward(np.array([0, 1]))}")
    print(f"AND(1, 0) = 0 -> predicted {model.forward(np.array([1, 0]))}")
    print(f"AND(1, 1) = 1 -> predicted {model.forward(np.array([1, 1]))}")
    print()
    model = Perceptron(2, use_bias=True)
    print("OR  gate - initial weights:", model.weights)
    train(
        model,
        dataset_or,
        learning_rate=0.001,
        epochs=100_000,
        log_every=10_000,
        loss=binary_cross_entropy,
    )
    print("OR  gate - final weights:", model.weights)
    print("OR  gate - predictions:")
    print(f"OR (0, 0) = 0 -> predicted {model.forward(np.array([0, 0]))}")
    print(f"OR (0, 1) = 0 -> predicted {model.forward(np.array([0, 1]))}")
    print(f"OR (1, 0) = 0 -> predicted {model.forward(np.array([1, 0]))}")
    print(f"OR (1, 1) = 1 -> predicted {model.forward(np.array([1, 1]))}")
