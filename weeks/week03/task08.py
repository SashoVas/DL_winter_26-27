import numpy as np

from dl_lib.perceptron.datasets import and_dataset, or_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train
from dl_lib.losses.functional import binary_cross_entropy, mse
from dl_lib.activations.functional import sigmoid
import matplotlib.pyplot as plt


def main() -> None:
    dataset_and = and_dataset()

    model = Perceptron(2, use_bias=True)
    res = train(
        model,
        dataset_and,
        learning_rate=0.1,
        epochs=100_000,
        log_every=1_000,
        loss=mse,
        enable_prints=False,
    )
    print("Without sigmoid - final loss:", res[-1][1])
    print("AND gate - predictions:")
    print(f"AND(0, 0) = 0 -> predicted {model.forward(np.array([0, 0]))}")
    print(f"AND(0, 1) = 0 -> predicted {model.forward(np.array([0, 1]))}")
    print(f"AND(1, 0) = 0 -> predicted {model.forward(np.array([1, 0]))}")
    print(f"AND(1, 1) = 1 -> predicted {model.forward(np.array([1, 1]))}")

    model = Perceptron(2, use_bias=True, activation=sigmoid)
    res2 = train(
        model,
        dataset_and,
        learning_rate=0.1,
        epochs=100_000,
        log_every=10,
        loss=mse,
        enable_prints=False,
    )
    print("Without sigmoid - final loss:", res2[-1][1])
    print("AND gate - predictions:")
    print(f"AND(0, 0) = 0 -> predicted {model.forward(np.array([0, 0]))}")
    print(f"AND(0, 1) = 0 -> predicted {model.forward(np.array([0, 1]))}")
    print(f"AND(1, 0) = 0 -> predicted {model.forward(np.array([1, 0]))}")
    print(f"AND(1, 1) = 1 -> predicted {model.forward(np.array([1, 1]))}")

    x = [i[0] for i in res]
    y = [i[1] for i in res]
    plt.plot(x, y, label="no activation")
    x = [i[0] for i in res2]
    y = [i[1] for i in res2]
    plt.plot(x, y, label="sigmoid")
    plt.grid(visible=True)
    plt.tight_layout()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("AND gate - loss over 100,000 epochs")
    plt.show()
