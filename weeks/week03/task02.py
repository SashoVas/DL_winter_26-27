import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights


def main() -> None:
    dataset = create_dataset(6)
    x = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])
    w = initialize_weights(0, 10, 1, 42)
    y_hat = x * w
    print("Dataset:", dataset)
    print("w:", w)
    print(mse(y, y_hat))
    print()
    print("Perturbing w:")
    for epsilon in [0.002, 0.001, -0.001, -0.002]:
        new_w = w + epsilon
        print(f"  w + ({epsilon}) -> w={new_w}, MSE={mse(y, x * new_w)}")
