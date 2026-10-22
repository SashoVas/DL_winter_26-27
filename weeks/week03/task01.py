import numpy as np
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights


def main() -> None:
    dataset = create_dataset(4)
    print("Dataset:", dataset)
    print("initialize_weights(0, 100):", initialize_weights(0, 100, 1))
    print("initialize_weights(0, 10):", initialize_weights(0, 10, 1))
