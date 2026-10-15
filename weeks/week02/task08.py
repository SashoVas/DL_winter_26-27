import numpy as np

from dl_lib.metrics.functional import accuracy, f1_score, precision, recall


def main() -> None:

    pred = np.array([0] * 1000)
    true = np.array([0] * 990 + [1] * 10)
    print("Dataset: 990 negative, 10 positive")
    print("Model: always predicts negative, never looks at the input")
    print(" accuracy:", accuracy(true, pred))
    print(" precision:", precision(true, pred))
    print(" recall:", recall(true, pred))
    print(" f1:", f1_score(true, pred))
