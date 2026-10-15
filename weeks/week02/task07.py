import numpy as np

from dl_lib.metrics.functional import accuracy, f1_score, precision, recall


def main() -> None:
    pred = np.array([1] * 500 + [0] * 500)
    np.random.shuffle(pred)
    true = np.array([1] * 500 + [0] * 500)

    print(
        "Balanced dataset:",
    )
    print(" accuracy:", accuracy(true, pred))
    print(" precision:", precision(true, pred))
    print(" recall:", recall(true, pred))
    print(" f1:", f1_score(true, pred))
    print()
    pred = np.array([0] * 990 + [1] * 10)
    np.random.shuffle(pred)

    true = np.array([0] * 990 + [1] * 10)
    print("Imbalanced dataset (990 negative, 10 positive):")
    print(" accuracy:", accuracy(true, pred))
    print(" precision:", precision(true, pred))
    print(" recall:", recall(true, pred))
    print(" f1:", f1_score(true, pred))
