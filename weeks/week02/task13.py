import numpy as np

from dl_lib.losses.functional import binary_cross_entropy, categorical_cross_entropy


def main() -> None:
    p = np.array([0.9, 0.2, 0.6])
    true = np.array([1, 0, 1])
    print(
        f"Binary cross-entropy on y_true={true}, p={p}: {binary_cross_entropy(true, p)}"
    )
    print(
        f"y_true=1, p=0.999 (confidently right): loss={
            binary_cross_entropy(np.array([1]), np.array([[0.999]]))
        }"
    )

    print(
        f"y_true=1, p=0.001 (confidently wrong): loss={
            binary_cross_entropy(np.array([1]), np.array([[0.001]]))
        }"
    )
    print()

    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    true = np.array([0, 1, 2])

    print(
        f"Multiclass cross-entropy on y_true={true}: {categorical_cross_entropy(true, p)}"
    )
    print()

    print(
        f"y_true=0, p_true=0.999 (confidently right): loss={
            categorical_cross_entropy(np.array([0]), np.array([[0.999, 0.001]]))
        }"
    )
    print(
        f"y_true=0, p_true=0.999 (confidently right): loss={
            categorical_cross_entropy(np.array([0]), np.array([[0.001, 0.999]]))
        }"
    )
