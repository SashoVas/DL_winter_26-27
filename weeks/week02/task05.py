import numpy as np

from dl_lib.metrics.functional import apply_threshold

DEFAULT_PROB = np.array([0.1, 0.8, 0.4, 0.7, 0.9])


def main(prob: np.ndarray = DEFAULT_PROB) -> None:
    print("Probabilities:", prob)
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"threshold={threshold}: predictions={apply_threshold(prob, threshold)}")
