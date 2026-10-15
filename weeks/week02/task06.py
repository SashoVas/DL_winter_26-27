import numpy as np

from dl_lib.metrics.functional import confusion_matrix

DEFAULT_TRUE = np.array([1, 0, 1, 0, 1])
DEFAULT_PRED = np.array([1, 0, 0, 1, 1])


def main(pred: np.ndarray = DEFAULT_PRED, true: np.ndarray = DEFAULT_TRUE) -> None:
    print("y_true:", true)
    print("y_pred:", pred)
    cm = confusion_matrix(true, pred)
    print("True positives:", cm["tp"])
    print("True negatives:", cm["tn"])
    print("False positives:", cm["fp"])
    print("False negatives:", cm["fn"])
