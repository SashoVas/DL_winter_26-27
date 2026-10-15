import numpy as np
import torch
import torch.nn.functional as F

from dl_lib.losses.functional import mae

DEFAULT_PRED = np.array([3, 5, 7, 9])
DEFAULT_TRUE = np.array([2, 6, 8, 7])


def main(pred: np.ndarray = DEFAULT_PRED, true: np.ndarray = DEFAULT_TRUE) -> None:
    my_mae = mae(pred, true)
    print("y_true:", true)
    print("y_pred:", pred)
    print("MAE:", my_mae)

    torch_mae = F.l1_loss(
        torch.from_numpy(pred).float(),
        torch.from_numpy(true).float(),
    )

    if not np.isclose(my_mae, torch_mae):
        raise ValueError("Numbers not equal")
