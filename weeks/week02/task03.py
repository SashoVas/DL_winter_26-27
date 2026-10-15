import numpy as np
import torch
import torch.nn.functional as F

from dl_lib.losses.functional import mae, mse

DEFAULT_PRED = np.array([3, 5, 7, 9])
DEFAULT_TRUE = np.array([2, 6, 8, 7])


def main(pred: np.ndarray = DEFAULT_PRED, true: np.ndarray = DEFAULT_TRUE) -> None:
    my_mse = mse(pred, true)
    print("y_true:", true)
    print("y_pred:", pred)
    print("MSE:", my_mse)
    print()
    torch_mae = F.mse_loss(
        torch.from_numpy(pred).float(),
        torch.from_numpy(true).float(),
    )

    if not np.isclose(my_mse, torch_mae):
        raise ValueError("Numbers not equal")

    pred2 = np.array([2.0, 6.0, 8.0, 100.0])
    print("y_pred with one extreme outlier:", pred2)
    print("MAE with outlier:", mae(pred2, true))
    print("MSE with outlier:", mse(pred2, true))
