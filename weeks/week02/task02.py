from dl_lib.losses.functional import mae
import numpy as np
import torch.nn.functional as F
import torch


def main(
    pred: np.ndarray = np.array([3, 5, 7, 9]), true: np.ndarray = np.array([2, 6, 8, 7])
) -> None:
    my_mae = mae(pred, true)
    print("y_true:", true)
    print("y_pred:", pred)
    print("MAE:", my_mae)

    torch_mae = F.l1_loss(
        torch.from_numpy(pred).float(),
        torch.from_numpy(true).float(),
    )

    if not np.isclose(my_mae, torch_mae):
        raise Exception("Numbers not equal")
