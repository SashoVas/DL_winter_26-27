import matplotlib.pyplot as plt
import numpy as np

from dl_lib.losses.functional import mae, mse, rmse


def main() -> None:
    pred = np.array([0.0, 5.0, 9.0, 6.5]).reshape(1, 4).repeat(50, axis=0)
    true = np.array([0.0, 6.0, 8.0, 7.0]).reshape(1, 4).repeat(50, axis=0)
    growth = np.linspace(0.0, 10.0, 50).reshape(50)
    pred[:, 0] = growth

    mae_losses = np.asarray(mae(pred, true, axis=1))
    mse_losses = np.asarray(mse(pred, true, axis=1))
    rmse_losses = np.asarray(rmse(pred, true, axis=1))

    print(
        f"Growing error={growth[0]}: MAE={mae_losses[0]}, MSE={mse_losses[0]}, RMSE={rmse_losses[0]}"
    )
    print(
        f"Growing error={growth[25]}: MAE={mae_losses[25]}, MSE={mse_losses[25]}, RMSE={rmse_losses[25]}"
    )
    print(
        f"Growing error={growth[-1]}: MAE={mae_losses[-1]}, MSE={mse_losses[-1]}, RMSE={rmse_losses[-1]}"
    )

    plt.plot(growth, mae_losses, color="blue", label="MAE")
    plt.plot(growth, mse_losses, color="orange", label="MSE")
    plt.plot(growth, rmse_losses, color="green", label="RMSE")
    plt.title("MAE, MSE and RMSE as one prediction error grows")
    plt.xlabel("the growing error, with three other errors (1, -1, 0.5) held fixed")
    plt.ylabel("loss value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
