import numpy as np


def main(
    pred: np.ndarray = np.array([3, 5, 7, 9]), true: np.ndarray = np.array([2, 6, 8, 7])
) -> np.float64:
    errors = true - pred
    me = errors.mean()

    absolute_errors = np.abs(errors)
    mae = absolute_errors.mean()

    squared_errors = np.pow(errors, 2)
    mse = squared_errors.mean()
    rmse = np.sqrt(mse)
    print("y_true:", true)
    print("y_pred:", pred)
    print("errors:", errors)
    print("absolute errors:", absolute_errors)
    print("squared errors:", squared_errors)
    print()
    print("mean error:", me)
    print("MAE (mean absolute error):", mae)
    print("MSE (mean squared error):", mse)
    print("RMSE (root mean squared error):", rmse)

    return me, mae, mse, rmse
