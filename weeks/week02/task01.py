import numpy as np

DEFAULT_PRED = np.array([3, 5, 7, 9])
DEFAULT_TRUE = np.array([2, 6, 8, 7])


def main(pred: np.ndarray = DEFAULT_PRED, true: np.ndarray = DEFAULT_TRUE) -> None:
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
