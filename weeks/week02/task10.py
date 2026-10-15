import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations.functional import sigmoid


def main() -> None:
    x = np.linspace(-10, 10, 500)
    res = torch.sigmoid(torch.from_numpy(x))
    y = sigmoid(x)
    if not np.allclose(res, y):
        raise ValueError("Sigmoid values does not match")
    print("sigmoid(-10) =", sigmoid(np.array([-10])).item())
    print("sigmoid(0) =", sigmoid(np.array([0])).item())
    print("sigmoid(10) =", sigmoid(np.array([10])).item())
    print("Output range:", [y.min().item(), y.max().item()])

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.title("sigmoid(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
