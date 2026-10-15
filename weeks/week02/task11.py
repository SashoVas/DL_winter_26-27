import matplotlib.pyplot as plt
import numpy as np

from dl_lib.activations.functional import relu, sigmoid, tanh


def main() -> None:
    x = np.linspace(-10, 10, 500)
    sigmoid_x = sigmoid(x)
    relu_x = relu(x)
    tanh_x = tanh(x)

    print("sigmoid range:", [sigmoid_x.min().item(), sigmoid_x.max().item()])
    print("relu range:", [relu_x.min().item(), relu_x.max().item()])
    print("tanh range:", [tanh_x.min().item(), tanh_x.max().item()])
    print()
    for i in [-5, 0, 5]:
        print(
            f"At x={i}:  sigmoid={sigmoid(np.array([i])).item()}, relu={relu(np.array([i])).item()}, tanh={tanh(np.array([i])).item()}"
        )

    plt.plot(x, sigmoid_x, label="sigmoid", color="blue")
    plt.plot(x, relu_x, label="relu", color="orange")
    plt.plot(x, tanh_x, label="tanh", color="green")
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.title("sigmoid, relu, tanh")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
