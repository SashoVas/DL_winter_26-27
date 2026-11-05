import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import SGD


def f(x: torch.Tensor) -> torch.Tensor:
    return x**4 + x**3 - 5 * x**2


def main() -> int:
    x = torch.tensor([2.0], requires_grad=True)
    optimizer = SGD([x], lr=0.02)
    xs = []
    ys = []
    for _ in range(10):
        optimizer.zero_grad()
        y = f(x)
        xs.append(x[0].item())
        ys.append(y[0].item())
        y.backward()
        optimizer.step()
    dx = torch.linspace(-3.0, 2.3, 100)
    plt.plot(dx, f(dx), color="gray", label="f(x) = x^4 + x^3 - 5x^2")
    plt.plot(xs, ys, label="SGD (momenutm=0.0)")
    plt.scatter(xs, ys, s=10)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Plain SGD trajectory trapped in a local minimum")
    plt.show()
    return 0
