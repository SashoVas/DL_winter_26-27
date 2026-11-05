import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import SGD


def f(x: torch.Tensor) -> torch.Tensor:
    return x**4 + x**3 - 5 * x**2


def f2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 10 * x**2 + 0.1 * y**2


def plot_with_momentum(momentum: float = 0, color: str | None = None) -> None:
    x = torch.tensor([2.0], requires_grad=True)
    optimizer = SGD([x], lr=0.02, momentum=momentum)
    xs = []
    ys = []
    for _ in range(50):
        optimizer.zero_grad()
        y = f(x)
        xs.append(x[0].item())
        ys.append(y[0].item())
        y.backward()
        optimizer.step()
    plt.plot(xs, ys, label=f"SGD (momenutm={optimizer.momentum})", color=color)
    plt.scatter(xs, ys, s=10)


def plot_second():
    x = torch.tensor([4.0], requires_grad=True)
    y = torch.tensor([4.0], requires_grad=True)
    optimizer = SGD([x, y], lr=0.1, momentum=0.9)
    xs = []
    ys = []
    for _ in range(50):
        optimizer.zero_grad()
        z = f2(x, y)
        xs.append(x[0].item())
        ys.append(y[0].item())
        z.backward()
        optimizer.step()

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-4.5, 4.5, 200), torch.linspace(-4.5, 4.5, 200), indexing="xy"
    )
    grid_z = f2(grid_x, grid_y)
    _, axis = plt.subplots()
    axis.plot(xs, ys, label=f"SGD (momenutm={optimizer.momentum})", color="orange")
    axis.scatter(xs, ys, s=10, c="orange")

    axis.contour(grid_x, grid_y, grid_z, levels=30, cmap="Greys")
    axis.legend()
    plt.title("SGD with momentum zig-zagging in a steep, narrow ravine")
    plt.tight_layout()
    plt.show()


def main() -> int:
    plot_with_momentum(0)
    plot_with_momentum(0.9)
    dx = torch.linspace(-3.0, 2.3, 100)
    plt.plot(dx, f(dx), color="gray", label="f(x) = x^4 + x^3 - 5x^2")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Plain SGD trajectory with and without momentum")
    plt.tight_layout()
    plt.show()
    plot_second()
    return 0
