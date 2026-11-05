import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import SGD, AdaGrad


def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 10 * x**2 + 0.1 * y**2


def f2(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x**2


def plot() -> None:
    x = torch.tensor([4.0], requires_grad=True)
    y = torch.tensor([4.0], requires_grad=True)
    x2 = torch.tensor([4.0], requires_grad=True)
    y2 = torch.tensor([4.0], requires_grad=True)
    optimizer = SGD([x, y], lr=0.1, momentum=0.9)
    optimizer2 = AdaGrad([x2, y2], lr=1)
    xs = []
    ys = []
    xs2 = []
    ys2 = []
    for _ in range(50):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        z = f(x, y)
        z2 = f(x2, y2)
        xs.append(x[0].item())
        ys.append(y[0].item())
        xs2.append(x2[0].item())
        ys2.append(y2[0].item())
        z.backward()
        z2.backward()
        optimizer.step()
        optimizer2.step()

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-4.5, 4.5, 200), torch.linspace(-4.5, 4.5, 200), indexing="xy"
    )
    grid_z = f(grid_x, grid_y)
    _, axis = plt.subplots()
    axis.plot(xs, ys, label="SGD(momentum=0.9)", color="orange")
    axis.plot(xs2, ys2, label="AdaGrad", color="green")
    axis.scatter(xs, ys, s=10, c="orange")
    axis.scatter(xs2, ys2, s=10, c="green")
    axis.legend()
    axis.contour(grid_x, grid_y, grid_z, levels=30, cmap="Greys")
    plt.title("AdaGrad vs. SGD with momentum on an elongated ravine")
    plt.show()
    print(f"SGD (momentum) final: x={x[0].item()}, y={y[0].item()}")
    print(f"AdaGrad final: x={x2[0].item()}, y={y2[0].item()}")


def plot2() -> None:
    x = torch.tensor([5.0], requires_grad=True)
    x2 = torch.tensor([5.0], requires_grad=True)
    optimizer = AdaGrad([x], lr=0.5)
    optimizer2 = AdaGrad([x2], lr=0.5)
    xs = []
    xs2 = []

    for i in range(150):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        y = f2(x)
        y2 = f2(x2)
        xs.append(x[0].item())
        xs2.append(x2[0].item())

        y.backward()
        y2.backward()
        if i == 0 and x.grad is not None:
            x.grad *= 100
        optimizer.step()
        optimizer2.step()

    plt.plot(
        list(range(len(xs2))), xs2, label="AdaGrad (no early spike)", color="green"
    )
    plt.plot(
        list(range(len(xs))),
        xs,
        label="AdaGrad (one early gradient spike)",
        color="red",
    )
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.title("AdaGrad stalls after one unusually large early gradient")
    plt.tight_layout()
    plt.show()
    print(f"AdaGrad (no spike) final x: {x2[0].item()}")
    print(f"AdaGrad (early spike) final x: {x[0].item()}")


def main() -> int:

    plot()
    plot2()
    return 0
