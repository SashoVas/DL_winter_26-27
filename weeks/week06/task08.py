import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import Adam, RMSprop


def f(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x**2


def plot() -> None:
    x = torch.tensor([5.0], requires_grad=True)
    x2 = torch.tensor([5.0], requires_grad=True)
    optimizer = RMSprop([x], lr=0.1)
    optimizer2 = Adam([x2], lr=0.1)
    xs = []
    xs2 = []
    for i in range(100):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        y = f(x)
        y2 = f(x2)
        xs.append(x[0].item())
        xs2.append(x2[0].item())

        y.backward()
        y2.backward()
        if x.grad is not None and x2.grad is not None:
            x.grad += torch.randn_like(x.grad)
            x2.grad += torch.randn_like(x.grad)
        optimizer.step()
        optimizer2.step()
    plt.plot(
        list(range(len(xs))),
        xs,
        label="RMSprop (noisy gradient)",
        color="red",
    )
    plt.plot(list(range(len(xs2))), xs2,
             label="Adam (noisy gradient)")

    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.title("Adam's momentum smooths out noisy gradients where RMSprop jitters")
    plt.tight_layout()
    plt.show()
    print(f"RMSprop (noisy gradient) final x {x[0].item()}")
    print(f"Adam  (clean gradient) final x: {x2[0].item()}")


def plot2() -> None:
    x = torch.tensor([1.0], requires_grad=True)
    x2 = torch.tensor([1.0], requires_grad=True)
    optimizer = Adam([x], lr=0.1, weight_decay=0.05)
    optimizer.v[0] = torch.ones_like(optimizer.v[0]) * 10
    optimizer2 = Adam([x2], lr=0.1, weight_decay=0.05)
    optimizer2.v[0] = torch.ones_like(optimizer2.v[0]) * 1 / 10
    xs = []
    xs2 = []
    for i in range(60):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        _ = f(x)
        _ = f(x2)
        xs.append(x[0].item())
        xs2.append(x2[0].item())

        # y.backward()
        # y2.backward()

        optimizer.step()
        optimizer2.step()
    plt.plot(
        list(range(len(xs))),
        xs,
        label="Adam (large gradient history)",
    )
    plt.plot(
        list(range(len(xs2))),
        xs2,
        label="Adam (small gradient history)",
        color="lightblue",
    )
    plt.tight_layout()

    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.title(
        "Adam decays two parameters with the same weight_decay at different rates"
    )
    plt.show()
    print(f"Adam (large gradient history) final p: {x[0].item()}")
    print(f"Adam (small gradient history) final p: {x2[0].item()}")


def main() -> int:

    plot()
    plot2()
    return 0
