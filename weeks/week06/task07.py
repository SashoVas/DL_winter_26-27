import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import AdaGrad, RMSprop


def f(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x**2


def plot() -> None:
    x = torch.tensor([5.0], requires_grad=True)
    x2 = torch.tensor([5.0], requires_grad=True)
    optimizer = AdaGrad([x], lr=0.5)
    optimizer2 = RMSprop([x2], lr=0.5)
    xs = []
    xs2 = []

    for i in range(150):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        y = f(x)
        y2 = f(x2)
        xs.append(x[0].item())
        xs2.append(x2[0].item())

        y.backward()
        y2.backward()
        if i == 0 and x.grad is not None and x2.grad is not None:
            x.grad *= 100
            x2.grad *= 100
        optimizer.step()
        optimizer2.step()
    plt.plot(
        list(range(len(xs))),
        xs,
        label="AdaGrad (early spike)",
        color="green",
    )
    plt.plot(list(range(len(xs2))), xs2, label="RMSprop (early spike)", color="purple")
    plt.title("RMSprop recovers from an early gradient spike where AdaGrad stalls")
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"AdaGrad (early spike) final x: {x[0].item()}")
    print(f"RMSprop  (early spike) final x: {x2[0].item()}")


def plot2() -> None:
    x = torch.tensor([5.0], requires_grad=True)
    x2 = torch.tensor([5.0], requires_grad=True)
    optimizer = RMSprop([x], lr=0.1)
    optimizer2 = RMSprop([x2], lr=0.1)
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
        if x.grad is not None:
            x.grad += torch.randn_like(x.grad)
        optimizer.step()
        optimizer2.step()
    plt.plot(
        list(range(len(xs))),
        xs,
        label="RMSprop (noisy gradient)",
        color="red",
    )
    plt.plot(
        list(range(len(xs2))), xs2, label="RMSprop (clean gradient)", color="purple"
    )
    plt.title("RMSprop jitters around the minimum under noisy gradients")
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()

    plt.show()
    print(f"RMSprop (noisy gradient) final x {x[0].item()}")
    print(f"RMSprop (clean gradient) final x: {x2[0].item()}")


def main() -> int:

    plot()
    plot2()
    return 0
