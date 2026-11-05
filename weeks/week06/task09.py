import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import AdamW


def f(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x**2


def plot() -> None:
    x = torch.tensor([1.0], requires_grad=True)
    x2 = torch.tensor([1.0], requires_grad=True)
    optimizer = AdamW([x], lr=0.1, weight_decay=0.05)
    optimizer.v[0] = torch.ones_like(optimizer.v[0]) * 10
    optimizer2 = AdamW([x2], lr=0.1, weight_decay=0.05)
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
        label="AdamW (large gradient history)",
    )
    plt.plot(
        list(range(len(xs2))),
        xs2,
        label="AdamW (small gradient history)",
        color="lightblue",
        ls="--",
    )
    plt.tight_layout()

    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.title("Adam decays both parameters by the same proportion")
    plt.show()
    print(f"AdamW (large gradient history) final p: {x[0].item()}")
    print(f"AdamW (small gradient history) final p: {x2[0].item()}")


def main() -> int:

    plot()
    return 0
