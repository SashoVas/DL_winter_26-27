import matplotlib.pyplot as plt
import torch
import random
from dl_lib.optim.optimizers import SGD


def f(x: torch.Tensor) -> torch.Tensor:
    return x**4 + x**3 - 5 * x**2


def run(lr: float, momentum: float = 0) -> None:
    x = torch.tensor([2.0], requires_grad=True)
    optimizer = SGD([x], lr=lr, momentum=momentum)
    xs = []
    for _ in range(50):
        optimizer.zero_grad()
        y = f(x)
        xs.append(x[0].item())
        y.backward()
        optimizer.step()
    return x[0].item(), y[0].item()


def main() -> int:
    print(f"{'lr':>8} {'momentum':>10} {'final x':>10} {'f(x)':>10}")
    rand = random.Random(42)
    best = None
    for i in range(8):
        lr = rand.uniform(0.02, 0.05)
        momentum = rand.uniform(0.1, 1.3)
        x_final, f_x = run(lr, momentum)
        print(f"{lr:8.4f} {momentum:10.4f} {x_final:10.4f} {f_x:10.4f}")
        if best is None or f_x < best[0]:
            best = (f_x, lr, momentum, x_final)
    f_x, lr, momentum, x_final = best
    print(
        f"Best config: lr={lr:.4f}, momentum={momentum:.4f} "
        f"-> x={x_final:.4f}, f(x)={f_x:.4f}"
    )
    return 0
