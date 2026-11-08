import torch

from dl_lib.optim.optimizer import Optimizer
from dl_lib.optim.optimizers import SGD, AdaGrad, Adam, AdamW, RMSprop


def f(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return 2.5 * x - 1.0 + torch.randn_like(x, generator=generator)


def train(
    w: torch.Tensor,
    b: torch.Tensor,
    optimizer: Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
):
    for _ in range(epochs):
        optimizer.zero_grad()
        y_hat = w * x + b
        loss = torch.nn.functional.mse_loss(y, y_hat)
        loss.backward()
        optimizer.step()

    return loss


EPOCHS = 50
OPTIMIZERS = [
    (SGD, {"lr": 0.01}, "SGD", "plain gradient descent, no momentum"),
    (
        AdaGrad,
        {"lr": 0.5},
        "AdaGrad",
        "per-parameter learning rate, needs a bigger lr here",
    ),
    (RMSprop, {"lr": 0.05}, "RMSprop", "decayed accumulator, fastest to converge here"),
    (
        Adam,
        {"lr": 0.1, "weight_decay": 0.01},
        "Adam",
        "L2-style decay folded into the gradient",
    ),
    (
        AdamW,
        {"lr": 0.1, "weight_decay": 0.01},
        "AdamW",
        "decoupled decay, differs from Adam above",
    ),
]


def main() -> int:
    generator = torch.Generator().manual_seed(42)
    x = torch.linspace(-3, 3, 100)
    y = f(x, generator)
    baseline_loss = torch.nn.functional.mse_loss(y, torch.mean(y))
    initial_w = torch.randn(1, generator=generator)
    initial_b = torch.zeros(1)
    res = []
    for optimizer_class, params, optimizer_name, comment in OPTIMIZERS:
        w = initial_w.clone().requires_grad_()
        b = initial_b.clone().requires_grad_()
        optimizer = optimizer_class([w, b], **params)
        loss = train(w, b, optimizer, x, y, epochs=EPOCHS)
        vs_baseline = (loss / baseline_loss - 1) * 100
        res.append((optimizer_name, loss, vs_baseline, comment))

    print(f"{'optimizer':<10} {'final loss':>12} {'vs baseline':>12}  comment")
    print(
        f"{'baseline':<10} {baseline_loss:>12.5f} "
        f"{0.0:>+11.1f}%  mean predictor, ignores x"
    )

    for name, loss, change, comment in res:
        print(f"{name:<10} {loss:>12.5f} {change:>+11.1f}%  {comment}")

    best_name, best_loss, _, _ = min(res, key=lambda result: result[1])
    print(f"Best optimizer: {best_name} (final loss = {best_loss:.5f})")
    return 0
