import matplotlib.pyplot as plt
import torch

from dl_lib.nn.activations import Tanh
from dl_lib.nn.module import Module
from dl_lib.nn.normalization import BatchNorm1d
from dl_lib.nn.sequential import Sequential
from dl_lib.optim.optimizer import Optimizer
from dl_lib.optim.optimizers import SGD


class LinearLayer(Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_scale: float = 1,
        bias_scale: float = 1,
    ):
        super().__init__()
        self.W = (
            torch.randn(in_features, out_features) * weight_scale
        ).requires_grad_()
        self.bias = bias
        if self.bias:
            self.b = (torch.randn(out_features) * bias_scale).requires_grad_()
        else:
            self.b = torch.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        if self.bias:
            return [self.W, self.b]
        return [self.W]


true_w_generator = torch.Generator().manual_seed(42)
data_generator = torch.Generator().manual_seed(123)

true_w = torch.randn(10, 1, generator=true_w_generator)


def generate_dataset(
    n: int, add_noise: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 10, generator=data_generator)

    if add_noise:
        noise = 0.1 * torch.randn(n, 1, generator=data_generator)
        y = x @ true_w + noise
    else:
        y = x @ true_w
    return x, y


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean((predictions - targets) ** 2)


def evaluate(model: Module, test_X: torch.Tensor, test_y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        y_hat = model(test_X)
        loss = mse_loss(y_hat, test_y)
    return loss.item()


def train(
    model: Sequential,
    optimizer: Optimizer,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    epochs=300,
) -> tuple:

    train_losses = []
    test_losses = []
    train_losses.append(evaluate(model, train_X, train_y))
    test_losses.append(evaluate(model, test_X, test_y))
    grads = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_hat = model(train_X)
        loss = mse_loss(y_hat, train_y)
        loss.backward()
        optimizer.step()
        grads.append(
            [
                layer.W.grad.abs().mean().item() if layer.W.grad is not None else 0
                for layer in model.modules
                if isinstance(layer, LinearLayer)
            ]
        )
        train_losses.append(evaluate(model, train_X, train_y))
        test_losses.append(evaluate(model, test_X, test_y))

    return train_losses, test_losses, grads


def first_experiment():
    batch_norm = BatchNorm1d(3)
    x = torch.tensor(
        [
            [3.2, -1.0, 3.0],
            [1.5, 3.3, -4.7],
            [-2.0, 0.8, 1.2],
            [5.4, -1.6, 2.1],
        ]
    )
    w = torch.rand(3)
    y = w * x
    batch_norm.eval()
    print(f"{batch_norm.gamma = }")
    print(f"{batch_norm.beta = }")
    print(f"{x.mean(dim=0) = }")
    print(f"{x.var(dim=0, unbiased=False)=}")
    print(f"{batch_norm(x).mean(dim=0) = }")
    print(f"{batch_norm(x).var(dim=0, unbiased=False) = }")
    print(
        f"Before any forward call: {batch_norm.running_mean = }, {batch_norm.running_var = }"
    )
    optimizer = SGD(batch_norm.parameters(), 0.01)
    batch_norm.train()
    for i in range(5):
        optimizer.zero_grad()
        y_hat = batch_norm(x)
        loss = mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        batch_norm.eval()

    print(
        f"After one training call: {batch_norm.running_mean = }, {batch_norm.running_var = }"
    )
    print(f"Evaluating: {batch_norm(x).mean(dim=0) = }")


def main() -> int:
    first_experiment()
    torch.manual_seed(42)

    mlp = Sequential(
        LinearLayer(10, 32, weight_scale=0.1, bias_scale=0), BatchNorm1d(32), Tanh()
    )
    for _ in range(11):
        mlp.append(LinearLayer(32, 32, weight_scale=0.1, bias_scale=0))
        mlp.append(BatchNorm1d(32))
        mlp.append(Tanh())

    mlp.append(LinearLayer(32, 1, weight_scale=0.1, bias_scale=0))

    x_train, y_train = generate_dataset(128, add_noise=False)
    x_test, y_test = generate_dataset(
        1
    )  # I create a dummy trainig set, so that i dont have to rewrite the train function

    optimizer = SGD(mlp.parameters(), lr=0.01)
    train_losses, _, grads = train(
        mlp, optimizer, x_train, y_train, x_test, y_test, epochs=50
    )
    print(
        f"With BatchNorm1d: loss[0] = {train_losses[0]}, loss[25] = {train_losses[25]}, loss[-1] = {train_losses[-1]}"
    )
    print("With BatchNorm1d: gradient magnitude per layer =", grads[-1])

    plt.plot(range(1, len(grads[-1]) + 1), grads[-1], label="Train Loss (20 examples)")
    plt.scatter(range(1, len(grads[-1]) + 1), grads[-1])

    plt.xlabel("Layer(1 = first, 12 = last)")
    plt.ylabel("Mean |gradient| (log scale)")
    # plt.legend()
    plt.title("Gradients of a deep plain MLP")
    plt.show()
    return 0
