import matplotlib.pyplot as plt
import torch

from dl_lib.nn.activations import Tanh
from dl_lib.nn.module import Module
from dl_lib.nn.sequential import Sequential
from dl_lib.optim.optimizer import Optimizer
from dl_lib.optim.optimizers import SGD
from dl_lib.nn.dropout import Dropout


class LinearLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = torch.randn(in_features, out_features, requires_grad=True)
        self.bias = bias
        if self.bias:
            self.b = torch.randn(out_features, requires_grad=True)
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


def generate_dataset(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 10, generator=data_generator)
    noise = 0.1 * torch.randn(n, 1, generator=data_generator)

    y = x @ true_w + noise

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
    model: Module,
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

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_hat = model(train_X)
        loss = mse_loss(y_hat, train_y)
        loss.backward()
        optimizer.step()

        train_losses.append(evaluate(model, train_X, train_y))
        test_losses.append(evaluate(model, test_X, test_y))

    return train_losses, test_losses


def main() -> int:
    torch.manual_seed(42)

    mlp = Sequential(
        LinearLayer(10, 64),
        Tanh(),
        Dropout(0.8),
        LinearLayer(64, 64),
        Tanh(),
        Dropout(0.8),
        LinearLayer(64, 1),
    )

    x_train, y_train = generate_dataset(20)
    x_test, y_test = generate_dataset(200)

    optimizer = SGD(mlp.parameters(), lr=0.01)

    train_losses, test_losses = train(mlp, optimizer, x_train, y_train, x_test, y_test)
    print(
        f"Final losses: train loss = {train_losses[-1]}, test loss = {test_losses[-1]}"
    )
    plt.plot(range(len(train_losses)), train_losses, label="Train loss, with Dropout")
    plt.plot(range(len(test_losses)), test_losses, label="Test loss, with Dropout")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("Dropout narrows the train/test gap from Task 01")
    plt.show()
    return 0
