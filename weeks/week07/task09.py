import torch

from dl_lib.nn.activations import Tanh
from dl_lib.nn.dropout import Dropout
from dl_lib.nn.module import Module
from dl_lib.nn.normalization import BatchNorm1d, LayerNorm
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


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    return (1 - ss_res / ss_tot).item()


def print_res(results: list[tuple[str, float, str]]):
    print(f"{'model':<17}{'test r2':>8}  comment")
    for name, r2, comment in results:
        print(f"{name:<17}{r2:>8.4f}  {comment}")

    best_name, best_r2, _ = max(results, key=lambda row: row[1])
    print(f"Best model: {best_name} (test r2 = {best_r2:.4f})")


def build_models() -> dict[str, tuple[Module, str]]:
    return {
        "plain MLP": (
            Sequential(
                LinearLayer(10, 64),
                Tanh(),
                LinearLayer(64, 64),
                Tanh(),
                LinearLayer(64, 1),
            ),
            "no regularization, overfits the 20-example train set",
        ),
        "+Dropout": (
            Sequential(
                LinearLayer(10, 64),
                Tanh(),
                Dropout(),
                LinearLayer(64, 64),
                Tanh(),
                Dropout(),
                LinearLayer(64, 1),
            ),
            "randomly zeroes activations, narrows the train/test gap",
        ),
        "+BatchNorm1d": (
            Sequential(
                LinearLayer(10, 64),
                BatchNorm1d(64),
                Tanh(),
                LinearLayer(64, 64),
                BatchNorm1d(64),
                Tanh(),
                LinearLayer(64, 1),
            ),
            "normalizes over the batch, mainly a training stabilizer here",
        ),
        "+LayerNorm": (
            Sequential(
                LinearLayer(10, 64),
                LayerNorm(64),
                Tanh(),
                LinearLayer(64, 64),
                LayerNorm(64),
                Tanh(),
                LinearLayer(64, 1),
            ),
            "normalizes per sample, stable even with this tiny batch",
        ),
    }


def main() -> int:
    torch.manual_seed(42)

    x_train, y_train = generate_dataset(20)
    x_test, y_test = generate_dataset(200)
    baseline_predictions = torch.full_like(y_test, y_train.mean().item())
    results: list[tuple[str, float, str]] = [
        (
            "baseline",
            r2_score(baseline_predictions, y_test),
            "mean predictor, ignores x",
        )
    ]
    for model_name, (model, comment) in build_models().items():
        optimizer = SGD(model.parameters(), lr=0.01)

        _, _ = train(model, optimizer, x_train, y_train, x_test, y_test)
        model.eval()
        with torch.no_grad():
            results.append((model_name, r2_score(model(x_test), y_test), comment))

    print_res(results)
    return 0
