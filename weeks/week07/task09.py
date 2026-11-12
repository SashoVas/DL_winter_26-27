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


class Identity(Module):
    def forward(self, x):
        return x


class MyModel(Module):
    def __init__(
        self,
        activation: str = "tanh",
        dropout: bool = False,
        norm: str = "batch",
        num_layers: int = 2,
        skip_connections: bool = False,
        hidden_size: int = 64,
        scale: float = 1.0,
    ):
        super().__init__()
        self.layers = []
        self.skip_connections = skip_connections
        self.initial_layer = Sequential(
            LinearLayer(10, hidden_size, weight_scale=scale, bias_scale=scale),
            BatchNorm1d(hidden_size)
            if norm == "batch"
            else LayerNorm(hidden_size)
            if norm == "layer"
            else Identity(),
            Tanh() if activation == "tanh" else torch.nn.ReLU(),
            Dropout() if dropout else Identity(),
        )
        self.output_layer = LinearLayer(
            hidden_size, 1, weight_scale=scale, bias_scale=scale
        )
        for _ in range(num_layers):
            self.layers.append(
                Sequential(
                    LinearLayer(
                        hidden_size, hidden_size, weight_scale=scale, bias_scale=scale
                    ),
                    BatchNorm1d(hidden_size)
                    if norm == "batch"
                    else LayerNorm(hidden_size)
                    if norm == "layer"
                    else Identity(),
                    Tanh() if activation == "tanh" else torch.nn.ReLU(),
                    Dropout() if dropout else Identity(),
                )
            )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.layers:
            if self.skip_connections:
                x = x + layer(x)
            else:
                x = layer(x)
        return self.output_layer(x)

    def parameters(self):
        params = self.initial_layer.parameters()
        params += self.output_layer.parameters()
        for layer in self.layers:
            params += layer.parameters()
        return params


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
    early_stopping: bool = False,
    patience: int = 10,
) -> tuple:

    train_losses = []
    test_losses = []
    train_losses.append(evaluate(model, train_X, train_y))
    test_losses.append(evaluate(model, test_X, test_y))
    best_loss = test_losses[-1]
    epochs_without_improvement = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_hat = model(train_X)
        loss = mse_loss(y_hat, train_y)
        loss.backward()
        optimizer.step()

        train_losses.append(evaluate(model, train_X, train_y))
        test_losses.append(evaluate(model, test_X, test_y))
        if early_stopping:
            if test_losses[-1] < best_loss:
                best_loss = test_losses[-1]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {len(train_losses)} epochs. Best test loss: {best_loss:.4f}"
                )
                break

    return train_losses, test_losses


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    return (1 - ss_res / ss_tot).item()


def print_res(results: list[tuple[str, float, str]]):
    print("'model', 'test r2'")
    for name, r2, comment in results:
        print(f"{name}, {r2}")
        # print(f"{r2}")

    best_name, best_r2, _ = max(results, key=lambda row: row[1])
    print(f"Best model: {best_name} (test r2 = {best_r2:.4f})")


def build_models() -> dict[str, tuple[Module, str]]:

    models: dict[str, tuple[Module, str]] = {}
    for activation in ["tanh"]:  # , "relu"]:
        for dropout in [True, False]:
            for norm in ["batch", "layer"]:
                for skip_connections in [True]:  # , False]:
                    for scale in [0.1, 0.2, 1 / (10**0.5)]:  # , 1, 10]:
                        for hidden_size in [4, 8]:  # , 16, 32, 64]:
                            model_name = f"{activation} {'+Dropout' if dropout else ''} {'+BatchNorm1d' if norm == 'batch' else '+LayerNorm' if norm == 'layer' else ''} {'+skip connections' if skip_connections else ''} (scale={scale}) layers={1} hidden_size={hidden_size}".strip()
                            models[model_name] = (
                                MyModel(
                                    activation=activation,
                                    dropout=dropout,
                                    norm=norm,
                                    skip_connections=skip_connections,
                                    scale=scale,
                                    hidden_size=hidden_size,
                                ),
                                f"{activation} {'with dropout' if dropout else 'without dropout'} {'with batch norm' if norm == 'batch' else 'with layer norm' if norm == 'layer' else 'without normalization'} {'with skip connections' if skip_connections else 'without skip connections'} (scale={scale}) layers={1} hidden_size={hidden_size}",
                            )
    return models


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
        optimizer = SGD(model.parameters(), lr=0.05)

        _, _ = train(
            model,
            optimizer,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=300,
            early_stopping=True,
            patience=10,
        )
        model.eval()
        with torch.no_grad():
            results.append((model_name, r2_score(model(x_test), y_test), comment))

    print_res(results)
    return 0
