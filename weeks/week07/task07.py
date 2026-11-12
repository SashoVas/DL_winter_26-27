import matplotlib.pyplot as plt
import torch

from dl_lib.nn.normalization import LayerNorm


def first_experiment() -> None:
    layer_norm = LayerNorm(4)
    x = torch.tensor(
        [
            [4.0820, 0.4131, -3.3576, 2.1369],
            [-1.1690, -1.7972, 1.8067, 2.6761],
            [-0.4385, 0.1933, -0.1933, 1.3641],
        ]
    )
    print(f"{x = }")
    layer_norm.train()
    training = layer_norm(x)
    layer_norm.eval()
    evaluating = layer_norm(x)
    print(f"Training:   layer_norm(x) = {training}")
    print(f"Evaluating:   layer_norm(x) = {evaluating}")
    print("Identical regardless of mode:", torch.allclose(training, evaluating))


def main() -> int:
    first_experiment()
    torch.manual_seed(42)
    layer_norm = LayerNorm(1)
    x_fixed = torch.randn(1, 1)
    to_plot = []
    for batch_size in [2, 8, 64]:
        outputs = []
        for _ in range(50):
            companions = torch.randn(batch_size - 1, 1)
            batch = torch.cat([x_fixed, companions], dim=0)
            with torch.no_grad():
                normalized = layer_norm(batch)
            outputs.append(normalized[0, 0].item())
        std = torch.tensor(outputs).std().item()
        print(
            f"Batch size {batch_size}: std of LayerNorm(x_fixed) across 30 random batches = {std}"
        )
        to_plot.append(std)

    plt.plot([2, 8, 64], to_plot)
    plt.scatter([2, 8, 64], to_plot)
    plt.xlabel("Batch size")
    plt.ylabel("Std of normalized output for the same example")
    plt.title("LayerNorm's output for one example ignores its batchmates entirely")
    plt.show()

    return 0
