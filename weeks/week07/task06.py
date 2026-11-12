import matplotlib.pyplot as plt
import torch

from dl_lib.nn.normalization import BatchNorm1d


def main() -> int:
    torch.manual_seed(42)
    batch_norm = BatchNorm1d(1)
    x_fixed = torch.randn(1, 1)
    to_plot = []
    for batch_size in [2, 8, 64]:
        outputs = []
        for _ in range(50):
            companions = torch.randn(batch_size - 1, 1)
            batch = torch.cat([x_fixed, companions], dim=0)
            with torch.no_grad():
                normalized = batch_norm(batch)
            outputs.append(normalized[0, 0].item())
        std = torch.tensor(outputs).std().item()
        print(
            f"Batch size {batch_size}: std of BatchNorm1d(x_fixed) across 30 random batches = {std}"
        )
        to_plot.append(std)

    plt.plot([2, 8, 64], to_plot)
    plt.scatter([2, 8, 64], to_plot)
    plt.xlabel("Batch size")
    plt.ylabel("Std of normalized output for the same example")
    plt.title("BatchNorm1d's output relative to size of batch")
    plt.show()

    return 0
