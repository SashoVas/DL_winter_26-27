import torch

from dl_lib.nn.activations import ReLU
from dl_lib.nn.module import Module
from dl_lib.nn.sequential import Sequential


class Transformation(Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.W = torch.randn(input_dim, output_dim)
        self.b = torch.randn(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.W) + self.b

    def parameters(self) -> list[torch.Tensor]:
        return [self.W, self.b]


def main() -> int:
    sequential = Sequential(Transformation(3, 4), ReLU(), Transformation(4, 2))
    params = sequential.parameters()
    print("number of learnable tensors:", len(params))
    for i, param in enumerate(params):
        print(f"params[{i}].shape = {param.shape}")
    return 0
