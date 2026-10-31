import torch

from dl_lib.nn.module import Module


class Sigmoid(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))
