import torch

from dl_lib.nn.module import Module


class Sigmoid(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))


class Tanh(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


class ReLU(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))
