import torch

from dl_lib.nn.module import Module


class Linear(Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super().__init__()
        self.weight = torch.randn(out_features, in_features, requires_grad=True)

        self.bias = torch.randn(out_features, requires_grad=True)

    def forward(self, x):
        return x @ self.weight.T + self.bias

    def parameters(self):
        return [self.weight, self.bias]

    def to(self, device):
        self.weight = self.weight.detach().to(device).requires_grad_(True)
        self.bias = self.bias.detach().to(device).requires_grad_(True)
        return self
