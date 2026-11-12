import torch

from dl_lib.nn.module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p > 1 or p < 0:
            raise ValueError("Probability of dropout should be between 0 and 1")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        if self.p == 1:
            return x * 0

        mask = torch.rand_like(x) > self.p

        return (x * mask) / (1 - self.p)
