from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Optimizer(ABC):
    def __init__(self, params: list[torch.Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = torch.zeros_like(param)

    @abstractmethod
    def step(self):  # pragma: no cover
        pass
