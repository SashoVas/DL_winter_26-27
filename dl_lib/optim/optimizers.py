from dl_lib.optim.optimizer import Optimizer
import torch


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad
