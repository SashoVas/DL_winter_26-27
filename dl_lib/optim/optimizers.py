import torch

from dl_lib.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params: list[torch.Tensor], lr: float, momentum: float = 0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [torch.zeros_like(param) for param in self.params]

    def step(self) -> None:
        with torch.no_grad():
            for param, velocity in zip(self.params, self.velocities):
                velocity *= self.momentum  # we do this inplace, so there are no copies
                velocity -= (
                    self.lr * param.grad if param.grad is not None else 1
                )  # we do this inplace, so there are no copies
                param += velocity


class AdaGrad(Optimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float,
        eps: float = 1e-10,
    ):
        super().__init__(params, lr)
        self.accumulators = [torch.zeros_like(param) for param in self.params]
        self.eps = eps

    def step(self) -> None:
        with torch.no_grad():
            for param, accumulator in zip(self.params, self.accumulators):
                accumulator += (
                    param.grad**2 if param.grad is not None else 1
                )  # we do this inplace, so there are no copies
                param -= (self.lr * (param.grad if param.grad is not None else 0)) / (
                    torch.sqrt(accumulator) + self.eps
                )


class RMSprop(Optimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.alpha = alpha
        self.accumulators = [torch.zeros_like(param) for param in self.params]
        self.eps = eps

    def step(self):
        with torch.no_grad():
            for param, accumulator in zip(self.params, self.accumulators):
                accumulator *= self.alpha
                accumulator += (1 - self.alpha) * (
                    param.grad**2 if param.grad is not None else 1
                )  # we do this inplace, so there are no copies
                param -= (self.lr * (param.grad if param.grad is not None else 0)) / (
                    torch.sqrt(accumulator) + self.eps
                )
