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
                velocity *= self.momentum
                velocity += param.grad if param.grad is not None else 0

                param -= self.lr * velocity


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
                    param.grad**2 if param.grad is not None else 0
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
                    param.grad**2 if param.grad is not None else 0
                )  # we do this inplace, so there are no copies
                param -= (self.lr * (param.grad if param.grad is not None else 0)) / (
                    torch.sqrt(accumulator) + self.eps
                )


class Adam(Optimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]
        self.eps = eps
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            for param, m_i, v_i in zip(self.params, self.m, self.v):
                grad_decayed = (
                    param.grad if param.grad is not None else 0
                ) + self.weight_decay * param
                m_i *= self.beta1
                m_i += (1 - self.beta1) * grad_decayed

                v_i *= self.beta2
                v_i += (1 - self.beta2) * (grad_decayed**2)

                m_i_hat = m_i / (1 - self.beta1**self.t)
                v_i_hat = v_i / (1 - self.beta2**self.t)

                param -= self.lr * (m_i_hat) / (torch.sqrt(v_i_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        super().__init__(params, lr, beta1, beta2, eps, weight_decay)

    def step(self):
        self.t += 1
        with torch.no_grad():
            for param, m_i, v_i in zip(self.params, self.m, self.v):
                grad = param.grad if param.grad is not None else 0
                m_i *= self.beta1
                m_i += (1 - self.beta1) * grad

                v_i *= self.beta2
                v_i += (1 - self.beta2) * (grad**2)

                m_i_hat = m_i / (1 - self.beta1**self.t)
                v_i_hat = v_i / (1 - self.beta2**self.t)

                param *= 1 - self.lr * self.weight_decay
                param -= self.lr * (m_i_hat) / (torch.sqrt(v_i_hat) + self.eps)
