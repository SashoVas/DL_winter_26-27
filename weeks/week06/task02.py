from dl_lib.optim.optimizers import SGD
import torch


def main() -> int:
    param = torch.tensor([1.0, 2.0])
    param.grad = torch.tensor([1.0, 2.0])
    optimizer = SGD([param], lr=0.01)
    print(f"{param=}")
    optimizer.step()
    print(f"{param=}")
