import torch

from dl_lib.nn.module import Module
from dl_lib.vision.functional import load_cifar10_loaders
from dl_lib.vision.resnet import ResNet


def count_params(model: Module | torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    train_dataloader, _, _, _ = load_cifar10_loaders(4)
    images, labels = next(iter(train_dataloader))
    print(f"Batch shape: {images.shape}, labels: {labels}")
    for depth in [20, 32, 56]:
        net = ResNet(depth=depth, global_avg_pool=False)
        output = net(images)
        print(
            f"depth={depth}: logits={output.shape} total_params={count_params(net):,} head_params={count_params(net.head):,}"
        )
    return 0
