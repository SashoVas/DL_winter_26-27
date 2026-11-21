import torch

from dl_lib.nn.module import Module
from dl_lib.vision.functional import load_cifar10_loaders
from dl_lib.vision.resnet import ResNet


def count_params(model: Module | torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    train_dataloader, _, _, _ = load_cifar10_loaders(4)
    images, _ = next(iter(train_dataloader))
    for depth in [20, 56]:
        print(f"depth={depth}:")
        net_flatten = ResNet(depth=depth, global_avg_pool=False)
        net_pool = ResNet(depth=depth, global_avg_pool=True)
        output_flatten = net_flatten(images)
        output_pool = net_pool(images)
        print(
            f"  flatten+Linear head: total_params={count_params(net_flatten):,} head_params={count_params(net_flatten.head):,} logits={output_flatten.shape}"
        )
        print(
            f"  flatten+Linear head: total_params={count_params(net_pool):,} head_params={count_params(net_pool.head):,} logits={output_pool.shape}"
        )
        print(
            f"  head params shrink {count_params(net_flatten.head) / count_params(net_pool.head)}x, total params shrink {count_params(net_flatten) / count_params(net_pool)}x"
        )
    return 0
