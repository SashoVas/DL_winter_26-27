import torch

from dl_lib.nn.module import Module
from dl_lib.vision.functional import compute_receptive_field, load_cifar10_loaders
from dl_lib.vision.vgg import Vgg
import matplotlib.pyplot as plt


def main() -> int:
    vgg = Vgg()
    vgg_batched = Vgg(batch_norm=True)
    dataloader_train, _, _, _ = load_cifar10_loaders(batch_size=4)
    images, labels = next(iter(dataloader_train))
    criterion = torch.nn.CrossEntropyLoss()
    loss_vgg = criterion(vgg(images), labels)
    loss_vgg_batched = criterion(vgg_batched(images), labels)
    loss_vgg.backward()
    loss_vgg_batched.backward()

    vgg_grads = [
        layer.weight.grad.abs().mean().item()
        for layer in vgg.features
        if isinstance(layer, torch.nn.Conv2d) and layer.weight.grad is not None
    ]
    vgg_batched_grads = [
        layer.weight.grad.abs().mean().item()
        for layer in vgg_batched.features
        if isinstance(layer, torch.nn.Conv2d) and layer.weight.grad is not None
    ]

    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(vgg_grads) + 1), vgg_grads, marker="o", label="batch_norm=False"
    )
    plt.plot(
        range(1, len(vgg_grads) + 1),
        vgg_batched_grads,
        marker="o",
        label="batch_norm=True",
    )
    plt.yscale("log")
    plt.xlabel("conv layer depth (1 = closest to input)")
    plt.ylabel("mean |gradient| (log scale)")
    plt.title(f"Gradient magnitude vs depth in an {len(vgg_grads)}-conv VGG")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return 0
