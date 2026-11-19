import matplotlib.pyplot as plt
import torch

from dl_lib.nn.init import kaiming_normal_
from dl_lib.vision.functional import load_cifar10_loaders
from dl_lib.vision.vgg import Vgg


def relu_activation_stds(model: Vgg, images: torch.Tensor):
    stds = []
    x = images
    with torch.no_grad():
        for layer in model.features:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                stds.append(x.std().item())
    return stds


def main() -> int:
    vgg = Vgg()
    vgg_initialized = Vgg()
    for layer in vgg_initialized.features:
        if not isinstance(layer, torch.nn.Conv2d):
            continue

        kaiming_normal_(layer.weight, nonlinearity="relu")

    dataloader_train, _, _, _ = load_cifar10_loaders(batch_size=4)
    images, labels = next(iter(dataloader_train))
    criterion = torch.nn.CrossEntropyLoss()
    loss_vgg = criterion(vgg(images), labels)
    loss_vgg_initialized = criterion(vgg_initialized(images), labels)
    loss_vgg.backward()
    loss_vgg_initialized.backward()

    vgg_grads = [
        layer.weight.grad.abs().mean().item()
        for layer in vgg.features
        if isinstance(layer, torch.nn.Conv2d) and layer.weight.grad is not None
    ]
    vgg_initialized_grads = [
        layer.weight.grad.abs().mean().item()
        for layer in vgg_initialized.features
        if isinstance(layer, torch.nn.Conv2d) and layer.weight.grad is not None
    ]
    vgg_relu_stds = relu_activation_stds(vgg, images)
    vgg_initialized_stds = relu_activation_stds(vgg_initialized, images)
    print("Activation std per ReLU block:")
    print(f"block 1: default={vgg_relu_stds[0]}  kaiming={vgg_initialized_stds[0]}")
    print("Mean |gradient| per conv layer:")
    print(f"layer 1: default={vgg_grads[0]}  kaiming={vgg_initialized_grads[0]}")
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(vgg_grads) + 1),
        vgg_grads,
        marker="o",
        label="PyTorch default init",
    )
    plt.plot(
        range(1, len(vgg_grads) + 1),
        vgg_initialized_grads,
        marker="o",
        label="Kaiming init (relu)",
    )
    plt.yscale("log")
    plt.xlabel("conv layer depth (1 = closest to input)")
    plt.ylabel("mean |gradient| (log scale)")
    plt.title("Gradient magnitude vs depth: default vs Kaiming init")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return 0
