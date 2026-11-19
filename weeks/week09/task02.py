import torch

from dl_lib.nn.module import Module
from dl_lib.vision.functional import compute_receptive_field, load_cifar10_loaders
from dl_lib.vision.vgg import Vgg


def count_params(model: torch.nn.Module | Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    vgg = Vgg()
    dataloader_train, _, _, classes = load_cifar10_loaders(batch_size=4)
    images, labels = next(iter(dataloader_train))
    print("Classes (10):", classes)
    print(f"Batch shape: {images.shape}, labels: {labels}")
    layers = []
    res = 64
    out_channels = 3
    for layer in vgg.features:
        if isinstance(layer, torch.nn.Conv2d):
            layers.append((3, 1))
            out_channels = layer.out_channels
        elif isinstance(layer, torch.nn.MaxPool2d):
            res = res / 2
            layers.append((2, 2))
            print(
                f"After MaxPool2d    shape=(4, {out_channels}, {res}, {res}) receptive_field={compute_receptive_field(layers)}"
            )
    output = vgg(images)
    print("Logits shape:", output.shape)
    print("Params in features:", count_params(vgg.features))
    print("Params in classifier:", count_params(vgg.classifier))
    print("Total params:", count_params(vgg))
    return 0
