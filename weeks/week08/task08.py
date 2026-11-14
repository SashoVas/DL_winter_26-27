import torch

from dl_lib.vision.alexnet import AlexNet


def main() -> None:
    model = AlexNet()
    input = torch.randn(4, 3, 64, 64)
    print("Input shape:", input.shape)
    print("features(x) output shape:", model.features(input).shape)
