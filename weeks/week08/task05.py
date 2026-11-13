from time import time

import torch
from torch.nn import Conv2d

from dl_lib.vision.conv import Conv2d as Conv2dMy


def test(config: dict) -> None:
    my_conv = Conv2dMy(
        config["in_channels"],
        config["out_channels"],
        config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
    )
    real_conv = Conv2d(
        config["in_channels"],
        config["out_channels"],
        config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
    )
    assert real_conv.weight is not None
    assert real_conv.bias is not None
    my_conv.weight = real_conv.weight
    my_conv.bias = real_conv.bias

    input = torch.randn(4, config["in_channels"], 64, 64)
    print("Input shape:", input.shape)

    with torch.no_grad():
        my_output = my_conv(input)
        real_output = real_conv(input)

    with torch.no_grad():
        start = time()
        my_output = my_conv(input)
        end = time()
        our = end - start
        print(f"Our Conv2d: {our:.4f}s")
        start = time()
        real_output = real_conv(input)
        end = time()
        their = end - start
        print(f"torch.nn.Conv2d: {their:.4f}s")

    print("Speedup:", our / their)
    print("Matches torch.nn.Conv2d:", torch.allclose(my_output, real_output, atol=1e-5))


def main() -> int:
    test(
        {
            "in_channels": 3,
            "out_channels": 96,
            "kernel_size": 11,
            "stride": 4,
            "padding": 2,
        }
    )
    return 0
