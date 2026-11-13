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

    input = torch.randn(2, config["in_channels"], 64, 64)

    with torch.no_grad():
        my_output = my_conv(input)
        real_output = real_conv(input)
    print("Config:", config)
    print(f"Output shape: {my_output.shape}")
    print("Matches torch.nn.Conv2d:", torch.allclose(my_output, real_output, atol=1e-5))


def main() -> int:
    test(
        {
            "in_channels": 3,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        }
    )
    test(
        {
            "in_channels": 3,
            "out_channels": 4,
            "kernel_size": 5,
            "stride": 2,
            "padding": 2,
        }
    )
    test(
        {
            "in_channels": 3,
            "out_channels": 96,
            "kernel_size": 11,
            "stride": 4,
            "padding": 2,
        }
    )
    print("The last config is AlexNet's real conv1:")
    print("in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2.")
    return 0
