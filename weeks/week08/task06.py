import torch
from torch.nn import MaxPool2d

from dl_lib.vision.pooling import MaxPool2d as MaxPool2dMy


def test(config: dict) -> None:
    my_conv = MaxPool2dMy(
        config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
    )
    real_conv = MaxPool2d(
        config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
    )

    input = torch.randn(2, 96, 64, 64)

    with torch.no_grad():
        my_output = my_conv(input)
        real_output = real_conv(input)
    print("Config:", config)
    print(f"Output shape: {my_output.shape}")
    print(
        "Matches torch.nn.MaxPool2d:", torch.allclose(my_output, real_output, atol=1e-5)
    )


def main() -> int:
    test({"kernel_size": 2, "stride": 2, "padding": 0})
    test({"kernel_size": 3, "stride": 1, "padding": 1})
    test({"kernel_size": 3, "stride": 2, "padding": 0})
    print("The last config is AlexNet's real conv1:")
    print("in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2.")
    return 0
