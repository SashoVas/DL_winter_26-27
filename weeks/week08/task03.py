import torch
from torch.nn import Conv1d

from dl_lib.vision.conv import Conv1d as Conv1dMy


def test(config: dict) -> None:
    my_conv = Conv1dMy(
        3, 4, config["kernel_size"], stride=config["stride"], padding=config["padding"]
    )
    real_conv = Conv1d(
        3, 4, config["kernel_size"], stride=config["stride"], padding=config["padding"]
    )
    assert real_conv.weight is not None
    assert real_conv.bias is not None
    my_conv.weight = real_conv.weight
    my_conv.bias = real_conv.bias

    input = torch.randn(2, 3, 20)

    with torch.no_grad():
        my_output = my_conv(input)
        real_output = real_conv(input)
    print("Config:", config)
    print(
        f"Computed output length: {my_output.shape[-1]}, reported: {real_output.shape[-1]}"
    )
    print("Matches torch.nn.Conv1d:", torch.allclose(my_output, real_output))


def main() -> int:
    test({"kernel_size": 3, "stride": 2, "padding": 0})
    test({"kernel_size": 3, "stride": 1, "padding": 2})
    test({"kernel_size": 4, "stride": 3, "padding": 2})
    return 0
