import torch

from dl_lib.nn.sequential import Sequential
from dl_lib.vision.conv import Conv2d
from dl_lib.vision.functional import compute_receptive_field


def main() -> int:
    seven_by_seven_receptive_field = compute_receptive_field([(7, 1)])
    three_by_three_receptive_field = compute_receptive_field(3 * [(3, 1)])
    print("Receptive field of one 7x7 conv:", seven_by_seven_receptive_field)
    print("Receptive field of three stacked 3x3s:", three_by_three_receptive_field)
    seven_by_seven_params = 7 * 7 * 64 * 64 + 64  # plus bias
    three_by_three_params = 3 * (3 * 3 * 64 * 64 + 64)  # plus bias
    print("Params in one 7x7 conv (64 -> 64 channels):", seven_by_seven_params)
    print(
        "Params in three stacked 3x3 convs (64 -> 64 channels)", three_by_three_params
    )
    print(
        f"Stacking 3x3s uses {seven_by_seven_params / three_by_three_params}x fewer parameters for the same receptive field."
    )
    seven_by_seven_network = Conv2d(64, 64, 7, padding=3)
    three_by_three_network = Sequential(
        Conv2d(64, 64, 3, 1, 1), Conv2d(64, 64, 3, 1, 1), Conv2d(64, 64, 3, 1, 1)
    )
    input = torch.randn(2, 64, 32, 32)
    print("Output shape of the 7x7 conv:", seven_by_seven_network(input).shape)
    print("Output shape of the stacked 3x3s:", three_by_three_network(input).shape)
    return 0
