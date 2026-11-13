from dl_lib.vision.conv import Conv1d as Conv1dMy
from torch.nn import Conv1d
import torch


def main() -> int:
    my_conv = Conv1dMy(3, 4, 5)
    real_conv = Conv1d(3, 4, 5)
    my_conv.weight = real_conv.weight
    my_conv.bias = real_conv.bias

    input = torch.randn(2,  3, 20)
    print("Copied torch.nn.Conv1d's weight and bias into our Conv1d.")
    print("Input shape:", input.shape)
    with torch.no_grad():
        my_output = my_conv(input)
        real_output = real_conv(input)

    print("Our output shape:", my_output.shape)
    print("Matches torch.nn.Conv1d:", torch.allclose(my_output, real_output))
    return 0
