import torch

from dl_lib.vision.resnet import ResidualBlock


def main() -> int:
    torch.manual_seed(42)
    block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
    with torch.no_grad():
        block.conv2.weight.zero_()
        if block.conv2.bias is not None:
            block.conv2.bias.zero_()
        block.bn2.weight.zero_()
        block.bn2.bias.zero_()
    block.eval()
    x = torch.rand(8, 16, 32, 32)

    with torch.no_grad():
        # with shortcut
        residual_out = block.forward(x)

        # with no shortcut
        plain_out = block.relu(
            block.bn2(block.conv2(block.relu(block.bn1(block.conv1(x)))))
        )

    residual_diff = (residual_out - x).abs().mean().item()
    plain_diff = (plain_out - x).abs().mean().item()

    print("With conv2/bn2 zeroed, F(x) starts at 0 for both paths:")
    print(
        f"  mean |F(x) + x - x| (residual block output vs input): {residual_diff:.6f}"
    )
    print(f"  mean |H(x) - x|     (plain block output vs input):    {plain_diff:.6f}")

    return 0
