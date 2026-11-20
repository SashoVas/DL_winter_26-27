import torch
from torch.utils.flop_counter import FlopCounterMode

from dl_lib.vision.functional import compute_receptive_field


def conv_flops(model: torch.nn.Module) -> int:
    counter = FlopCounterMode(display=False)
    model.eval()
    with counter, torch.no_grad():
        model(torch.randn(1, 64, 16, 16))
    return counter.get_flop_counts()["Global"][torch.ops.aten.convolution]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    three_by_three = torch.nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
    one_by_one = torch.nn.Conv2d(64, 16, kernel_size=1, stride=1)
    three_by_three_params = count_params(three_by_three)
    three_by_three_flops = conv_flops(three_by_three)
    one_by_one_params = count_params(one_by_one)
    one_by_one_flops = conv_flops(one_by_one)

    print("Changing 64 -> 16 channels on a 16x16 map:")
    print(f"  1x1 conv: {one_by_one_params:,} params, {one_by_one_flops:,} FLOPs")
    print(
        f"  3x3 conv: {three_by_three_params:,} params, {three_by_three_flops:,} FLOPs"
    )
    print(
        f"  3x3 uses {three_by_three_params / one_by_one_params}x the params and {three_by_three_flops / one_by_one_flops}x the FLOPs of 1x1."
    )
    print(
        "Receptive field after two 3x3 convs:",
        compute_receptive_field([(3, 1), (3, 1)]),
    )
    print(
        f"  + one more 1x1 conv: {compute_receptive_field([(3, 1), (3, 1), (1, 1)])} (unchanged, 1x1 only mixes channels)"
    )
    print(
        f"  + one more 3x3 conv: {compute_receptive_field([(3, 1), (3, 1), (3, 1)])} (grows, 3x3 also looks at neighbors)"
    )
    return 0
