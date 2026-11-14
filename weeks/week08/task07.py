from time import time

import torch
from torch.nn import Linear

from dl_lib.nn.linear import Linear as LinearMy


def set_weights(my: LinearMy, other: Linear) -> None:
    assert other.weight is not None
    assert other.bias is not None
    my.weight = other.weight
    my.bias = other.bias


def test() -> None:
    my_first = LinearMy(256, 1024)
    my_second = LinearMy(1024, 1024)
    my_third = LinearMy(1024, 10)

    other_first = Linear(256, 1024)
    other_second = Linear(1024, 1024)
    other_third = Linear(1024, 10)
    set_weights(my_first, other_first)
    set_weights(my_second, other_second)
    set_weights(my_third, other_third)
    input = torch.rand(4, 256)
    with torch.no_grad():
        first_output_my = my_first(input)
        second_output_my = my_second(first_output_my)
        third_output_my = my_third(second_output_my)

        first_output_other = other_first(input)
        second_output_other = other_second(first_output_other)
        third_output_other = other_third(second_output_other)

    print("Config: in_features=256, out_features=1024")
    print("Output shape:", first_output_my.shape)
    print(
        "Matches torch.nn.Linear",
        torch.allclose(first_output_my, first_output_other, atol=1e-5),
    )

    print("Config: in_features=1024, out_features=1024")
    print("Output shape:", second_output_my.shape)
    print(
        "Matches torch.nn.Linear",
        torch.allclose(second_output_my, second_output_other, atol=1e-5),
    )

    print("Config: in_features=1024, out_features=10")
    print("Output shape:", third_output_my.shape)
    print(
        "Matches torch.nn.Linear",
        torch.allclose(third_output_my, third_output_other, atol=1e-5),
    )
    print(
        "These three shapes are AlexNet's real classifier: 256->1024, 1024->1024, 1024->10."
    )
    with torch.no_grad():
        start = time()
        first_output_my = my_first(input)
        end = time()
        my = end - start
        print(f"Our Linear: {my:.4f}s")
        start = time()
        first_output_other = other_first(input)
        end = time()
        other = end - start
        print(f"torch.nn.Linear: {other:.4f}s")

    print("Speedup:", my / other)
    print(
        "Matches torch.nn.Linear:",
        torch.allclose(first_output_my, first_output_other, atol=1e-5),
    )


def main() -> int:
    test()
    return 0
