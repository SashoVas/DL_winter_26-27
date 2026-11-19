from collections.abc import Callable

import matplotlib.pyplot as plt
import torch

from dl_lib.nn.init import calculate_gain, kaiming_normal_, xavier_normal_


def get_network(
    init_func: Callable,
) -> torch.nn.Sequential:
    torch.manual_seed(42)
    network = torch.nn.Sequential(
        *[
            torch.nn.Linear(256, 256) if i % 2 == 0 else torch.nn.Tanh()
            for i in range(20)
        ]
    )
    for layer in network:
        if not isinstance(layer, torch.nn.Linear):
            continue

        init_func(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    return network


def custom_init(tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(0, 0.01)


def tanh_activation_stds(model: torch.nn.Sequential, x: torch.Tensor):
    stds = []
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if isinstance(layer, torch.nn.Tanh):
                stds.append(x.std().item())
    return stds


def main() -> int:
    kaiming = get_network(lambda x,: kaiming_normal_(x, nonlinearity="relu"))
    xavier = get_network(lambda x: xavier_normal_(x, nonlinearity="tanh"))
    custom = get_network(custom_init)

    x = torch.randn(256, 256)
    kaiming_stds = tanh_activation_stds(kaiming, x)
    xavier_stds = tanh_activation_stds(xavier, x)
    custom_stds = tanh_activation_stds(custom, x)
    tanh_gain = calculate_gain("tanh")
    print("Tanh gain used for Xavier:", tanh_gain)
    print("Activation std per tanh layer, a 10-layer Linear+Tanh stack:")
    print(
        f"  layer  1: kaiming(relu-tuned)={kaiming_stds[0]}  xavier(tanh-tuned)={xavier_stds[0]}  too_small(std=0.01)={custom_stds[0]}"
    )
    print(
        f"  layer  10: kaiming(relu-tuned)={kaiming_stds[-1]}  xavier(tanh-tuned)={xavier_stds[-1]}  too_small(std=0.01)={custom_stds[-1]}"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(kaiming_stds) + 1),
        kaiming_stds,
        marker="o",
        label="Kaiming init (relu gain)",
    )
    plt.plot(
        range(1, len(xavier_stds) + 1),
        xavier_stds,
        marker="o",
        label="Xavier init (tanh gain)",
    )
    plt.plot(
        range(1, len(custom_stds) + 1),
        custom_stds,
        marker="o",
        label="too-small fixed init (std=0.01)",
    )
    plt.plot(
        range(1, len(custom_stds) + 1),
        [tanh_gain] * len(custom_stds),
        linestyle="--",
        label="tanh saturation (std=1)",
        color="gray",
    )
    plt.yscale("log")
    plt.xlabel("tanh layer dept (1= closest to input)")
    plt.ylabel("activation std")
    plt.title("Kaiming vs Xavier init through a tanh stack")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return 0
