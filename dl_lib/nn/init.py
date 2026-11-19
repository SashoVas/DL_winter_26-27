import math
from typing import Literal

import torch


def calculate_gain(
    func: Literal[
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "leaky_relu",
        "linear",
        "relu",
        "selu",
        "sigmoid",
        "tanh",
    ],
):
    return torch.nn.init.calculate_gain(func)


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor):

    if tensor.dim() < 2:
        raise ValueError(
            "Fan in and fan out can't be computed for tensor with fewer than 2 dimensions"
        )
    weights_per_resolution = 1
    for dim in tensor.shape[2:]:
        weights_per_resolution *= dim
    return (
        tensor.shape[1] * weights_per_resolution,
        tensor.shape[0] * weights_per_resolution,
    )


def calculate_std_kaiming(
    tensor: torch.Tensor,
    mode: str = "fan_in",
    nonlinearity: Literal[
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "leaky_relu",
        "linear",
        "relu",
        "selu",
        "sigmoid",
        "tanh",
    ] = "leaky_relu",
) -> float:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity)
    return gain / math.sqrt(fan)


def kaiming_normal_(
    tensor: torch.Tensor,
    mode: str = "fan_in",
    nonlinearity: Literal[
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "leaky_relu",
        "linear",
        "relu",
        "selu",
        "sigmoid",
        "tanh",
    ] = "leaky_relu",
) -> torch.Tensor:
    std = calculate_std_kaiming(tensor, mode, nonlinearity)
    with torch.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(
    tensor: torch.Tensor,
    mode: str = "fan_in",
    nonlinearity: Literal[
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "leaky_relu",
        "linear",
        "relu",
        "selu",
        "sigmoid",
        "tanh",
    ] = "leaky_relu",
) -> torch.Tensor:
    std = calculate_std_kaiming(tensor, mode, nonlinearity)
    with torch.no_grad():
        return tensor.uniform_(-std, std)


def calculate_std_xavier(
    tensor: torch.Tensor,
    nonlinearity: Literal[
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "leaky_relu",
        "linear",
        "relu",
        "selu",
        "sigmoid",
        "tanh",
    ] = "leaky_relu",
) -> float:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    gain = calculate_gain(nonlinearity)
    return gain / math.sqrt(fan_in + fan_out)


def xavier_normal_(
    tensor: torch.Tensor,
    nonlinearity: Literal[
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "leaky_relu",
        "linear",
        "relu",
        "selu",
        "sigmoid",
        "tanh",
    ] = "leaky_relu",
) -> torch.Tensor:
    std = calculate_std_xavier(tensor, nonlinearity)
    with torch.no_grad():
        return tensor.normal_(0, std)


def xavier_uniform_(
    tensor: torch.Tensor,
    nonlinearity: Literal[
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "leaky_relu",
        "linear",
        "relu",
        "selu",
        "sigmoid",
        "tanh",
    ] = "leaky_relu",
) -> torch.Tensor:
    std = calculate_std_xavier(tensor, nonlinearity)
    with torch.no_grad():
        return tensor.uniform_(-std, std)
