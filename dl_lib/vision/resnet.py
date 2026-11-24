from typing import ClassVar

import torch

from dl_lib.nn.module import Module


class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.shortcut_conv: None | torch.nn.Conv2d = None
        self.shortcut_bn: None | torch.nn.BatchNorm2d = None
        if in_channels != out_channels or stride != 1:
            self.shortcut_conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.shortcut_bn = torch.nn.BatchNorm2d(out_channels)

        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def _layers(self) -> list:
        layers = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2]
        if self.shortcut_conv is not None:
            layers += [self.shortcut_conv, self.shortcut_bn]
        return layers

    def forward(self, x):
        shortcut = x
        if self.shortcut_conv and self.shortcut_bn:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        return self.relu(
            shortcut
            + self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        )

    def parameters(self) -> list[torch.Tensor]:
        return [p for layer in self._layers() for p in layer.parameters()]

    def train(self):
        for layer in self._layers():
            layer.train()
        return super().train()

    def eval(self):
        for layer in self._layers():
            layer.eval()
        return super().eval()

    def to(self, device):
        for layer in self._layers():
            layer.to(device)
        return self


class ResNet(Module):
    CHANNEL_PLAN: ClassVar[tuple[int, int, int, int]] = (16, 16, 32, 64)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = [
            ResidualBlock(
                in_channels=in_channels, out_channels=out_channels, stride=stride
            )
        ]
        layers += [
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1)
            for _ in range(num_blocks - 1)
        ]
        return layers

    def _stages(self):
        return [self.stage1, self.stage2, self.stage3]

    def __init__(self, depth: int, num_classes: int = 10, global_avg_pool: bool = True):
        super().__init__()

        if (depth - 2) % 6 != 0:
            raise ValueError(
                "Resnet depth must satisfy this equation: (depth - 2) % 6 == 0"
            )

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(
                3, self.CHANNEL_PLAN[0], kernel_size=3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(self.CHANNEL_PLAN[0]),
            torch.nn.ReLU(),
        )
        n = (depth - 2) // 6
        self.stage1 = self._make_stage(self.CHANNEL_PLAN[0], self.CHANNEL_PLAN[1], n, 1)
        self.stage2 = self._make_stage(self.CHANNEL_PLAN[1], self.CHANNEL_PLAN[2], n, 2)
        self.stage3 = self._make_stage(self.CHANNEL_PLAN[2], self.CHANNEL_PLAN[3], n, 2)
        if global_avg_pool:
            self.head = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self.CHANNEL_PLAN[-1], num_classes),
            )
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(self.CHANNEL_PLAN[-1] * 16 * 16, num_classes),
            )

    def forward(self, x):
        x = self.stem(x)
        for stage in self._stages():
            for layer in stage:
                x = layer(x)
        return self.head(x)

    def train(self):
        self.stem.train()
        for stage in self._stages():
            for block in stage:
                block.train()
        self.head.train()
        return super().train()

    def eval(self):
        self.stem.eval()
        for stage in self._stages():
            for block in stage:
                block.eval()
        self.head.eval()
        return super().eval()

    def parameters(self) -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        params.extend(self.stem.parameters())
        for stage in self._stages():
            for block in stage:
                params.extend(block.parameters())
        params.extend(self.head.parameters())
        return params

    def to(self, device):
        self.stem.to(device)
        for stage in self._stages():
            for block in stage:
                block.to(device)
        self.head.to(device)
        return self
