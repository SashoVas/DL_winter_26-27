import torch

from dl_lib.nn.module import Module


class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.shortcut_conv: None | torch.nn.Conv2d = None
        self.shortcut_bn: None | torch.nn.BatchNorm2d = None
        if in_channels != out_channels or stride != 1:
            self.shortcut_conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
            self.shortcut_bn = torch.nn.BatchNorm2d(out_channels)

        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
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
