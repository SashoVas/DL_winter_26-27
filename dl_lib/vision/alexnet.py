from dl_lib.nn.module import Module
from dl_lib.nn.linear import Linear
from dl_lib.vision.conv import Conv2d
from dl_lib.nn.sequential import Sequential
import torch


class AlexNet(Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=96,
                            kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=96, out_channels=256,
                            kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=256, out_channels=384,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=384,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=256,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

        )

    def forward(self, x):
        return super().forward(x)
