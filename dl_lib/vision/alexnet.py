import torch

from dl_lib.nn.module import Module
from dl_lib.nn.sequential import Sequential


class AlexNet(Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def parameters(self):
        return [*self.features.parameters(), *self.classifier.parameters()]

    def train(self):
        self.features.train()
        self.classifier.train()
        return super().train()

    def eval(self):
        self.features.eval()
        self.classifier.eval()
        return super().eval()

    def to(self, device):
        [i.to(device) for i in self.features.modules if isinstance(i, torch.nn.Module)]
        [
            i.to(device)
            for i in self.classifier.modules
            if isinstance(i, torch.nn.Module)
        ]
        return self
