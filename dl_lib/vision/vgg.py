import torch

from dl_lib.nn.module import Module

VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class Vgg(Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        cfg: list[int | str] = VGG11_CFG,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.features = torch.nn.Sequential()
        res = 64
        for layer in cfg:
            if isinstance(layer, int):
                self.features.append(
                    torch.nn.Conv2d(
                        in_channels, layer, kernel_size=3, stride=1, padding=1
                    )
                )
                if batch_norm:
                    self.features.append(torch.nn.BatchNorm2d(layer))
                self.features.append(torch.nn.ReLU())
                in_channels = layer
            elif isinstance(layer, str) and layer == "M":
                self.features.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
                res = res / 2

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(int(res * res * in_channels), 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def parameters(self) -> list[torch.Tensor]:
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
        self.features.to(device)
        self.classifier.to(device)
        return self
