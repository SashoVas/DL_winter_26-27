import torch
import torchvision
from torch.utils import data
from torchvision.transforms import v2


def load_fashion_mnist_loaders(
    batch_size: int,
) -> tuple[data.DataLoader, data.DataLoader, list[str]]:
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize([64, 64], antialias=True),
        ]
    )
    dataset_train = torchvision.datasets.FashionMNIST(
        root="DATA/fashion_mnist", train=True, download=True, transform=transform
    )
    dataset_test = torchvision.datasets.FashionMNIST(
        root="DATA/fashion_mnist", train=False, download=True, transform=transform
    )
    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size,
    )
    dataloader_test = data.DataLoader(
        dataset_test,
        shuffle=True,
        batch_size=batch_size,
    )

    return dataloader_train, dataloader_test, dataset_train.classes
