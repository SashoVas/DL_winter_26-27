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


def load_cifar10_loaders(
    batch_size: int,
) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader, list[str]]:
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize([64, 64], antialias=True),
        ]
    )
    torchvision.datasets.CIFAR10.url = (
        "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/"
        "resolve/c6e562aecc3a2dd5ee13f9d91630d81219c6a126/"
        "cifar-10-python.tar.gz"
    )
    dataset_train = torchvision.datasets.CIFAR10(
        root="DATA/cifar10", train=True, download=True, transform=transform
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root="DATA/cifar10", train=False, download=True, transform=transform
    )
    dataset_val, dataset_test = data.random_split(
        dataset_test,
        [0.5, 0.5],
    )
    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size,
    )
    dataloader_test = data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=batch_size,
    )
    dataloader_val = data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=batch_size,
    )

    return dataloader_train, dataloader_val, dataloader_test, dataset_train.classes


def load_clouds_loaders(
    batch_size: int,
) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader, list[str]]:
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize([64, 64], antialias=True),
        ]
    )
    dataset_train = torchvision.datasets.ImageFolder(
        root="DATA/clouds/clouds_train", transform=transform
    )
    dataset_test = torchvision.datasets.ImageFolder(
        root="DATA/clouds/clouds_test", transform=transform
    )
    dataset_val, dataset_test = data.random_split(
        dataset_test,
        [0.5, 0.5],
    )
    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size,
    )
    dataloader_test = data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=batch_size,
    )
    dataloader_val = data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=batch_size,
    )

    return dataloader_train, dataloader_val, dataloader_test, dataset_train.classes
