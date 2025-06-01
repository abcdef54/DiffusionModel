import torch
from torchvision import transforms, datasets
from . import config
from typing import Tuple


def download_MNIST(transformation: transforms.Compose | None = None) -> Tuple[datasets.MNIST, datasets.MNIST, int, int, int]:
    train_dataset = datasets.MNIST(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=transformation
    )
    test_dataset = datasets.MNIST(
        root=config.DATA_PATH,
        train=False,
        download=True,
        transform=transformation
    )
    return train_dataset, test_dataset, 1, 28, 28


def download_CIFAR10(transformation: transforms.Compose | None = None) -> Tuple[datasets.CIFAR10, datasets.CIFAR10, int, int, int]:
    train_dataset = datasets.CIFAR10(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=transformation
    )
    test_dataset = datasets.CIFAR10(
        root=config.DATA_PATH,
        train=False,
        download=True,
        transform=transformation
    )
    return train_dataset, test_dataset, 3, 32, 32

def download_CIFAR100(transformation: transforms.Compose | None = None) -> Tuple[datasets.CIFAR100, datasets.CIFAR100, int, int, int]:
    train_dataset = datasets.CIFAR100(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=transformation
    )
    test_dataset = datasets.CIFAR100(
        root=config.DATA_PATH,
        train=False,
        download=True,
        transform=transformation
    )
    
    return train_dataset, test_dataset, 3, 32, 32

def download_CelebA(transformation: transforms.Compose | None = None) -> Tuple[datasets.CelebA, datasets.CelebA, int, int, int]:
    train_dataset = datasets.CelebA(
        root=config.DATA_PATH,
        download=True,
        split='train',
        transform=transformation, 
    )
    valid_dataset = datasets.CelebA(
        root=config.DATA_PATH,
        download=True,
        split='valid',
        transform=transformation
    )
    return train_dataset, valid_dataset, 3, 64, 64

def download_ImageNet(train: bool = True, transformation: transforms.Compose | None = None) -> datasets.ImageNet:
    dataset = datasets.ImageNet(
        root=config.DATA_PATH,
        train=train,
        download=True,
        transform=transformation
    )
    return dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()