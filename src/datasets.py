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
    config.CURRENT_TRANSFORMATIONS = transformation
    config.CURRENT_DATASET = 'MNIST'
    config.TRAIN_DATA_LOADER = config.make_dataloader(config.TRAIN_DATASET, config.BATCH_SIZE)
    config.TEST_DATA_LOADER = config.make_dataloader(config.TEST_DATASET, config.BATCH_SIZE)
    return train_dataset, test_dataset, 1, 28, 28


def download_CIFAR10(transformation: transforms.Compose | None = None) -> Tuple[datasets.CIFAR10, datasets.CIFAR10, int, int, int]:
    if transformation is None:
        transformation = config.CIFAR10_TRANSFORM

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
    config.CURRENT_TRANSFORMATIONS = transformation
    config.CURRENT_DATASET = 'CIFAR10'
    config.TRAIN_DATA_LOADER = config.make_dataloader(config.TRAIN_DATASET, config.BATCH_SIZE)
    config.TEST_DATA_LOADER = config.make_dataloader(config.TEST_DATASET, config.BATCH_SIZE)
    return train_dataset, test_dataset, 3, 32, 32

def download_CIFAR100(transformation: transforms.Compose | None = None) -> Tuple[datasets.CIFAR100, datasets.CIFAR100, int, int, int]:
    if transformation is None:
        transformation = config.CIFAR100_TRANSFORM

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
    config.CURRENT_TRANSFORMATIONS = transformation
    config.CURRENT_DATASET = 'CIFAR100'
    config.TRAIN_DATA_LOADER = config.make_dataloader(config.TRAIN_DATASET, config.BATCH_SIZE)
    config.TEST_DATA_LOADER = config.make_dataloader(config.TEST_DATASET, config.BATCH_SIZE)
    return train_dataset, test_dataset, 3, 32, 32

def download_CelebA(transformation: transforms.Compose | None = None) -> Tuple[datasets.CelebA, datasets.CelebA, int, int, int]:
    if transformation is None:
        transformation = config.CELEBA_TRANSFORM

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
    config.CURRENT_TRANSFORMATIONS = transformation
    config.CURRENT_DATASET = 'CelebA'
    config.TRAIN_DATA_LOADER = config.make_dataloader(config.TRAIN_DATASET, config.BATCH_SIZE)
    config.TEST_DATA_LOADER = config.make_dataloader(config.TEST_DATASET, config.BATCH_SIZE)
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