import torch
from torchvision import transforms, datasets
from . import config
from typing import Tuple

def update_dataset_config(dataset_name: str,
                          transformation: transforms.Compose,
                          train_dataset: torch.utils.data.Dataset,
                          test_dataset: torch.utils.data.Dataset,
                          in_channels: int,
                          out_channels: int,
                          height: int,
                          width: int) -> None:
        
        # Update dateset config
        config.CURRENT_DATASET = dataset_name
        config.CURRENT_TRANSFORMATIONS = transformation
        config.TRAIN_DATASET = train_dataset
        config.TEST_DATASET = test_dataset
        config.IN_CHANNELS = in_channels
        config.OUT_CHANNELS = out_channels
        config.H = height
        config.W = width

        # Remake dataloaders
        config.TRAIN_DATA_LOADER = config.make_dataloader(train_dataset, config.BATCH_SIZE)
        config.TEST_DATA_LOADER = config.make_dataloader(test_dataset, config.BATCH_SIZE)

        # Update model config
        config.model_config.model.in_channels = in_channels
        config.model_config.model.out_ch = in_channels
        config.model_config.data.image_size = height
        
def download_MNIST(transformation: transforms.Compose | None = None) -> Tuple[datasets.MNIST, datasets.MNIST, int, int, int, int]:
    if transformation is None:
        transformation = config.MNIST_TRANSFORM
        
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

    update_dataset_config('MNIST', transformation, train_dataset, test_dataset, 1, 1, 28, 28)

    return train_dataset, test_dataset, 1, 1, 28, 28


def download_CIFAR10(transformation: transforms.Compose | None = None) -> Tuple[datasets.CIFAR10, datasets.CIFAR10, int, int, int, int]:
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

    update_dataset_config('CIFAR10', transformation, train_dataset, test_dataset, 3, 3, 32, 32)

    return train_dataset, test_dataset, 3, 3, 32, 32

def download_CIFAR100(transformation: transforms.Compose | None = None) -> Tuple[datasets.CIFAR100, datasets.CIFAR100, int, int, int, int]:
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

    update_dataset_config('CIFAR100', transformation, train_dataset, test_dataset, 3, 3, 32, 32)

    return train_dataset, test_dataset, 3, 3, 32, 32

def download_CelebA(transformation: transforms.Compose | None = None) -> Tuple[datasets.CelebA, datasets.CelebA, int, int, int, int]:
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

    update_dataset_config('CelebA', transformation, train_dataset, valid_dataset, 3, 3, 64, 64)

    return train_dataset, valid_dataset, 3, 3, 64, 64


'''
def download_ImageNet(train: bool = True, transformation: transforms.Compose | None = None) -> datasets.ImageNet:
    dataset = datasets.ImageNet(
        root=config.DATA_PATH,
        train=train,
        download=True,
        transform=transformation
    )
    
    return dataset
'''

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()