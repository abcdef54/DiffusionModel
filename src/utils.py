import random
import torch
import torchvision
from . import config, modified_pixelCNN

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_dataloader(dataset: torch.utils.data.Dataset,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    num_workers: int = config.WORKERS,
                    pin_memory: bool = True) -> torch.utils.data.DataLoader:
    
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return loader


def load_model(model_path: Path) -> torch.nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Invalid path, did not find: {model_path}')
    
    new_model = modified_pixelCNN.Model(config.model_config)
    new_model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    new_model = new_model.to(config.DEVICE)

    return new_model


def save_model(model: torch.nn.Module, model_name: str, save_dir: Path = config.MODEL_PATH, suffix: str = '.pth') -> Path:
    full_model_name = save_dir / (model_name + suffix)
    if os.path.exists(full_model_name):
        print(f'Already exists a model with similar name: {full_model_name}')
    
    torch.save(obj=model.state_dict(), f=full_model_name)
    return full_model_name


def show_grid(imgs: List[torch.Tensor], title: str = '', show: bool = False,
            figname: str = '', savefig: bool = False) -> None:
    fig, ax = plt.subplots()
    
    imgs = [(img - img.min()) / (img.max() - img.min()) for img in imgs]
    img = torchvision.utils.make_grid(imgs, padding=1, pad_value=1).numpy()
    ax.set_title(title)
    ax.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
    ax.set(xticks=[], yticks=[])

    if savefig:
        plt.savefig(config.OUTPUT_PATH / figname)
    if show:
        plt.show()


def show_images(imgs: List[torch.Tensor], title: str = '', show: bool = False,
                figname: str = '', savefig: bool = False) -> None:
    fig, ax = plt.subplots()

    imgs = [(img - img.min()) / (img.max() - img.min()) for img in imgs]

    for img in imgs:
        # Transpose for compatibility with matplotlib
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set(xticks=[], yticks=[])

    plt.title(title)
    if savefig:
        plt.savefig(config.OUTPUT_PATH / figname)
    if show:
        plt.show()
