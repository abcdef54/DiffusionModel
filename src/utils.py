import random
import torch
from . import config, modified_Unet

from typing import Generator
import matplotlib.pyplot as plt
import os
from pathlib import Path


def list_files_pathlib(directory: str | Path) -> Generator[str, None, None]:
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    return (file.name for file in dir_path.iterdir() if file.is_file())


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
    
    new_model = modified_Unet.Model(config.model_config)
    new_model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    new_model = new_model.to(config.DEVICE)

    return new_model


def save_model(model: torch.nn.Module, model_name: str, save_dir: Path = config.MODEL_PATH, suffix: str = '.pth') -> Path:
    full_model_name = save_dir / (model_name + suffix)
    if os.path.exists(full_model_name):
        print(f'Already exists a model with similar name: {full_model_name}')
    
    torch.save(obj=model.state_dict(), f=full_model_name)
    return full_model_name

def show_grid(image: torch.Tensor, title: str = '', figname: str = 'DefaulName.png',
              savefig: bool = True, show: bool = False) -> None:
    """
    Display batches of RGB images arranged in a grid.

    Parameters:
    - image_batch: torch.Tensor of shape (n_times, n_images, 3(or 1), H, W)
    - n_cols: Number of columns in the grid layout.
    """
    if image.ndim != 5 or image.size(2) not in [1, 3]:
        raise ValueError("Expected input tensor of shape (n_times, n_images, 3, H, W) or shape (n_times, n_images, 1, H, W)") 
    # Image size [1, 51, 1, 28, 28]
    # which mean from left to right: plot 1 time - 51 images - of 1 color channel - hieght 28 - width 28
    # Determine grid size (e.g., 2 rows x 5 columns for 10 images)
    n_times: int = image.shape[0]
    n_images = image.shape[1]
    channels: int = image.shape[2]
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division

    # Create subplots
    
    for time in range(n_times):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()  # Flatten in case of multiple rows
        for idx in range(n_images):
            # Transpose image to (64, 64, 3)
            img = image[time, idx]
            img = img.permute(1, 2, 0).cpu().numpy() # Numpy for matplotlib compatibility
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img = img.clip(0, 1)
            # imgs = [ (img - img.min()) / (img.max() - img.min()) for img in imgs ] # Normalize to [0, 1] for imshow()
            # Display image
            if channels == 1:
                axes[idx].imshow(img, cmap='gray')
            else:
                axes[idx].imshow(img)
            axes[idx].axis('off')  # Hide axes

        # Hide any unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        if show:
            plt.show()
        if savefig:
            plt.savefig(config.OUTPUT_PATH / figname)
            print(f'Output image saved to: {config.OUTPUT_PATH / figname}')

def show_final_image(image: torch.Tensor, title: str = '', figname: str = 'DefaultName.png',
                     savefig: bool = True, show: bool = True) -> None:
    """
    Display the final image from each batch in a PyTorch tensor.

    Parameters:
    - image: torch.Tensor of shape (n_times, n_images, C, H, W), where C is 1 or 3.
    - title: Optional title for the plot.
    - figname: File name to save the figure.
    - savefig: Whether to save the figure to a file.
    - show: Whether to display the figure.
    """
    if image.ndim != 5 or image.size(2) not in [1, 3]:
        raise ValueError("Expected input tensor of shape (n_times, n_images, 1|3, H, W)")

    n_times = image.shape[0]
    channels = image.shape[2]

    fig, axes = plt.subplots(1, n_times, figsize=(n_times * 2, 2))
    if n_times == 1:
        axes = [axes]

    for time in range(n_times):
        # Select the last image in the batch
        img = image[time, -1]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        img = img.clip(0, 1)

        if channels == 1:
            axes[time].imshow(img[:, :, 0], cmap='gray')
        else:
            axes[time].imshow(img)
        axes[time].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if savefig:
        plt.savefig(config.OUTPUT_PATH / figname)
        print(f'Output final image saved to: {config.OUTPUT_PATH / figname}')
    if show:
        plt.show()
    else:
        plt.close()
