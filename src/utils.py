import random
import torch
from . import config, modified_Unet
from typing import Generator, Dict, List
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


def denormalize_image(img: torch.Tensor, dataset_name: str = None) -> torch.Tensor:
    """Denormalize image based on dataset-specific normalization."""
    if dataset_name is None:
        dataset_name = config.CURRENT_DATASET
    
    if dataset_name == 'MNIST':
        # MNIST: Normalize((0.1307,), (0.3081,))
        mean = torch.tensor([0.1307])
        std = torch.tensor([0.3081])
    elif dataset_name in ['CIFAR10', 'CIFAR100']:
        if dataset_name == 'CIFAR10':
            # CIFAR10: Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
        else:
            # CIFAR100: Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            mean = torch.tensor([0.5071, 0.4867, 0.4408])
            std = torch.tensor([0.2675, 0.2565, 0.2761])
    elif dataset_name == 'CelebA':
        # CelebA: Normalize((0.5063, 0.4258, 0.3832), (0.2669, 0.2414, 0.2397))
        mean = torch.tensor([0.5063, 0.4258, 0.3832])
        std = torch.tensor([0.2669, 0.2414, 0.2397])
    else:
        # Fallback: assume [-1, 1] normalization
        return (img + 1) / 2
    
    # Reverse normalization: x = (x_norm * std) + mean
    if img.dim() >= 3:
        # Handle batched images: reshape mean/std for broadcasting
        if img.dim() == 3:  # (C, H, W)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif img.dim() == 4:  # (B, C, H, W)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
    
    denorm_img = img * std + mean
    return torch.clamp(denorm_img, 0, 1)

def show_grid(image: torch.Tensor, title: str = '', figname: str = 'DefaulName',
              savefig: bool = True, show: bool = False, steps: int = 1) -> None:
    """
    Display batches of images arranged in a grid.

    Parameters:
    - image: torch.Tensor of shape (n_samples, n_times, C, H, W) from inference
    - steps: Step size for sampling timesteps (1 = all timesteps, 2 = every other, etc.)
    """
    if image.ndim != 5 or image.size(2) not in [1, 3]:
        raise ValueError("Expected input tensor of shape (n_samples, n_times, C, H, W)")
    
    n_samples = image.shape[0]
    n_times = image.shape[1]
    channels = image.shape[2]
    
    # Select timesteps to display based on steps parameter
    if steps == 1:
        timesteps_to_show = list(range(n_times))
    else:
        timesteps_to_show = list(range(0, n_times, steps))
    
    n_cols = min(len(timesteps_to_show), 5)
    n_rows = min(n_samples, 3)  # Show max 3 samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    # Ensure axes is always 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[axes[i]] for i in range(n_rows)]
    
    for sample_idx in range(min(n_samples, n_rows)):
        for col_idx, time_idx in enumerate(timesteps_to_show[:n_cols]):
            img = image[sample_idx, time_idx].clone()
            
            # Denormalize and clamp to valid range
            img = denormalize_image(img)
            img = torch.clamp(img, 0, 1)
            
            # Convert to numpy for matplotlib (ensure on CPU)
            img = img.detach().cpu()
            if channels == 1:
                img_np = img.squeeze().numpy()
                axes[sample_idx][col_idx].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            else:
                img_np = img.permute(1, 2, 0).numpy()
                axes[sample_idx][col_idx].imshow(img_np)
            
            axes[sample_idx][col_idx].axis('off')
            if sample_idx == 0:  # Only add titles to top row
                axes[sample_idx][col_idx].set_title(f't={time_idx}', fontsize=10)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if savefig:
        save_name = f'{figname}_grid.png'
        save_path = config.OUTPUT_PATH / 'grid' / save_name
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Output grid saved to: {save_path}')
    
    if show:
        plt.show()
    else:
        plt.close()

def show_final_image(image: torch.Tensor, title: str = '', figname: str = 'DefaultName',
                     savefig: bool = True, show: bool = True) -> None:
    """
    Display the final denoised images from the diffusion process.

    Parameters:
    - image: torch.Tensor of shape (n_samples, n_timesteps, C, H, W) from inference
    - title: Optional title for the plot.
    - figname: File name to save the figure.
    - savefig: Whether to save the figure to a file.
    - show: Whether to display the figure.
    """
    if image.ndim != 5 or image.size(2) not in [1, 3]:
        raise ValueError("Expected input tensor of shape (n_samples, n_timesteps, C, H, W)")

    n_samples = image.shape[0]
    channels = image.shape[2]
    
    # Use final timestep (fully denoised images)
    final_images = image[:, -1]  # Shape: (n_samples, C, H, W)
    
    # Determine grid layout
    n_cols = min(n_samples, 5)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    # Flatten axes for consistent indexing
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx in range(n_samples):
        ax = axes[idx] if n_samples > 1 else axes[0]
        
        img = final_images[idx].clone()
        img = denormalize_image(img)
        img = torch.clamp(img, 0, 1)

        # Ensure tensor is on CPU before numpy conversion
        img = img.detach().cpu()
        if channels == 1:
            img_np = img.squeeze().numpy()
            ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        else:
            img_np = img.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
        
        ax.axis('off')
        ax.set_title(f'Sample {idx+1}', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes) if hasattr(axes, '__len__') else 1):
        if hasattr(axes, '__len__') and idx < len(axes):
            axes[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if savefig:
        save_path = config.OUTPUT_PATH / 'clear' / (figname + '_final.png')
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Final images saved to: {save_path}')
    if show:
        plt.show()
    else:
        plt.close()



def plot_loss_curve(train_result_dict: Dict[str, List[float]], figname: str) -> None:
    '''
    Plots training and validation loss curves per epoch and per batch.

    Parameters:
    - train_result_dict: Dictionary containing the following keys:
        - 'train_loss_per_batch': List of training losses per batch.
        - 'train_loss_per_epoch': List of training losses per epoch.
        - 'val_loss_per_batch': List of validation losses per batch.
        - 'val_loss_per_epoch': List of validation losses per epoch.
    '''
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Plot training loss per epoch
    axes[0, 0].plot(train_result_dict['train_loss_per_epoch'])
    axes[0, 0].set_title('Training Loss per Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')

    # Plot training loss per batch
    axes[0, 1].plot(train_result_dict['train_loss_per_batch'])
    axes[0, 1].set_title('Training Loss per Batch')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Loss')

    # Plot validation loss per epoch
    axes[1, 0].plot(train_result_dict['val_loss_per_epoch'])
    axes[1, 0].set_title('Validation Loss per Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')

    # Plot validation loss per batch
    axes[1, 1].plot(train_result_dict['val_loss_per_batch'])
    axes[1, 1].set_title('Validation Loss per Batch')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Loss')

    save_path = config.OUTPUT_PATH / 'loss_curves' / figname
    plt.savefig(save_path)
    print(f'Saved loss curve at: {save_path}')
    plt.tight_layout()
    plt.show()

def show_images_every_n_steps(image: torch.Tensor, n: int = 10, start: int = 1, figname: str = 'DefaultName', end: int = 500):  
    """
    For each nth step from start to end (inclusive), open a new plot showing all samples at that step.
    Parameters:
    - image: torch.Tensor of shape (n_samples, n_times, C, H, W)
    - n: interval of steps to show
    - start: first step to show (inclusive)
    - end: last step to show (inclusive)
    """
    n_samples = image.shape[0]
    n_times = image.shape[1]
    channels = image.shape[2]
    for t in range(start, min(end+1, n_times), n):
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
        if n_samples == 1:
            axes = [axes]
        for sample_idx in range(n_samples):
            img = image[sample_idx, t].clone()
            img = denormalize_image(img)
            img = torch.clamp(img, 0, 1)
            img = img.detach().cpu()
            if channels == 1:
                img_np = img.squeeze().numpy()
                axes[sample_idx].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            else:
                img_np = img.permute(1, 2, 0).numpy()
                axes[sample_idx].imshow(img_np)
            axes[sample_idx].axis('off')
            axes[sample_idx].set_title(f'Sample {sample_idx+1}')
        fig.suptitle(f'Reconstructed Images at Step t={t}', fontsize=14)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'grid' / f'{figname}_grid_{t}.png')
        plt.close()