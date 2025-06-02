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


'''
def fig_save_name() -> str:
    current_time = datetime.now()
    current_date = current_time.date()
    formatted_time = current_time.strftime('%H-%M-%S')
    full_time = str(current_date) + '_At_' + str(formatted_time)
    return full_time
'''


def show_grid(image: torch.Tensor, title: str = '', figname: str = 'DefaulName',
              savefig: bool = True, show: bool = False, steps: int = 1) -> None:
    """
    Display batches of RGB images arranged in a grid.

    Parameters:
    - image_batch: torch.Tensor of shape (n_times, n_images, 3(or 1), H, W)
    - steps: Step size for sampling images (1 = all images, 2 = every other, etc.)
    """
    if image.ndim != 5 or image.size(2) not in [1, 3]:
        raise ValueError("Expected input tensor of shape (n_times, n_images, 3, H, W) or shape (n_times, n_images, 1, H, W)") 
    
    # Calculate actual number of images to display after stepping
    n_times: int = image.shape[0]
    n_images = image.shape[1]
    channels: int = image.shape[2]
    displayed_images = len(range(0, n_images, steps))
    n_cols = 5
    n_rows = (displayed_images + n_cols - 1) // n_cols  # Ceiling division

    # Create subplots
    for time in range(n_times):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()  # Flatten in case of multiple rows
        
        plot_idx = 0
        for img_idx in range(0, n_images, steps):
            # Transpose image to (64, 64, 3)
            img = image[time, img_idx]
            img = img.permute(1, 2, 0).cpu().numpy() # Numpy for matplotlib compatibility
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img = img.clip(0, 1)
            
            # Display image
            if channels == 1:
                axes[plot_idx].imshow(img, cmap='gray')
            else:
                axes[plot_idx].imshow(img)
            axes[plot_idx].axis('off')  # Hide axes
            plot_idx += 1

        # Hide any unused subplots
        for idx in range(displayed_images, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'{title} - Time Step {time}', fontsize=12)
        plt.tight_layout()
        if savefig:
            save_name = f'{figname}_{time}.png' 
            save_path = config.OUTPUT_PATH / 'grid' / save_name
            os.makedirs(save_path.parent, exist_ok=True)
            plt.savefig(save_path)
            print(f'Output image saved to: {save_path}')
        if show:
            plt.show()



def show_final_image(image: torch.Tensor, title: str = '', figname: str = 'DefaultName',
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
        save_path = config.OUTPUT_PATH / 'clear' / (figname + '.png')
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)
        print(f'Output image saved to: {save_path}')
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