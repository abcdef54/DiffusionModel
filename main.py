import torch
from torchvision import transforms
import torchvision
from src import config, utils, modified_pixelCNN, diffusion
import src
import src.datasets

from pathlib import Path
from datetime import datetime
from typing import Tuple, List
from os import system, name


def clear_terminal():
    if name == 'nt':
        system('cls')
    else:
        system('clear')

def make_transformation() -> Tuple[torchvision.transforms.Compose, List[str]]:
    transformations = []
    transform_strings = []
    choice: int
    while True:
        print('NOTE: Once you have choosen a dataset it will override the default transform and will use that transform from now on.')
        print('1. ToTensor')
        print('2. Resize()')
        print('3. Normalize(-1,1)')
        print('4. Normalize(0,1)')
        print('5. Reset')
        print('6. See current transforms')
        print('7. Use default transform')
        print('0. Exist')
        choice = int(input('Choice: '))

        match choice:
            case 1:
                transformations.append(transforms.ToTensor())
                transform_strings.append('ToTensor()')
            case 2:
                W = int(input('Image width: '))
                H = int(input('Image height: '))
                transformations.append(transforms.Resize((W,H)))
                transform_strings.append(f'Resize({W},{H})')
            case 3:
                transformations.append(transforms.Normalize((0.5,), (0.5,)))
                transform_strings.append('Normalize(-1,1)')
            case 4:
                transformations.append(transforms.Normalize((0.0,), (1.0,)))
                transform_strings.append('Normalize(0,1)')
            case 5:
                transformations.clear()
                transform_strings.clear()
            case 6:
                print('transform.Compose([')
                for transform in transform_strings:
                    print(transform + ',')
                print('])')
            case 7:
                print('Default transforms:\n' \
                'transform.Compose([\n' \
                'ToTensor(),\n' \
                'Normalize((0.5,),(0.5,))\n' \
                '])')
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5))
                ]), ['ToTensor()', 'Normalize((0.5,),(0.5))']
            case 0:
                break
            case _:
                print(f'Invalid choice: {choice} please choose again')
    return transforms.Compose([*transformations]), transform_strings


def choose_dataset() -> str:
    datasets_list = ['MNIST', 'CIFAR10', 'CIFAR100', 'CelebA']
    print('''
        NOTE: Once you have choosen a dataset it will override the default dataset (which is MNIST) and will use that dataset from now on.
        Available datasets:
        1. MNIST
        2. CIFAR10
        3. CIFAR100
        4. CelebA
        0. Exist
            ''')
    choice = int(input('Choice: '))
    if choice == 0:
        print(f'Current dataset: {config.CURRENT_DATASET}')
    if choice == 1:
        config.TRAIN_DATASET, config.TEST_DATASET, config.IN_CHANNELS, config.H, config.W = src.datasets.download_MNIST(transformation=config.TRANSFORMATIONS)
    if choice == 2:
        config.TRAIN_DATASET, config.TEST_DATASET, config.IN_CHANNELS, config.H, config.W = src.datasets.download_CIFAR10(transformation=config.TRANSFORMATIONS)
    if choice == 3:
        config.TRAIN_DATASET, config.TEST_DATASET, config.IN_CHANNELS, config.H, config.W = src.datasets.download_CIFAR100(transformation=config.TRANSFORMATIONS)
    if choice == 4:
        config.TRAIN_DATASET, config.TEST_DATASET, config.IN_CHANNELS, config.H, config.W = src.datasets.download_CelebA(transformation=config.TRANSFORMATIONS)
    else:
        print(f'Invalid choice: {choice}')
        print(f'Current dataset: {config.CURRENT_DATASET}')

    return datasets_list[choice]

def fig_save_name() -> str:
    current_time = datetime.now()
    current_date = current_time.date()
    formatted_time = current_time.strftime('%H-%M-%S')
    full_time = str(current_date) + '_At_' + str(formatted_time)
    return full_time

def set_hyperparameters():
    while True:
        print('1. Epochs')
        print('2. Batch Size')
        print('3. Learning Rate')
        print('4. Logging Step')
        print('5. T')
        print('6. Drop Rate')
        print('7. Res Block')
        choice: int = int(input('Choice: '))

        match choice:
            case 1:
                epochs: int = int(input('n_epochs: '))
                config.N_EPOCHS = epochs
            case 2:
                batch_size: int = int(input('batch_size: '))
                config.BATCH_SIZE = batch_size
            case 3:
                lr: float = float(input('lr: '))
                config.LR = lr
            case 4:
                logging_steps: int = int(input('logging_steps: '))
                config.LOGGING_STEPS = logging_steps
            case 5:
                T: int = int(input('T: '))
                config.T = T
            case 6:
                drop_rate: float = float(input('drop_rate: '))
                config.DROP_RATE = drop_rate
            case 7:
                res_blocks: int = int(input('res_blocks: '))
                config.NUM_RES_BLOCK = res_blocks
            case 0:
                break
            case _:
                print(f'Invalid choice: {choice} please choose again')

def update_config_device(device: torch.device):
    config.betas = config.betas.to(config.DEVICE)
    config.alphas = config.alphas.to(config.DEVICE)
    config.alpha_bars = config.alpha_bars.to(config.DEVICE)
    config.sigmas = config.sigmas.to(config.DEVICE)


def main_loop() -> None:
    model: torch.nn.Module = modified_pixelCNN.Model(config.model_config).to(config.DEVICE)
    opti = torch.optim.Adam(model.parameters(), lr=config.LR)
    name: str
    suffix: str
    full_model_name: Path
    dataset_name: str
    checkpoint: str
    transform_strings: List[str]

    while True:
        clear_terminal()
        print('-------------------------------------------------------------Diffusion-Model-----------------------------------------------------------------')
        print('1. Train Model')
        print('2. Generate Images')
        print('3. Load Model')
        print('4. Choose Dataset')
        print('5. Make transformation')
        print('6. Save model')
        print('7. Training config')
        print('8. Change device')
        print('9. Set Hyperparameters')
        print('0. Exist')
        choice = int(input('Choice: '))

        match choice:
            case 1:
                checkpoint: str = input('Checkpoint save (Y/N): ')
                if checkpoint.lower() == 'y':
                    model_name = input('Model name for checkpoint save: ')
                    diffusion.train(model, config.TRAIN_DATA_LOADER, config.TEST_DATA_LOADER, config.N_EPOCHS, opti, config.LOGGING_STEPS, model_name, checkpoint=True)
                else:
                    diffusion.train(model, config.TRAIN_DATA_LOADER, config.TEST_DATA_LOADER, config.N_EPOCHS, opti, config.LOGGING_STEPS, checkpoint=False)
            case 2:
                n_images: int = int(input('Number of images to generate: '))
                generated_images = diffusion.inference(model, n_images)
                image_name = fig_save_name() + '.png'

                utils.show_grid(generated_images, 'Generated Image From Diffusion Model', savefig=True, figname=image_name, show=True)
                utils.show_images(generated_images, 'Generated Image From Diffusion Model', savefig=True, figname=image_name, show=False)
            case 3:
                name = input('Model name: ')
                model_path = config.MODEL_PATH / name
                model = utils.load_model(model_path)
                opti = torch.optim.Adam(model.parameters(), lr=config.LR)
            case 4:
                dataset_name = choose_dataset()
                print(f'Current dataset: {dataset_name}')
            case 5:
                config.TRANSFORMATIONS, transform_strings = make_transformation()
            case 6:
                name = input('Model name: ')
                suffix = input('Suffix (.pth or .pt): ')

                full_model_name = utils.save_model(model, name, suffix=suffix)
                print(f'Saved model at: {full_model_name}')
            case 7:
                print('-------------------------------------------------------------Current-Config-----------------------------------------------------------------')
                print(f'Model name: {full_model_name if "full_model_name" in locals() else "Not set"}')
                print(f'Dataset: {dataset_name if "dataset_name" in locals() else config.CURRENT_DATASET}')

                print(f'Current transforms: ')
                print('transform.Compose([')
                for transform in transform_strings:
                    print(transform)
                print('])\n')

                print(f'n_epochs: {config.N_EPOCHS}')
                print(f'Training device: {config.DEVICE}')
            case 8:
                print('1. CPU')
                print('2. CUDA')
                print('3. MPS')
                device = int(input('Choose: '))

                if device == 1:
                    config.DEVICE = torch.device('cpu')
                    update_config_device(torch.device('cpu'))
                if device == 2:
                    if torch.cuda.is_available():
                        config.DEVICE = torch.device('cuda')
                        update_config_device(torch.device('cuda'))
                    else:
                        print(f'Device: "cuda" is not available on your machine please choose a different one')
                if device == 3:
                    if torch.mps.is_available():
                        config.DEVICE = torch.device('mps')
                        update_config_device(torch.device('mps'))
                    else:
                        print(f'Device: "mps" is not available on your machine please choose a different one')
            case 0:
                break
            case _:
                print(f'Invalid choice: {choice} please choose again')