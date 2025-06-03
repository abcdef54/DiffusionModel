import os
# Set CUDA memory allocation configuration BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torchvision import transforms
import torchvision
from src import config, modified_Unet, utils, diffusion
import src
import src.datasets

from pathlib import Path
from typing import Tuple, List
from os import system, name

# Clear CUDA cache at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def clear_terminal():
    if name == 'nt':
        system('cls')
    else:
        system('clear')

def take_integer_input(input_string: str = '') -> int:
   try:
       in_put = int(input(input_string))
       return in_put
   except ValueError:
       print("Invalid input. Please enter a valid integer.")
       return take_integer_input(input_string)
   except KeyboardInterrupt:
       input("\nOperation cancelled.")
       return 0

def get_transform_strings(dataset_name: str) -> List[str]:
    transform_map = {
        'MNIST': ['ToTensor()', 'Normalize((0.1307,), (0.3081,))'],
        'CIFAR10': ['ToTensor()', 'Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))'],
        'CIFAR100': ['ToTensor()', 'Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))'],
        'CelebA': ['CenterCrop(178)', 'Resize(64)', 'ToTensor()', 'Normalize((0.5063, 0.4258, 0.3832), (0.2669, 0.2414, 0.2397))']
    }
    return transform_map.get(dataset_name, ['ToTensor()'])

def make_transformation() -> Tuple[torchvision.transforms.Compose, List[str]]:
    transformations = []
    transform_strings = ['ToTensor()', 'Normalize((0.1307,), (0.3081,))']
    choice: int
    while True:
        print('NOTE: Once you have choosen a dataset it will override the default transform and will use that transform from now on.')
        print('1. ToTensor')
        print('2. Resize()')
        print('3. Normalize(-1,1)')
        print('4. Normalize(0,1)')
        print('5. Reset')
        print('6. See current transforms')
        print('7. Set to default transform')
        print('0. Exist')
        choice = take_integer_input('Choice: ')

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
                print('Default transforms (For MNIST):\n' \
                'transform.Compose([\n' \
                'ToTensor(),\n' \
                'Normalize((0.1307,), (0.3081,))\n' \
                '])')
                return config.MNIST_TRANSFORM, transform_strings
            case 0:
                break
            case _:
                input(f'Invalid choice: {choice} please choose again')
    return transforms.Compose([*transformations]), transform_strings


def choose_dataset() -> Tuple[str, List[str]]:
    datasets_list = ['MNIST', 'CIFAR10', 'CIFAR100', 'CelebA']
    while True:
        clear_terminal()
        print(f'''
NOTE: Once you have choosen a dataset it will override the current dataset (which is {config.CURRENT_DATASET}) and will use that dataset from now on.
NOTE: The appropriate transformation for each dataset is automatically set every time you choose a dataset
      so you don't need to remake the transformation using option 5 in the main menu.
      Unless you want to create your custom transformation.

Current dataset: {config.CURRENT_DATASET}
Available datasets:
1. MNIST
2. CIFAR10
3. CIFAR100
4. CelebA
0. Exist
                ''')
        choice = take_integer_input('Choice: ')
        if choice == 0:
            break
        elif choice == 1:
            src.datasets.download_MNIST(transformation=config.CUSTOM_TRANSFORMATION)
            transform_strings = get_transform_strings('MNIST')
        elif choice == 2:
            src.datasets.download_CIFAR10(transformation=config.CUSTOM_TRANSFORMATION)
            transform_strings = get_transform_strings('CIFAR10')
        elif choice == 3:
            src.datasets.download_CIFAR100(transformation=config.CUSTOM_TRANSFORMATION)
            transform_strings = get_transform_strings('CIFAR100')
        elif choice == 4:
            src.datasets.download_CelebA(transformation=config.CUSTOM_TRANSFORMATION)
            transform_strings = get_transform_strings('CelebA')
        else:
            print(f'Invalid choice: {choice}')
            input(f'Current dataset: {config.CURRENT_DATASET}')
        
        input(f'Current dataset: {config.CURRENT_DATASET}')

    return datasets_list[choice], transform_strings

def set_hyperparameters():
    while True:
        print('1. Epochs')
        print('2. Batch Size')
        print('3. Learning Rate')
        print('4. Logging Step')
        print('5. T')
        print('6. Drop Rate')
        print('7. Res Block')
        print('0. Exist')
        choice: int = take_integer_input('Choice: ')

        match choice:
            case 1:
                epochs: int = int(input('n_epochs: '))
                config.N_EPOCHS = epochs
                config.model_config.runner.n_epochs = epochs

            case 2:
                batch_size: int = int(input('batch_size: '))
                config.BATCH_SIZE = batch_size

                config.TRAIN_DATA_LOADER = config.make_dataloader(config.TRAIN_DATASET, config.BATCH_SIZE)
                config.TEST_DATA_LOADER = config.make_dataloader(config.TEST_DATASET, config.BATCH_SIZE)
                input(f'Remade dataloaders with the new batch size')

            case 3:
                lr: float = float(input('lr: '))
                config.LR = lr

            case 4:
                logging_steps: int = int(input('logging_steps: '))
                config.LOGGING_STEPS = logging_steps
                config.model_config.runner.logging_steps = logging_steps

            case 5:
                T: int = int(input('T: '))
                config.T = T
                config.model_config.diffusion.num_diffusion_timesteps = T

            case 6:
                drop_rate: float = float(input('drop_rate: '))
                config.DROP_RATE = drop_rate
                config.model_config.model.dropout = drop_rate

            case 7:
                res_blocks: int = int(input('res_blocks: '))
                config.NUM_RES_BLOCK = res_blocks
                config.model_config.model.num_res_blocks = res_blocks

            case 0:
                break
            case _:
                input(f'Invalid choice: {choice} please choose again')

def set_device():
    while True:
        print('1. CPU')
        print('2. CUDA')
        print('3. MPS')
        print('0. Exist')
        device = take_integer_input('Choose: ')

        if device == 1:
            update_config_device(torch.device('cpu'))
            input('Device is set to: CPU')
        elif device == 2:
            if torch.cuda.is_available():
                update_config_device(torch.device('cuda'))
            else:
                input(f'Device: "cuda" is not available on your machine please choose a different one')
        elif device == 3:
            if torch.mps.is_available():
                update_config_device(torch.device('mps'))
            else:
                input(f'Device: "mps" is not available on your machine please choose a different one')
        elif device == 0:
            break
        else:
            input(f'Invalid choice: {device} please choose again.')

def update_config_device(device: torch.device):
    config.DEVICE = device
    config.betas = config.betas.to(config.DEVICE)
    config.alphas = config.alphas.to(config.DEVICE)
    config.alpha_bars = config.alpha_bars.to(config.DEVICE)
    config.sigmas = config.sigmas.to(config.DEVICE)

def print_hyperparameters(header: bool = True):
    if header:
        print(f'-------------------------------------------------------Training-Config-------------------------------------------------------')
    print(f'Image size: ({config.IN_CHANNELS},{config.W},{config.H})')
    print(f'Noising step: {config.T}')
    print(f'n_epochs: {config.N_EPOCHS}')
    print(f'Learning rate: {config.LR}')
    print(f'Drop rate: {config.DROP_RATE}')
    print(f'Res block: {config.NUM_RES_BLOCK}')
    print(f'Training device: {config.DEVICE}')

def print_dir(directory: str | Path, suffixes: str | List[str] | None = None, header: str = ''):
    print(f'-------------------------------------------------------{header}-------------------------------------------------------')
    files = list(utils.list_files_pathlib(directory))
    files_with_suffixes = []
    
    if suffixes is not None and isinstance(suffixes, list):
        for suffix in suffixes:
            files_with_suffixes.extend([file for file in files if file.endswith(suffix)])

    for i, file in enumerate(files_with_suffixes):
        print(f'{i}. {file}')
    
    print(f'----------------------------------------------------------------------------------------------------------------------')
    return files_with_suffixes

def show_context():
    """Display comprehensive context about the diffusion model program"""
    clear_terminal()
    
    print("="*100)
    print("                              DIFFUSION MODEL - PROGRAM CONTEXT")
    print("="*100)
    
    print("\n📋 PROGRAM OVERVIEW:")
    print("-" * 50)
    print("This is a Denoising Diffusion Probabilistic Model (DDPM) implementation that generates")
    print("images by learning to reverse a noise diffusion process. The model is trained to predict")
    print("noise at each timestep, enabling it to gradually denoise random samples into realistic images.")
    
    print("\n🏗️ MODEL ARCHITECTURE:")
    print("-" * 50)
    print("• Architecture: U-Net with residual blocks and attention mechanisms")
    print("• Encoder-Decoder: Symmetric downsampling and upsampling paths")
    print("• Skip Connections: Feature maps from encoder are concatenated with decoder")
    print("• Attention: Self-attention at specified resolutions for better global coherence")
    print("• Time Embedding: Sinusoidal embeddings for diffusion timestep conditioning")
    print("• Residual Blocks: ResNet-style blocks with time embedding injection")
    print(f"• Current Channels: {config.model_config.model.ch} base channels")
    print(f"• Channel Multipliers: {config.model_config.model.ch_mult}")
    print(f"• Attention Resolutions: {config.model_config.model.attn_resolutions}")
    
    print("\n🎯 HOW TO USE THIS PROGRAM:")
    print("-" * 50)
    print("TYPICAL WORKFLOW:")
    print("1. Choose Dataset (Option 4) - Select MNIST, CIFAR10, CIFAR100, or CelebA")
    print("2. Configure Transforms (Option 5) - Set image preprocessing")
    print("3. Set Hyperparameters (Option 9) - Adjust training parameters")
    print("4. Train Model (Option 1) - Train the diffusion model")
    print("5. Generate Images (Option 2) - Create new images from trained model")
    print("6. Save/Load Models (Options 3,6) - Manage model checkpoints")
    
    print("\n📊 AVAILABLE MENU OPTIONS:")
    print("-" * 50)
    print("1. Train Model      - Train diffusion model on selected dataset")
    print("2. Generate Images  - Generate images using trained model")
    print("3. Load Model       - Load a previously saved model")
    print("4. Choose Dataset   - Switch between MNIST, CIFAR10, CIFAR100, CelebA")
    print("5. Make Transform   - Configure image preprocessing transforms")
    print("6. Save Model       - Save current model state")
    print("7. Training Config  - View current configuration")
    print("8. Change Device    - Switch between CPU, CUDA, MPS")
    print("9. Set Hyperparams  - Adjust epochs, batch size, learning rate, etc.")
    print("10. Get Context     - This help screen")
    
    print("\n🗂️ SUPPORTED DATASETS:")
    print("-" * 50)
    print("• MNIST:     28x28 grayscale handwritten digits (1 channel)")
    print("• CIFAR10:   32x32 color images, 10 classes (3 channels)")
    print("• CIFAR100:  32x32 color images, 100 classes (3 channels)")
    print("• CelebA:    64x64 celebrity face images (3 channels)")
    
    print("\n🔧 CURRENT CONFIGURATION:")
    print("-" * 50)
    print(f"Dataset: {config.CURRENT_DATASET}")
    print(f"Image Size: {config.IN_CHANNELS}x{config.H}x{config.W}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LR}")
    print(f"Epochs: {config.N_EPOCHS}")
    print(f"Diffusion Steps (T): {config.T}")
    print(f"Dropout Rate: {config.DROP_RATE}")
    print(f"Residual Blocks: {config.NUM_RES_BLOCK}")
    
    print("\n🧠 DIFFUSION PROCESS EXPLAINED:")
    print("-" * 50)
    print("FORWARD PROCESS (Training):")
    print("• Real images → Gradually add Gaussian noise over T timesteps")
    print("• x₀ (real) → x₁ → x₂ → ... → xₜ (pure noise)")
    print("• Model learns to predict the noise added at each step")
    
    print("\nREVERSE PROCESS (Generation):")
    print("• Pure noise → Gradually denoise over T timesteps")
    print("• xₜ (noise) → xₜ₋₁ → ... → x₁ → x₀ (generated image)")
    print("• Model predicts noise to remove at each step")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("-" * 50)
    print("• main.py                 - Main program interface (this file)")
    print("• src/config.py          - Configuration and hyperparameters")
    print("• src/modified_pixelCNN.py - U-Net model architecture")
    print("• src/diffusion.py       - Training and inference functions")
    print("• src/utils.py           - Utility functions (save/load, visualization)")
    print("• src/datasets.py        - Dataset loading functions")
    print("• models/                - Saved model checkpoints")
    print("• output/                - Generated images")
    print("• data/                  - Downloaded datasets")
    
    print("\n⚡ PERFORMANCE TIPS:")
    print("-" * 50)
    print("• Use CUDA/MPS if available for faster training")
    print("• Start with MNIST for quick experiments")
    print("• Increase epochs for better quality (default: 3)")
    print("• Save checkpoints during long training runs")
    print("• Monitor loss to ensure proper training")
    
    print("\n🔬 TECHNICAL DETAILS:")
    print("-" * 50)
    print("• Loss Function: MSE between predicted and actual noise")
    print("• Scheduler: Cosine beta schedule for noise variance")
    print("• Optimizer: Adam with configurable learning rate")
    print("• Normalization: Images normalized to [-1, 1] range")
    print("• Time Embedding: Sinusoidal positional encoding for timesteps")
    
    print("\n" + "="*100)
    print("                              END OF CONTEXT")
    print("="*100)
    
    input("\nPress Enter to return to main menu...")


def main_loop() -> None:
    model: torch.nn.Module = modified_Unet.Model(config.model_config).to(config.DEVICE)
    opti = torch.optim.Adam(model.parameters(), lr=config.LR)
    name: str
    suffix: str
    full_model_path: Path = None # type: ignore
    checkpoint: str
    transform_strings: List[str]  = ['ToTensor()', 'Normalize((0.1307,), (0.3081,))']
    loss_dict = None

    not_allow = [
                '<', '>', ':', '"', '/', '\\', '|', '?', '*',  # Windows forbidden characters
                '#', '%', '&', '{', '}', '$', '!', "'", '`',   # Commonly problematic in URLs and scripts
                '@', '+', '=', '[', ']', ';', ',', '^', '~',   # Additional special characters
                ' '  # Space character
                ]
    

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
        print('10. Get Context')
        print('11. New Model')
        print('0. Exist')
        choice = take_integer_input('Choice: ')

        match choice:
            case 1:
                model_name = input('Model name (without suffix): ')
                if not model_name or any(char in model_name for char in not_allow):
                    input(f'Model name is empty or contains not allowed characters: {model_name}\nOnly special characters allowed are: \"-\", \"_\"')

                full_model_path = config.MODEL_PATH / model_name
                checkpoint: str = input('Checkpoint save (Y/N): ')
                print_hyperparameters()
                if checkpoint.lower() == 'y':
                    loss_dict = diffusion.train(model, config.TRAIN_DATA_LOADER, config.TEST_DATA_LOADER, config.N_EPOCHS, opti, config.LOGGING_STEPS, model_name, checkpoint=True)
                else:
                    loss_dict = diffusion.train(model, config.TRAIN_DATA_LOADER, config.TEST_DATA_LOADER, config.N_EPOCHS, opti, config.LOGGING_STEPS, checkpoint=False)
                    
                plot = input('Plot loss curve (Y/N): ')
                if plot.lower() == 'y':
                    utils.plot_loss_curve(loss_dict, figname=model_name + 'loss_curve.png')

                input('Press enter to continue.')
            case 2:
                n_images: int = take_integer_input('Number of images to generate: ')
                if n_images <= 0:
                    input('Please enter a number bigger than 0')
                    continue
                generated_images = diffusion.inference(model, n_images)

                figname = input('Image name for saving: ')
                if not figname or any(char in figname for char in not_allow):
                    input(f'Image name is empty or contains not allowed characters: {figname}\nOnly special characters allowed are: \"-\", \"_\"')
                
                steps: int = int(input('Progress image step (default 1): ')) 
                utils.show_grid(generated_images, 'Generated Image From Diffusion Model', savefig=True, show=False, figname=figname, steps=steps)
                utils.show_final_image(generated_images, 'Generated Image From Diffusion Model', savefig=True, show=True, figname=figname)
                input()
            case 3:
                while True:
                    clear_terminal()
                    files = print_dir(config.MODEL_PATH, header='Available-Models', suffixes=['.pth', '.pt'])
                    
                    print('Enter "-1" to exist.')
                    try:
                        index = int(input('Enter the index of the chosen model (0, 1, 2): '))
                    except ValueError:
                        input("Invalid input. Please enter a valid integer.")
                        continue
                    except KeyboardInterrupt:
                        input("\nOperation cancelled.")
                        index = -1

                    if index == -1:
                        break
                    elif index < -1 or index >= len(files):
                        input(f'Index out of bound: {index}')
                        continue
                    else:
                        break

                model_path = config.MODEL_PATH / files[index]
                model = utils.load_model(model_path)
                opti = torch.optim.Adam(model.parameters(), lr=config.LR)
                full_model_path = model_path

                input(f'Successfully loaded model at: {model_path}')
            case 4:
                dataset_name, transform_strings = choose_dataset()
                input(f'Recreated a new model compatible with the new dataset')
                model: torch.nn.Module = modified_Unet.Model(config.model_config).to(config.DEVICE)
                opti = torch.optim.Adam(model.parameters(), lr=config.LR)
            case 5:
                input(f'Feature not implemented')
                # config.CUSTOM_TRANSFORMATION, transform_strings = make_transformation()
            case 6:
                name = input('Model name: ')
                if not name or any(char in name for char in not_allow):
                    input(f'Model name is empty or contains not allowed characters: {name}\nOnly special characters allowed are: \"-\", \"_\"')

                suffix = input('Suffix (.pth or .pt): ')
                if not suffix or suffix not in ['.pth', '.pt']:
                    input(f'Invalid suffix: {suffix}, please enter ".pth" or .pt"')

                full_model_path = utils.save_model(model, name, suffix=suffix)
                input(f'Saved model at: {full_model_path}')

            case 7:
                print('-------------------------------------------------------Current-Config-------------------------------------------------------')
                print(f'Model path: {full_model_path}')
                print(f'Dataset: {config.CURRENT_DATASET}')
                print(f'Current transforms: ')
                print('transform.Compose([')
                for transform in transform_strings:
                    print('\t' + transform + ',')
                print('])')
                print_hyperparameters(header=False)
                input()
                
            case 8:
                set_device()
            case 9:
                set_hyperparameters()
                model: torch.nn.Module = modified_Unet.Model(config.model_config).to(config.DEVICE)
                opti = torch.optim.Adam(model.parameters(), lr=config.LR)
                input('Remade a new model with the new hyperparams')
            case 10:
                show_context()
            case 11:
                full_model_path: Path = None # type: ignore
                model: torch.nn.Module = modified_Unet.Model(config.model_config).to(config.DEVICE)
                opti = torch.optim.Adam(model.parameters(), lr=config.LR)
                print('Successfully recreated a new model')
                print(f'Model path: {full_model_path if "full_model_path" in locals() else "Not set"}')
                print(f'Dataset: {config.CURRENT_DATASET}')
                print(f'Current transforms: ')
                print('transform.Compose([')
                for transform in transform_strings:
                    print('\t' + transform + ',')
                print('])')
                print_hyperparameters(header=False)
                input()
            case 0:
                break
            case _:
                input(f'Invalid choice: "{choice}", please choose again')

if __name__ == '__main__':
    main_loop()