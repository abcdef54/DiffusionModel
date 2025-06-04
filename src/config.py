import os
from pathlib import Path
import torch
import argparse
from jaxtyping import Float
import math
from torchvision import transforms
import torchvision
from typing import Dict, List


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DATA_PATH = Path('data/')
CUSTOM_DATA_PATH = Path('data/custom/')
MODEL_PATH = Path('models/')
OUTPUT_PATH = Path('output/')


DEVICE = get_best_device()
DROP_RATE = 0.1
BATCH_SIZE: int = 32 
WORKERS = os.cpu_count() or 0
N_EPOCHS: int = 1
LOGGING_STEPS: int = BATCH_SIZE
T: int = 500
LR: float = 2e-4
NUM_RES_BLOCK: int = 2
HIDDEN_OUT_CHANNELS = 128


CURRENT_TRANSFORMATIONS: torchvision.transforms.Compose | None = None
CUSTOM_TRANSFORMATION = None

MNIST_TRANSFORM = transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
CIFAR100_TRANSFORM = cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])
CELEBA_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5063, 0.4258, 0.3832),
                         (0.2669, 0.2414, 0.2397))
])

# Dataset configuration - will be set dynamically when dataset is loaded - default to MNIST dataset
TRAIN_DATASET: torch.utils.data.Dataset | None = None
TEST_DATASET: torch.utils.data.Dataset | None = None
CURRENT_DATASET: str | None = TRAIN_DATASET.__class__.__name__ if TRAIN_DATASET else None

IN_CHANNELS: int | None = None
H: int | None = None
W: int | None = None
OUT_CHANNELS: int | None = None

# Setting up dataloader
TRAIN_DATA_LOADER: torch.utils.data.DataLoader | None = None
TEST_DATA_LOADER: torch.utils.data.DataLoader | None = None


betas: Float[torch.Tensor, 'T'] = cosine_beta_schedule(T).to(DEVICE)
alphas: Float[torch.Tensor, 'T'] = (1 - betas).to(DEVICE)
alpha_bars: Float[torch.Tensor, 'T'] = alphas.cumprod(dim=-1).to(DEVICE)
sigmas: Float[torch.Tensor, 'T'] = betas.sqrt().to(DEVICE)


# Dataset-specific model configurations
DATASET_CONFIGS: Dict[str, Dict[str, List[int]]] = {
    'MNIST': {
        'ch_mult': [1, 2, 2],
        'attn_resolutions': [14, 7],  # 28->14->7
    },
    'CIFAR10': {
        'ch_mult': [1, 2, 2, 2],
        'attn_resolutions': [16, 8],  # 32->16->8->4
    },
    'CIFAR100': {
        'ch_mult': [1, 2, 2, 2],
        'attn_resolutions': [16, 8],  # 32->16->8->4
    },
    'CelebA': {
        'ch_mult': [1, 2, 2, 2, 4],
        'attn_resolutions': [32, 16, 8],  # 64->32->16->8->4
    }
}

# Get config for current dataset - will be updated dynamically
current_config = DATASET_CONFIGS['MNIST']  # Default to MNIST

model_config = {
    'data': {
        'image_size': H,  # Will be updated when dataset is loaded
    },
    'model': {
        'type': "simple",
        'in_channels': IN_CHANNELS,  # Will be updated when dataset is loaded
        'out_ch': OUT_CHANNELS,      # Will be updated when dataset is loaded
        'ch': HIDDEN_OUT_CHANNELS,
        'ch_mult': current_config['ch_mult'],
        'num_res_blocks': NUM_RES_BLOCK,
        'attn_resolutions': current_config['attn_resolutions'],
        'dropout': DROP_RATE,
        'resamp_with_conv': True,
    },
    'diffusion': {
        'num_diffusion_timesteps': T,
    },
    'runner' : {
        'n_epochs' : N_EPOCHS,
        'logging_steps' : LOGGING_STEPS,
    }
}

def dict2namespace(model_config):
    namespace = argparse.Namespace()
    for key, value in model_config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def update_model_config_for_dataset(dataset_name: str):
    """Update model configuration when dataset changes"""
    global model_config, current_config
    if dataset_name in DATASET_CONFIGS:
        config_dict = DATASET_CONFIGS[dataset_name]
        
        # Update the current_config dictionary used for printing
        current_config = config_dict
        
        # Convert back to dict to modify
        temp_dict = {
            'data': {'image_size': model_config.data.image_size},
            'model': {
                'type': model_config.model.type,
                'in_channels': model_config.model.in_channels,
                'out_ch': model_config.model.out_ch,
                'ch': model_config.model.ch,
                'ch_mult': config_dict['ch_mult'],
                'num_res_blocks': model_config.model.num_res_blocks,
                'attn_resolutions': config_dict['attn_resolutions'],
                'dropout': model_config.model.dropout,
                'resamp_with_conv': model_config.model.resamp_with_conv,
            },
            'diffusion': {'num_diffusion_timesteps': model_config.diffusion.num_diffusion_timesteps},
            'runner': {
                'n_epochs': model_config.runner.n_epochs,
                'logging_steps': model_config.runner.logging_steps
            }
        }
        model_config = dict2namespace(temp_dict)
    
model_config = dict2namespace(model_config)


