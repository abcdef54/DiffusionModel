import os
from pathlib import Path
import torch
import argparse
from jaxtyping import Float
import math
from torchvision import transforms
import torchvision



def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

DATA_PATH = Path('data/')
CUSTOM_DATA_PATH = Path('data/custom/')
MODEL_PATH = Path('models/')
OUTPUT_PATH = Path('output/')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DROP_RATE = 0.1
BATCH_SIZE: int = 32
WORKERS = os.cpu_count() or 0
N_EPOCHS: int = 3
LOGGING_STEPS: int = 10
T: int = 50
LR: float = 2e-4
NUM_RES_BLOCK: int = 2

TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])

# Dataset configuration - will be set dynamically when dataset is loaded - default to MNIST dataset
TRAIN_DATASET: torch.utils.data.Dataset = torchvision.datasets.MNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=TRANSFORMATIONS
    )
TEST_DATASET: torch.utils.data.Dataset = torchvision.datasets.MNIST(
    root=DATA_PATH,
    train=False,
    transform=TRANSFORMATIONS,
    download=True
)

CURRENT_DATASET: str = TRAIN_DATASET.__class__.__name__

IN_CHANNELS: int = 1  # Default for MNIST
H: int = 28  # Default for MNIST
W: int = 28  # Default for MNIST
OUT_CHANNELS: int = IN_CHANNELS  # Default for MNIST

# Setting up dataloader
def make_dataloader(dataset: torch.utils.data.Dataset,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    num_workers: int = WORKERS,
                    pin_memory: bool = True) -> torch.utils.data.DataLoader:
    
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return loader

TRAIN_DATA_LOADER = make_dataloader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
TEST_DATA_LOADER = make_dataloader(dataset=TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)


betas: Float[torch.Tensor, 'T'] = cosine_beta_schedule(T).to(DEVICE)
alphas: Float[torch.Tensor, 'T'] = (1 - betas).to(DEVICE)
alpha_bars: Float[torch.Tensor, 'T'] = alphas.cumprod(dim=-1).to(DEVICE)
sigmas: Float[torch.Tensor, 'T'] = betas.sqrt().to(DEVICE)


model_config = {
    'data': {
        'image_size': H,  # Will be updated when dataset is loaded
    },
    'model': {
        'type': "simple",
        'in_channels': IN_CHANNELS,  # Will be updated when dataset is loaded
        'out_ch': OUT_CHANNELS,      # Will be updated when dataset is loaded
        'ch': 128,
        'ch_mult': [1, 2, 2,],
        'num_res_blocks': NUM_RES_BLOCK,
        'attn_resolutions': [1, ],
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
model_config = dict2namespace(model_config)


