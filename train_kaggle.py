#!/usr/bin/env python3
"""
Train Kaggle Script for Diffusion Model
=======================================

This script is designed to train a diffusion model on Kaggle with the CIFAR10 dataset.
It uses the current project configurations and generates loss curves after training.

Usage:
    python train_kaggle.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--model-name MODEL_NAME]

Example:
    python train_kaggle.py --epochs 5 --batch-size 16 --lr 2e-4 --model-name "cifar10_diffusion"
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

# Set CUDA memory allocation configuration BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import project modules
try:
    from src import config, modified_Unet, utils, diffusion, datasets
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def setup_kaggle_environment():
    """Setup environment for Kaggle training"""
    print("🔧 Setting up Kaggle environment...")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) available")
    else:
        print("⚠️  Using CPU - training will be slower")
    
    # Create necessary directories
    directories = [config.DATA_PATH, config.MODEL_PATH, config.OUTPUT_PATH]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for outputs
    (config.OUTPUT_PATH / 'grid').mkdir(exist_ok=True)
    (config.OUTPUT_PATH / 'clear').mkdir(exist_ok=True)
    (config.OUTPUT_PATH / 'loss_curves').mkdir(exist_ok=True)
    
    print("✅ Environment setup complete!")


def load_cifar10_dataset(batch_size: int = None) -> tuple:
    """Load and prepare CIFAR10 dataset"""
    print("📂 Loading CIFAR10 dataset...")
    
    if batch_size is not None:
        config.BATCH_SIZE = batch_size
    
    # Download and setup CIFAR10
    train_dataset, test_dataset, in_channels, out_channels, height, width = datasets.download_CIFAR10()
    
    print(f"✅ CIFAR10 dataset loaded successfully!")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    print(f"   Image shape: {in_channels} x {height} x {width}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    
    return train_dataset, test_dataset


def setup_model_and_optimizer(learning_rate: float = None, model_path: str = None) -> tuple:
    """Initialize model and optimizer, optionally loading from checkpoint"""
    print("🏗️  Setting up model and optimizer...")
    
    if learning_rate is not None:
        config.LR = learning_rate
    
    # Create model
    model = modified_Unet.Model(config.model_config).to(config.DEVICE)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    start_epoch = 0
    
    # Load pre-trained model if specified
    if model_path is not None:
        start_epoch = load_model_checkpoint(model, optimizer, model_path)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Learning rate: {config.LR}")
    print(f"   Device: {config.DEVICE}")
    if model_path:
        print(f"   Loaded from checkpoint: {model_path}")
        print(f"   Starting from epoch: {start_epoch + 1}")
    
    return model, optimizer, start_epoch


def load_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, checkpoint_path: str) -> int:
    """Load model and optimizer state from checkpoint"""
    print(f"💾 Loading checkpoint from: {checkpoint_path}")
    
    # Handle different path formats
    if not checkpoint_path.startswith('/'):
        # If it's a relative path, check in the models directory first
        full_path = config.MODEL_PATH / checkpoint_path
        if not full_path.exists():
            # If not found in models directory, treat as relative to current directory
            full_path = Path(checkpoint_path)
    else:
        full_path = Path(checkpoint_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {full_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(full_path, map_location=config.DEVICE)
        
        # If checkpoint is just state_dict (old format), load it directly
        if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
            model.load_state_dict(checkpoint)
            start_epoch = 0
            print("✅ Loaded model state dict (no optimizer state or epoch info)")
        else:
            # New format with full checkpoint info
            model.load_state_dict(checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint)))
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Loaded optimizer state")
            
            # Get starting epoch
            start_epoch = checkpoint.get('epoch', 0)
            
            # Load other training info if available
            if 'loss' in checkpoint:
                print(f"   Last recorded loss: {checkpoint['loss']:.6f}")
            if 'learning_rate' in checkpoint:
                print(f"   Checkpoint learning rate: {checkpoint['learning_rate']}")
        
        print(f"✅ Successfully loaded checkpoint from epoch {start_epoch}")
        return start_epoch
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Continuing with fresh model...")
        return 0


def save_full_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                        loss: float, model_name: str, is_best: bool = False) -> Path:
    """Save full checkpoint with model, optimizer, and training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate': config.LR,
        'model_config': config.model_config,
        'timestamp': time.time()
    }
    
    if is_best:
        checkpoint_name = f"{model_name}_best_checkpoint.pth"
    else:
        checkpoint_name = f"{model_name}_epoch_{epoch}_checkpoint.pth"
    
    checkpoint_path = config.MODEL_PATH / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def cleanup_old_best_models(model_name: str, current_best_path: Path):
    """Delete old best model files to save space"""
    model_dir = config.MODEL_PATH
    best_pattern = f"{model_name}_*_BEST.pth"
    
    # Find all best model files
    import glob
    old_best_files = glob.glob(str(model_dir / best_pattern))
    
    for old_file in old_best_files:
        old_path = Path(old_file)
        # Don't delete the current best model
        if old_path != current_best_path:
            try:
                old_path.unlink()
                print(f"🗑️  Deleted old best model: {old_path.name}")
            except Exception as e:
                print(f"⚠️  Could not delete {old_path.name}: {e}")


def print_training_config():
    """Print current training configuration"""
    print("\n" + "="*60)
    print("📋 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {config.CURRENT_DATASET}")
    print(f"Image size: {config.IN_CHANNELS} x {config.H} x {config.W}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.N_EPOCHS}")
    print(f"Learning rate: {config.LR}")
    print(f"Diffusion timesteps (T): {config.T}")
    print(f"Dropout rate: {config.DROP_RATE}")
    print(f"Residual blocks: {config.NUM_RES_BLOCK}")
    print(f"Device: {config.DEVICE}")
    print(f"Logging steps: {config.LOGGING_STEPS}")
    print("="*60 + "\n")


def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                test_loader: DataLoader,
                optimizer: optim.Optimizer,
                epochs: int,
                model_name: str,
                start_epoch: int = 0) -> Dict[str, List[float]]:
    """Train the diffusion model"""
    print(f"🚀 Starting training for {epochs} epochs (starting from epoch {start_epoch + 1})...")
    
    # Train the model
    loss_dict = diffusion.train(
        model=model,
        train_data_loader=train_loader,
        test_data_loader=test_loader,
        n_epochs=epochs,
        optimizer=optimizer,
        logging_steps=config.LOGGING_STEPS,
        model_name=model_name,
        checkpoint=True,  # Save checkpoints
        suffix='.pth',
        valid=True,
        valid_interval=1,
        start_epoch=start_epoch
    )
    
    print("✅ Training completed successfully!")
    return loss_dict


def plot_and_save_loss_curves(loss_dict: Optional[Dict[str, List[float]]], model_name: str):
    """Plot and save loss curves"""
    if loss_dict is None:
        print("⚠️  No loss data available (inference-only mode)")
        return
        
    print("📊 Generating loss curves...")
    
    # Create a comprehensive loss plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss per epoch
    if loss_dict.get('train_loss_per_epoch'):
        axes[0, 0].plot(loss_dict['train_loss_per_epoch'], 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Training Loss per Epoch', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Validation loss per epoch
    if loss_dict.get('val_loss_per_epoch'):
        axes[0, 1].plot(loss_dict['val_loss_per_epoch'], 'r-', linewidth=2, label='Validation Loss')
        axes[0, 1].set_title('Validation Loss per Epoch', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Training and validation loss comparison
    if loss_dict.get('train_loss_per_epoch') and loss_dict.get('val_loss_per_epoch'):
        axes[1, 0].plot(loss_dict['train_loss_per_epoch'], 'b-', linewidth=2, label='Training')
        axes[1, 0].plot(loss_dict['val_loss_per_epoch'], 'r-', linewidth=2, label='Validation')
        axes[1, 0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Training loss per batch (smoothed)
    if loss_dict.get('train_loss_per_batch'):
        # Apply smoothing to batch losses for better visualization
        batch_losses = loss_dict['train_loss_per_batch']
        if len(batch_losses) > 100:
            # Apply moving average for smoother curve
            window_size = max(1, len(batch_losses) // 50)
            smoothed_losses = []
            for i in range(len(batch_losses)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(batch_losses), i + window_size // 2 + 1)
                smoothed_losses.append(np.mean(batch_losses[start_idx:end_idx]))
            axes[1, 1].plot(smoothed_losses, 'g-', linewidth=1, alpha=0.8, label='Smoothed Training Loss')
        else:
            axes[1, 1].plot(batch_losses, 'g-', linewidth=1, alpha=0.7, label='Training Loss per Batch')
        
        axes[1, 1].set_title('Training Loss per Batch', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save the plot
    loss_curve_path = config.OUTPUT_PATH / 'loss_curves' / f'{model_name}_loss_curves.png'
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curves saved to: {loss_curve_path}")
    
    # Also save loss data as numpy arrays for future analysis
    loss_data_path = config.OUTPUT_PATH / 'loss_curves' / f'{model_name}_loss_data.npz'
    np.savez(loss_data_path, **loss_dict)
    print(f"✅ Loss data saved to: {loss_data_path}")
    
    # Display the plot
    plt.show()
    
    # Print final loss statistics
    print("\n📈 Final Loss Statistics:")
    if loss_dict.get('train_loss_per_epoch'):
        print(f"   Final training loss: {loss_dict['train_loss_per_epoch'][-1]:.6f}")
        print(f"   Best training loss: {min(loss_dict['train_loss_per_epoch']):.6f}")
    if loss_dict.get('val_loss_per_epoch'):
        print(f"   Final validation loss: {loss_dict['val_loss_per_epoch'][-1]:.6f}")
        print(f"   Best validation loss: {min(loss_dict['val_loss_per_epoch']):.6f}")


def generate_sample_images(model: nn.Module, model_name: str, n_samples: int = 4, show_process: bool = False):
    """Generate sample images from the trained model"""
    print(f"🎨 Generating {n_samples} sample images...")
    
    # Generate images
    generated_images = diffusion.inference(model, n_samples)
    
    # Save final sample images
    utils.show_final_image(
        generated_images, 
        title=f'Generated Images - {model_name}',
        figname=f'{model_name}_samples',
        savefig=True,
        show=True
    )
    
    # Optionally show the generation process
    if show_process:
        print("🎬 Saving generation process images...")
        # Show images at different steps of the denoising process
        step_indices = [0, config.T//4, config.T//2, 3*config.T//4, config.T-1]  # Show process at key steps
        process_images = generated_images[:, step_indices]  # Select specific steps
        
        utils.show_grid(
            process_images.unsqueeze(0),  # Add time dimension for show_grid
            title=f'Generation Process - {model_name}',
            figname=f'{model_name}_process',
            savefig=True,
            show=True
        )
    
    print("✅ Sample images generated and saved!")
    return generated_images


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Diffusion Model on CIFAR10')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--model-name', type=str, default='cifar10_diffusion_kaggle', help='Model name for saving')
    parser.add_argument('--load-model', type=str, default=None, help='Path to pre-trained model checkpoint to continue training')
    parser.add_argument('--generate-samples', action='store_true', help='Generate sample images after training')
    parser.add_argument('--n-samples', type=int, default=4, help='Number of sample images to generate')
    parser.add_argument('--show-process', action='store_true', help='Show the generation process (denoising steps)')
    parser.add_argument('--inference-only', action='store_true', help='Skip training and only generate images from loaded model')
    
    args = parser.parse_args()
    
    if args.inference_only:
        print("🌟 Starting CIFAR10 Diffusion Model Inference")
    else:
        print("🌟 Starting CIFAR10 Diffusion Model Training on Kaggle")
    print("="*60)
    
    # Setup environment
    setup_kaggle_environment()
    
    # Load dataset
    train_dataset, test_dataset = load_cifar10_dataset(args.batch_size)
    
    # Update epochs if specified
    if args.epochs is not None:
        config.N_EPOCHS = args.epochs
    
    # Setup model and optimizer
    model, optimizer, start_epoch = setup_model_and_optimizer(args.lr, args.load_model)
    
    # Print configuration
    print_training_config()
    
    # Start training or inference
    start_time = time.time()
    
    try:
        if args.inference_only:
            if not args.load_model:
                print("❌ Error: --inference-only requires --load-model to be specified")
                sys.exit(1)
            print("🎨 Running inference only (no training)...")
            loss_dict = None
        else:
            loss_dict = train_model(
                model=model,
                train_loader=config.TRAIN_DATA_LOADER,
                test_loader=config.TEST_DATA_LOADER,
                optimizer=optimizer,
                epochs=config.N_EPOCHS,
                model_name=args.model_name,
                start_epoch=start_epoch
            )
        
        training_time = time.time() - start_time
        print(f"⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Plot and save loss curves
        plot_and_save_loss_curves(loss_dict, args.model_name)
        
        # Generate sample images if requested
        if args.generate_samples:
            generate_sample_images(model, args.model_name, args.n_samples)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved in: {config.MODEL_PATH}")
        print(f"📊 Loss curves saved in: {config.OUTPUT_PATH / 'loss_curves'}")
        print(f"🖼️  Generated images saved in: {config.OUTPUT_PATH}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        # Save current model state
        emergency_save_path = utils.save_model(model, f"{args.model_name}_interrupted", suffix='.pth')
        print(f"💾 Emergency save completed: {emergency_save_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save model state
        try:
            emergency_save_path = utils.save_model(model, f"{args.model_name}_error", suffix='.pth')
            print(f"💾 Emergency save completed: {emergency_save_path}")
        except:
            print("❌ Could not save model state")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
