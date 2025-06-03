import torch
from tqdm import tqdm
from typing import List, Dict
from . import config, utils
import time
import numpy as np


def train(model: torch.nn.Module,
          train_data_loader: torch.utils.data.DataLoader,
          test_data_loader: torch.utils.data.DataLoader,
          n_epochs: int,
          optimizer: torch.optim.Optimizer,
          logging_steps: int,
          model_name: str = '',
          checkpoint: bool = True,
          suffix: str = '.pth',
          valid: bool = True,
          valid_interval: int = 1,
          start_epoch: int = 0) -> Dict[str, List[float]]:

    model = model.to(config.DEVICE)

    train_loss_batch = []
    val_loss_batch = []
    train_loss_epoch = []
    val_loss_epoch = []

    
    for epoch in tqdm(range(start_epoch, start_epoch + n_epochs)):
        start = time.time()
        train_step_losses = train_step(model, train_data_loader, optimizer, logging_steps)
        # Store loss of each batch
        train_loss_batch.extend(train_step_losses)
        # Store mean loss of each epoch
        train_loss_epoch.append(np.mean(train_step_losses))

        print(f'Avg loss train epoch {epoch + 1}: {train_loss_epoch[-1]:.4f} | Time: {(time.time() - start):.3f}s')
        
        # Validation
        if valid and (epoch + 1) % valid_interval == 0:
            start = time.time()
            val_step_losses = validate(model, test_data_loader, logging_steps)
            # Store loss of each batch
            val_loss_batch.extend(val_step_losses)
            # Store mean loss of each epoch
            val_loss_epoch.append(np.mean(val_loss_batch))
            
            print(f'Avg loss eval epoch {epoch + 1}: {val_loss_epoch[-1]:.4f} | Time: {(time.time() - start):.3f}s')
            if len(val_loss_epoch) > 1:
                if val_loss_epoch[-1] <= min(val_loss_epoch[:-1]):
                    best_model_name = model_name + f'_epoch_{epoch+1}_BEST'
                    path = utils.save_model(model, best_model_name, suffix=suffix)
                    print(f'Saved new best model at: {path}')
                    
                    # Cleanup old best models to save space
                    try:
                        from pathlib import Path
                        import glob
                        model_dir = config.MODEL_PATH
                        best_pattern = str(model_dir / f"{model_name}_*_BEST{suffix}")
                        old_best_files = glob.glob(best_pattern)
                        
                        for old_file in old_best_files:
                            old_path = Path(old_file)
                            # Don't delete the current best model
                            if old_path != path:
                                try:
                                    old_path.unlink()
                                    print(f'🗑️  Deleted old best model: {old_path.name}')
                                except Exception as e:
                                    print(f'⚠️  Could not delete {old_path.name}: {e}')
                    except Exception as e:
                        print(f'⚠️  Error during best model cleanup: {e}')

        if checkpoint:
            checkpoint_name = model_name + f'_epoch_{epoch+1}'
            path = utils.save_model(model, checkpoint_name, suffix=suffix)
            print(f'Saved model at: {path}')
        
    result = {
        'train_loss_per_batch' : train_loss_batch,
        'train_loss_per_epoch' : train_loss_epoch,
        'val_loss_per_batch' : val_loss_batch,
        'val_loss_per_epoch' : val_loss_epoch
    }
    return result

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               logging_steps: int = 10) -> List[float]:
    model.train()
    losses = []
    train_loss = 0.0
    start = time.time()
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for batch_idx, (x_0, _) in enumerate(train_dataloader):
        x_0 = x_0.to(config.DEVICE)
        # print(f'train_step function: X_0_TENSOR_SHAPE: {x_0.shape}')
        B = x_0.shape[0] # Batch size

        # Sample t ~ Uniform(0, T-1)
        t: torch.Tensor = torch.randint(0, config.T, (B,)).to(config.DEVICE) # Sample a total amount of 'B' 'T's
        # Sample noise vector e ~ N(0, I)
        eps: torch.Tensor = torch.randn_like(x_0).to(config.DEVICE)
        # Calculate x_t
        x_0_coef = torch.sqrt(config.alpha_bars[t]).view(-1,1,1,1)
        eps_coef = torch.sqrt(1 - config.alpha_bars[t]).view(-1,1,1,1)

        x_t = x_0_coef*x_0 + eps*eps_coef # Forward diffusion
        # print(f'train_step function: X_t_TENSOR_SHAPE: {x_t.shape}')
        # eps theta
        # Reverse diffusion - predict the noise
        eps_theta = model(x_t, t)
        # print(f'train_step function: eps_theta_TENSOR_SHAPE: {eps_theta.shape}')
        # Calculate MSE loss
        loss = torch.mean((eps - eps_theta)**2)
        train_loss += loss.item()
        losses.append(loss.item())
        
        # Gradient descent to update model params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % logging_steps == 0:
                avg_loss = train_loss / logging_steps
                print(f'Batch {batch_idx+1}, '
                    f'Loss: {avg_loss:.4f}, '
                    f'Time: {time.time() - start:.4f}s')
                train_loss = 0.0
                start = time.time()
                
                # Clear CUDA cache periodically to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return losses


def validate(model: torch.nn.Module,
             test_dataloader: torch.utils.data.DataLoader,
             logging_steps: int = 10) -> List[float]:
    model.eval()
    losses = []
    batch_val_loss = 0.0
    epoch_val_loss = 0.0
    val_loss = 0.0
    start = time.time()
    with torch.inference_mode():
        for batch_idx, (x_0, _) in enumerate(test_dataloader):
            x_0 = x_0.to(config.DEVICE)
            B = x_0.shape[0] # Batch size

            t: torch.Tensor = torch.randint(0, config.T, (B,)).to(config.DEVICE)
            eps = torch.randn_like(x_0).to(config.DEVICE)

            x_0_coef = torch.sqrt(config.alpha_bars[t]).view(-1,1,1,1)
            eps_coef = torch.sqrt(1 - config.alpha_bars[t]).view(-1,1,1,1)
            x_t = x_0_coef*x_0 + eps*eps_coef

            eps_theta = model(x_t, t)

            loss = torch.mean((eps - eps_theta)**2)
            val_loss += loss.item()
            losses.append(loss.item())

            if (batch_idx + 1) % logging_steps == 0:
                avg_loss = val_loss / logging_steps
                print(f'Batch {batch_idx+1}, '
                    f'Val Loss: {avg_loss:.4f}, '
                    f'Time: {time.time() - start:.4f}s')
                val_loss = 0.0
                start = time.time()

    return losses


def inference(model: torch.nn.Module, n_samples: int) -> ...:
    model.to(config.DEVICE)
    model.eval()

    C = config.IN_CHANNELS
    W = config.W
    H = config.H
    T = config.T

    # Generate random noise for reverse diffusion
    x_t = torch.randn((n_samples,C,W,H)).to(config.DEVICE)
    x_ts = [] # Store the random noise image after every denoising step
    with torch.inference_mode():
        for t in tqdm(range(T-1, -1, -1)):
            # Add noise only if not the final step
            z: torch.Tensor = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            # Create time vector for all samples
            t_tensor: torch.Tensor = torch.full((n_samples,), t, device=config.DEVICE, dtype=torch.long)

            # Predict noise
            eps_theta = model(x_t, t_tensor)

            # Get coefficients for this timestep
            alpha_t = config.alphas[t]
            alpha_bar_t = config.alpha_bars[t]
            sigma_t = config.sigmas[t]
            
            # Reshape coefficients for broadcasting with (n_samples, C, H, W)
            alpha_t = alpha_t.view(1, 1, 1, 1)
            alpha_bar_t = alpha_bar_t.view(1, 1, 1, 1)
            sigma_t = sigma_t.view(1, 1, 1, 1)

            # Reverse diffusion step MATH
            # x_t-1 = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / (sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            x_t_minus_1 = coeff1 * (x_t - coeff2 * eps_theta) + sigma_t * z

            # Store current x_t before updating
            x_ts.append(x_t.cpu())
            x_t = x_t_minus_1

        x_ts.append(x_t.cpu())

    # Stack and return: shape will be (T+1, n_samples, C, H, W)
    # Transpose to get (n_samples, T+1, C, H, W)
    return torch.stack(x_ts).transpose(0, 1)
