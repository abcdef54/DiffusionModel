import torch
from tqdm import tqdm
from typing import List, Tuple
from . import config, utils
import time


def train(model: torch.nn.Module,
          train_data_loader: torch.utils.data.DataLoader,
          test_data_loader: torch.utils.data.DataLoader,
          n_epochs: int, optimizer: torch.optim.Optimizer,
          logging_steps: int,
          model_name: str = '',
          checkpoint: bool = True,
          suffix: str = '.pth',
          valid: bool = True,
          valid_interval: int = 1) -> Tuple[List[float], List[float]]:

    model = model.to(config.DEVICE)

    train_losses = []
    val_losses = []

    train_loss = 0.0
    start = time.time()
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for batch_idx, (x_0,_) in enumerate(train_data_loader):
            x_0 = x_0.to(config.DEVICE)
            B = x_0.shape[0] # Batch size

            # Sample t ~ Uniform(0, T-1)
            t: torch.Tensor = torch.randint(0, config.T, (B,)).to(config.DEVICE) # Sample a total amount of 'B' 'T's
            # Sample noise vector e ~ N(0, I)
            eps: torch.Tensor = torch.randn_like(x_0).to(config.DEVICE)
            # Calculate x_t
            x_0_coef = torch.sqrt(config.alpha_bars[t]).view(-1,1,1,1)
            eps_coef = torch.sqrt(1 - config.alpha_bars[t]).view(-1,1,1,1)

            x_t = x_0_coef*x_0 + eps*eps_coef # Forward diffusion
            # eps theta
            # Reverse diffusion - predict the noise
            eps_theta = model(x_t, t)
            # Calculate MSE loss
            loss = torch.mean((eps - eps_theta)**2)
            
            # Gradient descent to update model params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_loss += loss.item()

            if (batch_idx + 1) % logging_steps == 0:
                avg_loss = train_loss / logging_steps
                print(f'Epoch {epoch+1}/{n_epochs}, '
                    f'Batch {batch_idx+1}, '
                    f'Loss: {avg_loss:.4f}, '
                    f'Time: {time.time() - start:.4f}s')
                train_loss = 0.0
                start = time.time()
            
        # Validation
        if valid and (epoch + 1) % valid_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch_idx, (x_0, _) in enumerate(test_data_loader):
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

                val_loss /= len(test_data_loader)
                val_losses.append(val_loss)
                print(f'Validation Loss after epoch {epoch + 1}: {val_loss:.4f}')

                if val_loss < min(val_losses):
                    model_name = model_name + f'epoch_{epoch}_BEST'
                    path = utils.save_model(model, model_name, suffix=suffix)
                    print(f'Saved new best model at: {path}')
        
        if checkpoint:
            model_name = model_name + f'epoch_{epoch}'
            path = utils.save_model(model, model_name, suffix=suffix)
            print(f'Saved model at: {path}')
        
    return train_losses, val_losses


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
            z: torch.Tensor = torch.rand_like(x_t) if t > 0 else torch.zeros_like(x_t)

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
