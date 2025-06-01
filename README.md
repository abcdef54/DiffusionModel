# Diffusion Model Implementation

A simple diffusion model implementation for MNIST digit generation, built for educational purposes.

## Overview

This project implements a denoising diffusion probabilistic model (DDPM) that learns to generate MNIST digits by reversing a noise diffusion process. The model is trained to predict noise at each timestep, enabling it to gradually denoise random samples into realistic digit images.

## Architecture

- **Model**: U-Net with residual blocks, attention mechanisms, and skip connections
- **Dataset**: MNIST (28x28 grayscale digits)
- **Diffusion Steps**: 50 timesteps with cosine beta schedule
- **Training**: MSE loss on noise prediction

## Key Files

- `train.py` - Main training and inference pipeline
- `src/config.py` - Model parameters and diffusion schedule
- `src/modified_pixelCNN.py` - U-Net architecture implementation

## Usage

```bash
# Train the model (uncomment training code in main())
python train.py

# Run inference only
python train.py  # with training code commented out
```

## Results

The model generates recognizable MNIST digits after minimal training, demonstrating the effectiveness of the diffusion approach even with limited epochs.

## Acknowledgments

Based on the tutorial by Michael Wornow: https://michaelwornow.net/2023/07/01/diffusion-models-from-scratch with modifications for learning purposes.
