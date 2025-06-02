# Diffusion Model Implementation

A comprehensive implementation of Denoising Diffusion Probabilistic Models (DDPM) with an interactive command-line interface for training and generating images from multiple datasets.

## 🏗️ Architecture

The model implements a U-Net architecture specifically designed for diffusion models:

- **Encoder-Decoder Structure**: Symmetric downsampling and upsampling paths
- **Residual Blocks**: ResNet-style blocks with time embedding injection
- **Self-Attention**: Applied at specified resolutions for global coherence
- **Skip Connections**: Feature concatenation between encoder and decoder
- **Time Embeddings**: Sinusoidal positional encoding for diffusion timesteps
- **Configurable Channels**: Base channels: 128, multipliers: [1, 2, 2]

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DiffusionModel
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data models output
```

## Usage

### Quick Start

Run the interactive interface:
```bash
python main.py
```

### Main Menu Options

1. **Train Model** - Train the diffusion model on selected dataset
2. **Generate Images** - Create new images using trained model
3. **Load Model** - Load previously saved model checkpoints
4. **Choose Dataset** - Switch between MNIST, CIFAR-10, CIFAR-100, CelebA
5. **Configure Transforms** - Set up image preprocessing pipeline
6. **Save Model** - Save current model state
7. **View Config** - Display current training configuration
8. **Change Device** - Switch between CPU, CUDA, MPS
9. **Set Hyperparameters** - Adjust training parameters
10. **Get Context** - Display comprehensive help and documentation

### Typical Workflow

1. **Choose Dataset**: Select from available datasets (MNIST recommended for first run)
2. **Configure Settings**: Set hyperparameters, transforms, and device
3. **Train Model**: Train the diffusion model (default: 3 epochs)
4. **Generate Images**: Create new images from the trained model
5. **Save Results**: Models and generated images are automatically saved

### Example Training Session

```bash
# Start the program
python main.py

# Menu choices:
# 4 -> Choose MNIST dataset
# 9 -> Set epochs to 10, batch size to 64
# 8 -> Set device to CUDA (if available)
# 1 -> Train model with name "mnist_model"
# 2 -> Generate 16 images
```

## 📁 Project Structure

```
DiffusionModel/
├── main.py                    # Interactive CLI interface
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration and hyperparameters
│   ├── modified_Unet.py      # U-Net model architecture
│   ├── diffusion.py          # Training and inference functions
│   ├── utils.py              # Utility functions (save/load, visualization)
│   └── datasets.py           # Dataset loading functions
├── models/                   # Saved model checkpoints
├── output/                   # Generated images
├── data/                     # Downloaded datasets
├── tests/                    # Test functions to check if the program still runs correctly
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🎯 Supported Datasets

| Dataset | Image Size | Channels | Classes | Description |
|---------|------------|----------|---------|-------------|
| MNIST | 28×28 | 1 | 10 | Handwritten digits |
| CIFAR-10 | 32×32 | 3 | 10 | Natural images |
| CIFAR-100 | 32×32 | 3 | 100 | Natural images (fine-grained) |
| CelebA | 64×64 | 3 | - | Celebrity faces |

## ⚙️ Configuration

### Default Hyperparameters

- **Epochs**: 3
- **Batch Size**: 32
- **Learning Rate**: 2e-4
- **Diffusion Steps (T)**: 50
- **Dropout Rate**: 0.1
- **Residual Blocks**: 2
- **Beta Schedule**: Cosine

### Customizable Parameters

All parameters can be modified through the interactive menu:
- Training epochs and batch size
- Learning rate and optimizer settings
- Model architecture (channels, blocks, attention)
- Diffusion process parameters
- Image preprocessing transforms

## 🧠 How Diffusion Models Work

### Forward Process (Training)
1. Start with real images from the dataset
2. Gradually add Gaussian noise over T timesteps
3. Train the model to predict the noise added at each step

### Reverse Process (Generation)
1. Start with pure random noise
2. Use the trained model to predict and remove noise
3. Iteratively denoise over T timesteps to generate images

The model learns to reverse the noise corruption process, enabling it to generate new samples from noise.

## 📊 Model Performance

The implementation includes:
- **Training Loss Tracking**: Monitor loss per batch and epoch
- **Validation**: Optional validation on test set
- **Checkpointing**: Save best models and epoch checkpoints
- **Visualization**: Real-time image generation and saving

## 🔧 Advanced Usage

### Custom Transforms

Configure image preprocessing through the interactive menu:
- Resize images to custom dimensions
- Normalize to different ranges ([-1,1] or [0,1])
- Chain multiple transformations

### Device Selection

Automatically detects and allows selection of:
- **CPU**: Universal compatibility
- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon acceleration

### Model Checkpointing

- Automatic saving of best models based on validation loss
- Manual model saving with custom names
- Easy model loading and resuming training

## 🎨 Generated Samples

Generated images are saved to the `output/` directory with timestamps. The model can generate:

## 📚 References

- **DDPM Paper**: "Denoising Diffusion Probabilistic Models" by Ho et al.
- **Original Tutorial**: Based on Michael Wornow's tutorial with significant extensions
- **U-Net Architecture**: Adapted for diffusion model requirements
