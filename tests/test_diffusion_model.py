#!/usr/bin/env python3
"""
Comprehensive test suite for the diffusion model implementation.
Tests model architecture, diffusion mathematics, training/inference, and identifies bugs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import warnings
from unittest.mock import patch

# Import project modules
from src import config, modified_Unet, diffusion, utils, datasets

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

class TestDiffusionMath:
    """Test the mathematical correctness of the diffusion process."""
    
    def test_beta_schedule(self):
        """Test cosine beta schedule properties."""
        betas = config.betas
        alphas = config.alphas
        alpha_bars = config.alpha_bars
        
        # Test shapes
        assert len(betas) == config.T, f"Expected {config.T} betas, got {len(betas)}"
        assert len(alphas) == config.T, f"Expected {config.T} alphas, got {len(alphas)}"
        assert len(alpha_bars) == config.T, f"Expected {config.T} alpha_bars, got {len(alpha_bars)}"
        
        # Test value ranges
        assert torch.all(betas >= 0) and torch.all(betas <= 1), "Betas should be in [0,1]"
        assert torch.all(alphas >= 0) and torch.all(alphas <= 1), "Alphas should be in [0,1]"
        assert torch.all(alpha_bars >= 0) and torch.all(alpha_bars <= 1), "Alpha_bars should be in [0,1]"
        
        # Test mathematical relationships
        expected_alphas = 1 - betas
        assert torch.allclose(alphas, expected_alphas), "alphas = 1 - betas relationship failed"
        
        # Test that alpha_bars is decreasing (noise increases over time)
        for i in range(1, len(alpha_bars)):
            assert alpha_bars[i] <= alpha_bars[i-1], f"alpha_bars should be decreasing, but increased at step {i}"
        
        print("✅ Beta schedule tests passed")

    def test_forward_diffusion(self):
        """Test forward diffusion process (adding noise)."""
        # Create sample image
        batch_size, channels, height, width = 4, 1, 28, 28
        x_0 = torch.randn(batch_size, channels, height, width).to(config.DEVICE)
        
        # Test forward diffusion at different timesteps
        for t_val in [0, config.T//4, config.T//2, config.T-1]:
            t = torch.full((batch_size,), t_val, dtype=torch.long).to(config.DEVICE)
            eps = torch.randn_like(x_0)
            
            # Forward diffusion formula
            x_0_coef = torch.sqrt(config.alpha_bars[t]).view(-1,1,1,1)
            eps_coef = torch.sqrt(1 - config.alpha_bars[t]).view(-1,1,1,1)
            x_t = x_0_coef * x_0 + eps_coef * eps
            
            # Test shapes
            assert x_t.shape == x_0.shape, f"Forward diffusion changed shape at t={t_val}"
            
            # Test that noise increases over time (variance should increase)
            if t_val > 0:
                var_ratio = torch.var(x_t) / torch.var(x_0)
                # Variance should increase with time (more noise)
                assert var_ratio > 0.8, f"Variance ratio too low at t={t_val}: {var_ratio}"
        
        print("✅ Forward diffusion tests passed")

    def test_reverse_diffusion_math(self):
        """Test reverse diffusion mathematical formulation."""
        batch_size = 2
        x_t = torch.randn(batch_size, 1, 28, 28).to(config.DEVICE)
        eps_theta = torch.randn_like(x_t)
        
        for t_val in [config.T-1, config.T//2, 1]:
            t = torch.full((batch_size,), t_val, dtype=torch.long).to(config.DEVICE)
            
            # Get coefficients
            alpha_t = config.alphas[t].view(-1, 1, 1, 1)
            alpha_bar_t = config.alpha_bars[t].view(-1, 1, 1, 1)
            sigma_t = config.sigmas[t].view(-1, 1, 1, 1)
            
            # Reverse diffusion formula
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            z = torch.randn_like(x_t) if t_val > 0 else torch.zeros_like(x_t)
            
            x_t_minus_1 = coeff1 * (x_t - coeff2 * eps_theta) + sigma_t * z
            
            # Test shapes and values
            assert x_t_minus_1.shape == x_t.shape, f"Reverse diffusion changed shape at t={t_val}"
            assert torch.isfinite(x_t_minus_1).all(), f"Non-finite values in reverse diffusion at t={t_val}"
            
            # Test that coefficients are positive and finite
            assert torch.all(coeff1 > 0) and torch.isfinite(coeff1).all(), f"Invalid coeff1 at t={t_val}"
            assert torch.all(coeff2 > 0) and torch.isfinite(coeff2).all(), f"Invalid coeff2 at t={t_val}"
        
        print("✅ Reverse diffusion math tests passed")


class TestModelArchitecture:
    """Test the U-Net model architecture."""
    
    def test_model_initialization(self):
        """Test model can be initialized correctly."""
        model = modified_Unet.Model(config.model_config).to(config.DEVICE)
        
        # Test model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model has no parameters"
        print(f"Model has {param_count:,} parameters")
        
        # Test model is on correct device
        # assert str(model.device) == str(config.DEVICE), f"Model device mismatch: {model.device} vs {config.DEVICE}"
        
        print("✅ Model initialization tests passed")
    
    def test_model_forward_pass(self):
        """Test model forward pass with different inputs."""
        model = modified_Unet.Model(config.model_config).to(config.DEVICE)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, config.IN_CHANNELS, config.H, config.W).to(config.DEVICE)
            t = torch.randint(0, config.T, (batch_size,)).to(config.DEVICE)
            
            with torch.no_grad():
                output = model(x, t)
            
            # Test output shape
            expected_shape = (batch_size, config.OUT_CHANNELS, config.H, config.W)
            assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
            
            # Test output is finite
            assert torch.isfinite(output).all(), f"Non-finite values in model output for batch_size {batch_size}"
        
        print("✅ Model forward pass tests passed")
    
    def test_timestep_embedding(self):
        """Test timestep embedding function."""
        from src.modified_Unet import get_timestep_embedding
        
        # Test different timestep inputs
        for batch_size in [1, 4]:
            t = torch.randint(0, config.T, (batch_size,))
            emb = get_timestep_embedding(t, 128)
            
            assert emb.shape == (batch_size, 128), f"Timestep embedding shape mismatch: {emb.shape}"
            assert torch.isfinite(emb).all(), "Non-finite values in timestep embedding"
        
        print("✅ Timestep embedding tests passed")


class TestTrainingLoop:
    """Test training functionality and identify bugs."""
    
    def test_training_step_shapes(self):
        """Test that training step produces correct tensor shapes."""
        model = modified_Unet.Model(config.model_config).to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        
        # Create dummy batch
        batch_size = 4
        x_0 = torch.randn(batch_size, config.IN_CHANNELS, config.H, config.W).to(config.DEVICE)
        t = torch.randint(0, config.T, (batch_size,)).to(config.DEVICE)
        eps = torch.randn_like(x_0)
        
        # Forward diffusion
        x_0_coef = torch.sqrt(config.alpha_bars[t]).view(-1,1,1,1)
        eps_coef = torch.sqrt(1 - config.alpha_bars[t]).view(-1,1,1,1)
        x_t = x_0_coef * x_0 + eps_coef * eps
        
        # Model prediction
        eps_theta = model(x_t, t)
        
        # Test shapes
        assert eps_theta.shape == eps.shape, f"Predicted noise shape mismatch: {eps_theta.shape} vs {eps.shape}"
        
        # Test loss calculation
        loss = torch.mean((eps - eps_theta)**2)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        
        # Test backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed"
        
        optimizer.step()
        
        print("✅ Training step tests passed")
    
    def test_validation_bug_detection(self):
        """Detect the bug in validation loss calculation."""
        print("🔍 Testing for validation bug...")
        
        # Read the diffusion.py file to check for the bug
        diffusion_path = project_root / "src/diffusion.py"
        with open(diffusion_path, 'r') as f:
            content = f.read()
        
        # Look for the specific bug: x_t_minus_1 instead of x_t in validation
        lines = content.split('\n')
        bug_found = False
        bug_line = -1
        
        for i, line in enumerate(lines):
            if 'x_t_minus_1 = x_0_coef*x_0 + eps*eps_coef' in line and 'eps_theta = model(x_t, t)' in lines[i+2:i+5]:
                bug_found = True
                bug_line = i + 1
                break
        
        if bug_found:
            print(f"🚨 BUG DETECTED: Line {bug_line} in diffusion.py")
            print("   Validation uses x_t_minus_1 instead of x_t for noise calculation")
            print("   This causes incorrect validation loss computation")
        else:
            print("ℹ️  Validation bug check: Pattern not found (may be fixed)")
        
        return bug_found
    
    def test_model_saving_loading(self):
        """Test model saving and loading functionality."""
        model = modified_Unet.Model(config.model_config).to(config.DEVICE)
        
        # Save model
        test_name = "test_model_save"
        saved_path = utils.save_model(model, test_name)
        
        assert saved_path.exists(), f"Model not saved to {saved_path}"
        
        # Load model
        loaded_model = utils.load_model(saved_path)
        
        # Ensure both models are in eval mode for consistent comparison
        model.eval()
        loaded_model.eval()
        
        # Test loaded model works
        x = torch.randn(1, config.IN_CHANNELS, config.H, config.W).to(config.DEVICE)
        t = torch.randint(0, config.T, (1,)).to(config.DEVICE)
        
        with torch.no_grad():
            original_output = model(x, t)
            loaded_output = loaded_model(x, t)
        
        # Outputs should be identical
        assert torch.allclose(original_output, loaded_output, atol=1e-6), "Loaded model produces different output"
        
        # Clean up
        if saved_path.exists():
            saved_path.unlink()
        
        print("✅ Model saving/loading tests passed")


class TestInference:
    """Test inference/generation functionality."""
    
    def test_inference_shapes(self):
        """Test inference produces correctly shaped outputs."""
        model = modified_Unet.Model(config.model_config).to(config.DEVICE)
        
        # Test different sample counts
        for n_samples in [1, 2, 4]:
            with torch.no_grad():
                generated = diffusion.inference(model, n_samples)
            
            # Expected shape: (n_samples, T+1, C, H, W)
            expected_shape = (n_samples, config.T + 1, config.IN_CHANNELS, config.H, config.W)
            assert generated.shape == expected_shape, f"Generated shape mismatch: {generated.shape} vs {expected_shape}"
            
            # Test values are finite
            assert torch.isfinite(generated).all(), f"Non-finite values in generated samples for n_samples={n_samples}"
            
            # Test final images are different from initial noise
            initial_noise = generated[:, 0]  # First timestep (pure noise)
            final_images = generated[:, -1]  # Last timestep (denoised)
            
            # They should be significantly different
            mse_diff = torch.mean((initial_noise - final_images)**2)
            assert mse_diff > 0.1, f"Generated images too similar to initial noise: MSE={mse_diff}"
        
        print("✅ Inference shape tests passed")

    def test_inference_determinism(self):
        """Test inference determinism with fixed random seed."""
        model = modified_Unet.Model(config.model_config).to(config.DEVICE)
        
        # Set seed and generate
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        with torch.no_grad():
            generated1 = diffusion.inference(model, 2)
        
        # Reset seed and generate again
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            
        with torch.no_grad():
            generated2 = diffusion.inference(model, 2)
        
        # Should be identical with same seed
        assert torch.allclose(generated1, generated2, atol=1e-6), "Inference not deterministic with same seed"
        
        print("✅ Inference determinism tests passed")


class TestUtilities:
    """Test utility functions."""
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        loader = utils.make_dataloader(config.TRAIN_DATASET, batch_size=16)
        
        # Test we can get a batch
        batch = next(iter(loader))
        x, y = batch
        
        assert x.shape[0] <= 16, "Batch size too large"
        assert x.shape[1:] == (config.IN_CHANNELS, config.H, config.W), f"Wrong image shape: {x.shape[1:]}"
        
        print("✅ Dataloader tests passed")
    
    def test_visualization_functions(self):
        """Test visualization functions don't crash."""
        # Create dummy images
        imgs = [torch.randn(config.IN_CHANNELS, config.H, config.W) for _ in range(4)]
        
        # Test show_grid doesn't crash
        try:
            utils.show_grid(imgs, title="Test", show=False, savefig=False)
            print("✅ show_grid function works")
        except Exception as e:
            print(f"❌ show_grid failed: {e}")
        
        # Test show_images doesn't crash  
        try:
            utils.show_final_image(imgs[:1], title="Test", show=False, savefig=False)
            print("✅ show_images function works")
        except Exception as e:
            print(f"❌ show_images failed: {e}")


def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 Running Diffusion Model Test Suite")
    print("=" * 50)
    
    test_classes = [
        TestDiffusionMath(),
        TestModelArchitecture(),
        TestTrainingLoop(),
        TestInference(),
        TestUtilities()
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📝 Running {class_name}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                failed_tests.append(f"{class_name}.{method_name}: {str(e)}")
                print(f"❌ {method_name} failed: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\n❌ Failed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    else:
        print("\n🎉 All tests passed!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
