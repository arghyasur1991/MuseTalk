#!/usr/bin/env python3
"""
Test Deterministic PyTorch VAE vs ONNX VAE
This script compares PyTorch VAE in deterministic mode vs ONNX VAE
"""

import os
import sys
import cv2
import torch
import numpy as np
import onnxruntime as ort

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.utils import load_all_model

def test_deterministic_pytorch_vs_onnx():
    """Test deterministic PyTorch VAE vs ONNX VAE"""
    
    print("=== Deterministic PyTorch VAE vs ONNX VAE Test ===")
    
    # Load PyTorch VAE in deterministic mode
    device = "cpu"
    
    # Temporarily modify load_all_model to use deterministic VAE
    from musetalk.models.vae import VAE
    
    # Create deterministic VAE
    vae_deterministic = VAE(
        model_path="./models/sd-vae",
        deterministic=True  # Use mode() instead of sample()
    )
    
    # Load ONNX VAE encoder
    providers = ['CPUExecutionProvider']
    vae_encoder_session = ort.InferenceSession("./models/onnx/vae_encoder_v15_hq.onnx", providers=providers)
    
    # Load test image
    img_path = "temp_comparison_imgs/00000000.png"
    test_image = cv2.imread(img_path)
    test_image = cv2.resize(test_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    print(f"Test image shape: {test_image.shape}")
    
    # === Test 1: Full Image Encoding ===
    print("\n--- Test 1: Full Image Encoding (Deterministic) ---")
    
    # PyTorch VAE encoding (deterministic)
    pytorch_latents = vae_deterministic.get_latents_for_unet(test_image)
    print(f"PyTorch latents (deterministic): {pytorch_latents.shape}, range: [{pytorch_latents.min():.6f}, {pytorch_latents.max():.6f}]")
    
    # ONNX VAE encoding (using corrected preprocessing)
    # Half-masked image
    image_normalized = test_image.astype(np.float32) / 255.0
    image_tensor_masked = np.transpose(image_normalized, (2, 0, 1))  # [C, H, W]
    
    # Apply mask to [C, H, W] tensor
    mask_tensor = np.zeros((256, 256), dtype=np.float32)
    mask_tensor[:256//2, :] = 1  # Upper half = 1, lower half = 0
    for c in range(3):
        image_tensor_masked[c] = image_tensor_masked[c] * mask_tensor
    
    # Apply normalization per channel
    for c in range(3):
        image_tensor_masked[c] = (image_tensor_masked[c] - 0.5) / 0.5
    
    # Add batch dimension
    image_tensor_masked = np.expand_dims(image_tensor_masked, 0)
    
    # Full image
    image_normalized_full = test_image.astype(np.float32) / 255.0
    image_tensor_full = np.transpose(image_normalized_full, (2, 0, 1))  # [C, H, W]
    
    # Apply normalization per channel
    for c in range(3):
        image_tensor_full[c] = (image_tensor_full[c] - 0.5) / 0.5
    
    # Add batch dimension
    image_tensor_full = np.expand_dims(image_tensor_full, 0)
    
    # ONNX VAE encoding
    masked_latents = vae_encoder_session.run(['latents'], {'image': image_tensor_masked})[0]
    ref_latents = vae_encoder_session.run(['latents'], {'image': image_tensor_full})[0]
    onnx_latents = np.concatenate([masked_latents, ref_latents], axis=1)
    
    print(f"ONNX latents: {onnx_latents.shape}, range: [{onnx_latents.min():.6f}, {onnx_latents.max():.6f}]")
    
    # Compare deterministic PyTorch vs ONNX
    latent_diff = np.abs(pytorch_latents.detach().numpy() - onnx_latents)
    print(f"\nDeterministic PyTorch vs ONNX MAE: {np.mean(latent_diff):.6f}")
    print(f"Deterministic PyTorch vs ONNX Max Diff: {np.max(latent_diff):.6f}")
    
    if np.mean(latent_diff) < 0.01:
        print("✅ Deterministic PyTorch and ONNX VAE are very similar!")
        return True
    else:
        print("❌ Deterministic PyTorch and ONNX VAE still differ significantly")
        return False

if __name__ == "__main__":
    test_deterministic_pytorch_vs_onnx() 