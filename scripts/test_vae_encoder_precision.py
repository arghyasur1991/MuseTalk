#!/usr/bin/env python3
"""
Test VAE Encoder Precision: PyTorch vs ONNX
This script directly compares VAE encoder outputs to identify precision issues
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

def test_vae_encoder_precision():
    """Test VAE encoder precision comparing PyTorch vs ONNX directly"""
    
    print("=== VAE Encoder Precision Test ===")
    
    # Load PyTorch VAE
    device = "cpu"
    vae, _, _ = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    
    # Load ONNX VAE encoder
    providers = ['CPUExecutionProvider']
    vae_encoder_session = ort.InferenceSession("./models/onnx/vae_encoder_v15_hq.onnx", providers=providers)
    
    # Load test image
    img_path = "temp_comparison_imgs/00000000.png"
    test_image = cv2.imread(img_path)
    test_image = cv2.resize(test_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Test image range: [{test_image.min()}, {test_image.max()}]")
    
    # === Test 1: Full Image Encoding ===
    print("\n--- Test 1: Full Image Encoding ---")
    
    # PyTorch VAE encoding (full image)
    ref_image_full = vae.preprocess_img(test_image, half_mask=False)
    print(f"PyTorch preprocessed shape: {ref_image_full.shape}")
    print(f"PyTorch preprocessed range: [{ref_image_full.min():.6f}, {ref_image_full.max():.6f}]")
    
    with torch.no_grad():
        pytorch_latent_dist = vae.vae.encode(ref_image_full.to(vae.vae.dtype)).latent_dist
        pytorch_latents_mode = vae.scaling_factor * pytorch_latent_dist.mode()
        pytorch_latents_sample = vae.scaling_factor * pytorch_latent_dist.sample()
    
    print(f"PyTorch latents (mode): {pytorch_latents_mode.shape}, range: [{pytorch_latents_mode.min():.6f}, {pytorch_latents_mode.max():.6f}]")
    print(f"PyTorch latents (sample): {pytorch_latents_sample.shape}, range: [{pytorch_latents_sample.min():.6f}, {pytorch_latents_sample.max():.6f}]")
    
    # ONNX VAE encoding (using same preprocessing as PyTorch)
    # Convert PyTorch tensor to numpy for ONNX
    onnx_input = ref_image_full.cpu().numpy()
    print(f"ONNX input shape: {onnx_input.shape}")
    print(f"ONNX input range: [{onnx_input.min():.6f}, {onnx_input.max():.6f}]")
    
    onnx_latents = vae_encoder_session.run(['latents'], {'image': onnx_input})[0]
    print(f"ONNX latents: {onnx_latents.shape}, range: [{onnx_latents.min():.6f}, {onnx_latents.max():.6f}]")
    
    # Compare PyTorch mode() vs ONNX
    diff_mode = np.abs(pytorch_latents_mode.numpy() - onnx_latents)
    print(f"PyTorch mode() vs ONNX MAE: {np.mean(diff_mode):.6f}")
    print(f"PyTorch mode() vs ONNX Max Diff: {np.max(diff_mode):.6f}")
    
    # Compare PyTorch sample() vs ONNX  
    diff_sample = np.abs(pytorch_latents_sample.numpy() - onnx_latents)
    print(f"PyTorch sample() vs ONNX MAE: {np.mean(diff_sample):.6f}")
    print(f"PyTorch sample() vs ONNX Max Diff: {np.max(diff_sample):.6f}")
    
    # === Test 2: Half-Masked Image Encoding ===
    print("\n--- Test 2: Half-Masked Image Encoding ---")
    
    # PyTorch VAE encoding (half-masked)
    ref_image_masked = vae.preprocess_img(test_image, half_mask=True)
    print(f"PyTorch masked preprocessed range: [{ref_image_masked.min():.6f}, {ref_image_masked.max():.6f}]")
    
    with torch.no_grad():
        pytorch_masked_dist = vae.vae.encode(ref_image_masked.to(vae.vae.dtype)).latent_dist
        pytorch_masked_mode = vae.scaling_factor * pytorch_masked_dist.mode()
    
    print(f"PyTorch masked latents (mode): range: [{pytorch_masked_mode.min():.6f}, {pytorch_masked_mode.max():.6f}]")
    
    # ONNX VAE encoding (half-masked, using manual preprocessing to match exactly)
    image_normalized = test_image.astype(np.float32) / 255.0
    image_tensor_masked = np.transpose(image_normalized, (2, 0, 1))  # [C, H, W]
    
    # Apply mask to [C, H, W] tensor (same as PyTorch)
    mask_tensor = np.zeros((256, 256), dtype=np.float32)
    mask_tensor[:256//2, :] = 1  # Upper half = 1, lower half = 0
    for c in range(3):
        image_tensor_masked[c] = image_tensor_masked[c] * mask_tensor
    
    # Apply normalization per channel
    for c in range(3):
        image_tensor_masked[c] = (image_tensor_masked[c] - 0.5) / 0.5
    
    # Add batch dimension
    image_tensor_masked = np.expand_dims(image_tensor_masked, 0)
    
    print(f"ONNX masked input range: [{image_tensor_masked.min():.6f}, {image_tensor_masked.max():.6f}]")
    
    onnx_masked_latents = vae_encoder_session.run(['latents'], {'image': image_tensor_masked})[0]
    print(f"ONNX masked latents: range: [{onnx_masked_latents.min():.6f}, {onnx_masked_latents.max():.6f}]")
    
    # Compare masked latents
    diff_masked = np.abs(pytorch_masked_mode.numpy() - onnx_masked_latents)
    print(f"PyTorch vs ONNX masked MAE: {np.mean(diff_masked):.6f}")
    print(f"PyTorch vs ONNX masked Max Diff: {np.max(diff_masked):.6f}")
    
    # === Test 3: Verify input preprocessing exactly matches ===
    print("\n--- Test 3: Input Preprocessing Verification ---")
    
    # Check if PyTorch preprocessing exactly matches our manual ONNX preprocessing
    # Use the EXACT same order as PyTorch and fixed ONNX inference
    manual_normalized = test_image.astype(np.float32) / 255.0
    manual_tensor = np.transpose(manual_normalized, (2, 0, 1))  # [C, H, W]
    # Apply normalization per channel
    for c in range(3):
        manual_tensor[c] = (manual_tensor[c] - 0.5) / 0.5
    # Add batch dimension
    manual_full = np.expand_dims(manual_tensor, 0)
    
    preprocessing_diff = np.abs(ref_image_full.cpu().numpy() - manual_full)
    print(f"PyTorch vs Manual preprocessing MAE: {np.mean(preprocessing_diff):.6f}")
    print(f"PyTorch vs Manual preprocessing Max Diff: {np.max(preprocessing_diff):.6f}")
    
    if np.mean(preprocessing_diff) < 1e-6:
        print("✅ Preprocessing is identical")
    else:
        print("❌ Preprocessing differs - this might be the issue!")
        
    # === Summary ===
    print("\n=== SUMMARY ===")
    if np.mean(diff_mode) < 0.01:
        print("✅ VAE encoder ONNX export is high quality (mode)")
    else:
        print("❌ VAE encoder ONNX export has significant differences")
        
    if np.mean(diff_sample) < 0.01:
        print("✅ VAE encoder ONNX export is high quality (sample)")
    else:
        print("❌ VAE encoder ONNX export has significant differences vs sample")
        
    print(f"Best VAE encoder difference: {min(np.mean(diff_mode), np.mean(diff_sample)):.6f}")

if __name__ == "__main__":
    test_vae_encoder_precision() 