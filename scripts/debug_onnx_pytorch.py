#!/usr/bin/env python3
"""
Debug PyTorch vs ONNX intermediate values to find the blurriness source
"""

import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
import sys
import pickle

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.utils import load_all_model

def debug_pytorch_vs_onnx():
    """Compare PyTorch and ONNX intermediate values step by step"""
    
    print("=== DEBUG: PyTorch vs ONNX Intermediate Values ===")
    
    # Load PyTorch models
    device = "cpu"
    
    print("Loading PyTorch models...")
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    
    # Load ONNX models
    print("Loading ONNX models...")
    providers = ['CPUExecutionProvider']
    unet_session = ort.InferenceSession("./models/onnx/unet_v15.onnx", providers=providers)
    vae_encoder_session = ort.InferenceSession("./models/onnx/vae_encoder_v15_hq.onnx", providers=providers)
    vae_decoder_session = ort.InferenceSession("./models/onnx/vae_decoder_v15_hq.onnx", providers=providers)
    pe_session = ort.InferenceSession("./models/onnx/positional_encoding_v15.onnx", providers=providers)
    
    # Load test image
    img_path = "temp_comparison_imgs/00000000.png"
    test_image = cv2.imread(img_path)
    
    # Load face coordinates
    with open("results/temp_comparison_imgs.pkl", 'rb') as f:
        coord_list = pickle.load(f)
    
    bbox = coord_list[0]
    x1, y1, x2, y2 = bbox
    y2 = y2 + 10  # v15 extra margin
    y2 = min(y2, test_image.shape[0])
    
    # Crop face region
    crop_frame = test_image[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    print(f"Test image shape: {crop_frame.shape}")
    
    # === STEP 1: VAE Encoding ===
    print("\n=== STEP 1: VAE Encoding ===")
    
    # Test PyTorch VAE with both sample() and mode() for comparison
    print("\n--- Testing PyTorch VAE: sample() vs mode() ---")
    
    # Encode using original sample() method
    pytorch_latents_sample = vae.get_latents_for_unet(crop_frame)
    print(f"PyTorch latents (sample): {pytorch_latents_sample.shape}, range: [{pytorch_latents_sample.min():.6f}, {pytorch_latents_sample.max():.6f}]")
    
    # Manually test with mode() to see if this reduces differences
    ref_image_masked = vae.preprocess_img(crop_frame, half_mask=True)
    ref_image_full = vae.preprocess_img(crop_frame, half_mask=False)
    
    with torch.no_grad():
        # Test with mode() instead of sample()
        masked_latent_dist = vae.vae.encode(ref_image_masked.to(vae.vae.dtype)).latent_dist
        ref_latent_dist = vae.vae.encode(ref_image_full.to(vae.vae.dtype)).latent_dist
        
        masked_latents_mode = vae.scaling_factor * masked_latent_dist.mode()
        ref_latents_mode = vae.scaling_factor * ref_latent_dist.mode()
        pytorch_latents_mode = torch.cat([masked_latents_mode, ref_latents_mode], dim=1)
    
    print(f"PyTorch latents (mode): {pytorch_latents_mode.shape}, range: [{pytorch_latents_mode.min():.6f}, {pytorch_latents_mode.max():.6f}]")
    
    # Compare sample() vs mode() in PyTorch
    mode_sample_diff = torch.abs(pytorch_latents_sample - pytorch_latents_mode)
    print(f"PyTorch sample() vs mode() MAE: {torch.mean(mode_sample_diff):.6f}")
    print(f"PyTorch sample() vs mode() Max Diff: {torch.max(mode_sample_diff):.6f}")
    
    # ONNX VAE encoding (matching the ONNX inference exactly)
    # Half-masked image - FIXED: Use corrected preprocessing order
    image_normalized = crop_frame.astype(np.float32) / 255.0
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
    
    # Full image - FIXED: Use corrected preprocessing order
    image_normalized_full = crop_frame.astype(np.float32) / 255.0
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
    
    print(f"\nONNX latents shape: {onnx_latents.shape}")
    print(f"ONNX latents range: [{onnx_latents.min():.6f}, {onnx_latents.max():.6f}]")
    
    # Compare PyTorch sample() vs ONNX
    latent_diff_sample = np.abs(pytorch_latents_sample.detach().numpy() - onnx_latents)
    print(f"\nPyTorch sample() vs ONNX MAE: {np.mean(latent_diff_sample):.6f}")
    print(f"PyTorch sample() vs ONNX Max Diff: {np.max(latent_diff_sample):.6f}")
    
    # Compare PyTorch mode() vs ONNX (should be much closer!)
    latent_diff_mode = np.abs(pytorch_latents_mode.detach().numpy() - onnx_latents)
    print(f"PyTorch mode() vs ONNX MAE: {np.mean(latent_diff_mode):.6f}")
    print(f"PyTorch mode() vs ONNX Max Diff: {np.max(latent_diff_mode):.6f}")
    
    if np.mean(latent_diff_mode) < np.mean(latent_diff_sample) / 2:
        print("✅ Using mode() significantly reduces VAE encoding differences!")
        use_mode_latents = pytorch_latents_mode
    else:
        print("⚠️ Mode vs sample difference doesn't explain the issue")
        use_mode_latents = pytorch_latents_sample
    
    # === STEP 2: UNet Processing ===
    print("\n=== STEP 2: UNet Processing ===")
    
    # Create dummy audio features
    dummy_audio = np.random.randn(1, 50, 384).astype(np.float32)
    
    # PyTorch PE
    pytorch_audio = pe(torch.from_numpy(dummy_audio))
    print(f"PyTorch audio shape: {pytorch_audio.shape}")
    
    # ONNX PE
    onnx_audio = pe_session.run(['audio_features_with_pe'], {'audio_features': dummy_audio})[0]
    print(f"ONNX audio shape: {onnx_audio.shape}")
    
    timesteps = torch.tensor([0], dtype=torch.long, device=device)
    timesteps_np = np.array([0], dtype=np.int64)
    
    # PyTorch UNet (use the latents that matched better)
    with torch.no_grad():
        pytorch_unet_out = unet.model(
            sample=use_mode_latents,
            timestep=timesteps,
            encoder_hidden_states=pytorch_audio,
            return_dict=False
        )[0]
    
    print(f"PyTorch UNet out shape: {pytorch_unet_out.shape}")
    print(f"PyTorch UNet out range: [{pytorch_unet_out.min():.6f}, {pytorch_unet_out.max():.6f}]")
    
    # ONNX UNet
    onnx_unet_out = unet_session.run(
        ['noise_prediction'],
        {
            'input_latents': onnx_latents,
            'timesteps': timesteps_np,
            'audio_prompts': onnx_audio
        }
    )[0]
    
    print(f"ONNX UNet out shape: {onnx_unet_out.shape}")
    print(f"ONNX UNet out range: [{onnx_unet_out.min():.6f}, {onnx_unet_out.max():.6f}]")
    
    # Compare UNet outputs - THIS IS THE CRITICAL COMPARISON
    unet_diff = np.abs(pytorch_unet_out.detach().numpy() - onnx_unet_out)
    print(f"\n🔍 CRITICAL: UNet MAE: {np.mean(unet_diff):.6f}")
    print(f"🔍 CRITICAL: UNet Max Diff: {np.max(unet_diff):.6f}")
    
    if np.mean(unet_diff) > 0.01:
        print("❌ UNet outputs differ significantly - this is likely the blurriness source!")
    else:
        print("✅ UNet outputs are very similar")

if __name__ == "__main__":
    debug_pytorch_vs_onnx() 