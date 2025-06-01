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
    
    # PyTorch VAE encoding
    pytorch_latents = vae.get_latents_for_unet(crop_frame)
    print(f"PyTorch latents shape: {pytorch_latents.shape}")
    print(f"PyTorch latents range: [{pytorch_latents.min():.6f}, {pytorch_latents.max():.6f}]")
    
    # ONNX VAE encoding (matching the ONNX inference exactly)
    # Half-masked image
    image_normalized = crop_frame.astype(np.float32) / 255.0
    mask = np.ones((256, 256), dtype=np.float32)
    mask[256//2:, :] = 0  # Set lower half to 0
    for c in range(3):
        image_normalized[:, :, c] *= mask
    image_tensor_masked = 2.0 * image_normalized - 1.0
    image_tensor_masked = np.expand_dims(np.transpose(image_tensor_masked, (2, 0, 1)), 0)
    
    # Full image
    image_normalized_full = crop_frame.astype(np.float32) / 255.0
    image_tensor_full = 2.0 * image_normalized_full - 1.0
    image_tensor_full = np.expand_dims(np.transpose(image_tensor_full, (2, 0, 1)), 0)
    
    # ONNX VAE encoding
    masked_latents = vae_encoder_session.run(['latents'], {'image': image_tensor_masked})[0]
    ref_latents = vae_encoder_session.run(['latents'], {'image': image_tensor_full})[0]
    onnx_latents = np.concatenate([masked_latents, ref_latents], axis=1)
    
    print(f"ONNX latents shape: {onnx_latents.shape}")
    print(f"ONNX latents range: [{onnx_latents.min():.6f}, {onnx_latents.max():.6f}]")
    
    # Compare latents
    latent_diff = np.abs(pytorch_latents.detach().numpy() - onnx_latents)
    print(f"Latent MAE: {np.mean(latent_diff):.6f}")
    print(f"Latent Max Diff: {np.max(latent_diff):.6f}")
    
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
    
    # PyTorch UNet
    with torch.no_grad():
        pytorch_unet_out = unet.model(
            sample=pytorch_latents,
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