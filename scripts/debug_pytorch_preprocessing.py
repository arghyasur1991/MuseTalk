#!/usr/bin/env python3
"""
Stage-by-Stage Debugging: PyTorch vs ONNX Pipeline
Find exactly where differences start and fix that stage
"""

import os
import sys
import cv2
import torch
import numpy as np
import onnxruntime as ort
import pickle

# Add the project root to Python path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.utils import load_all_model

def debug_stage_by_stage():
    """Debug each stage of the pipeline to find where differences start"""
    
    print("=== STAGE-BY-STAGE DEBUG: PyTorch vs ONNX ===")
    
    # Load PyTorch models
    device = "cpu"
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    
    # Load ONNX models
    providers = ['CPUExecutionProvider']
    unet_session = ort.InferenceSession("./models/onnx/unet_v15_hq.onnx", providers=providers)
    vae_encoder_session = ort.InferenceSession("./models/onnx/vae_encoder_v15_hq.onnx", providers=providers)
    vae_decoder_session = ort.InferenceSession("./models/onnx/vae_decoder_v15_hq.onnx", providers=providers)
    pe_session = ort.InferenceSession("./models/onnx/positional_encoding_v15.onnx", providers=providers)
    
    # Load test image and coordinates
    img_path = "temp_comparison_imgs/00000000.png"
    test_image = cv2.imread(img_path)
    
    with open("results/temp_comparison_imgs.pkl", 'rb') as f:
        coord_list = pickle.load(f)
    
    bbox = coord_list[0]
    x1, y1, x2, y2 = bbox
    y2 = y2 + 10  # v15 extra margin
    y2 = min(y2, test_image.shape[0])
    
    # Crop face region exactly like inference
    crop_frame = test_image[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    print(f"Cropped face shape: {crop_frame.shape}")
    
    # === STAGE 1: VAE Preprocessing ===
    print("\n=== STAGE 1: VAE Preprocessing ===")
    
    # PyTorch preprocessing (full image)
    pytorch_full = vae.preprocess_img(crop_frame, half_mask=False)
    print(f"PyTorch full preprocessing: shape={pytorch_full.shape}, range=[{pytorch_full.min():.6f}, {pytorch_full.max():.6f}]")
    
    # PyTorch preprocessing (masked image)  
    pytorch_masked = vae.preprocess_img(crop_frame, half_mask=True)
    print(f"PyTorch masked preprocessing: shape={pytorch_masked.shape}, range=[{pytorch_masked.min():.6f}, {pytorch_masked.max():.6f}]")
    
    # ONNX preprocessing (full image) - FIXED: exactly match PyTorch order
    # PyTorch does: cv2.imread (BGR) -> cv2.cvtColor(BGR2RGB) -> /255 -> transpose -> normalize
    # But crop_frame is already BGR, so we need BGR->RGB conversion
    img_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)  # CRITICAL: convert BGR to RGB first
    window = [img_rgb]
    x = np.asarray(window, dtype=np.float32) / 255.0  # Ensure float32
    x = np.transpose(x, (3, 0, 1, 2))  # [C, B, H, W]
    x = np.squeeze(x)  # [C, H, W]
    # Apply normalization transform
    for c in range(3):
        x[c] = (x[c] - 0.5) / 0.5
    onnx_full = np.expand_dims(x, 0)
    
    # ONNX preprocessing (masked image) - FIXED: exactly match PyTorch order
    img_rgb_masked = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
    window_masked = [img_rgb_masked]
    x_masked = np.asarray(window_masked, dtype=np.float32) / 255.0  # Ensure float32
    x_masked = np.transpose(x_masked, (3, 0, 1, 2))  # [C, B, H, W]
    x_masked = np.squeeze(x_masked)  # [C, H, W]
    # Apply mask first
    mask_tensor = np.zeros((256, 256), dtype=np.float32)
    mask_tensor[:256//2, :] = 1  # Upper half = 1
    for c in range(3):
        x_masked[c] = x_masked[c] * mask_tensor
    # Then apply normalization transform
    for c in range(3):
        x_masked[c] = (x_masked[c] - 0.5) / 0.5
    onnx_masked = np.expand_dims(x_masked, 0)
    
    print(f"ONNX full preprocessing: shape={onnx_full.shape}, range=[{onnx_full.min():.6f}, {onnx_full.max():.6f}]")
    print(f"ONNX masked preprocessing: shape={onnx_masked.shape}, range=[{onnx_masked.min():.6f}, {onnx_masked.max():.6f}]")
    
    # Compare preprocessing
    prep_diff_full = np.abs(pytorch_full.cpu().numpy() - onnx_full)
    prep_diff_masked = np.abs(pytorch_masked.cpu().numpy() - onnx_masked)
    print(f"Preprocessing diff (full): MAE={np.mean(prep_diff_full):.8f}")
    print(f"Preprocessing diff (masked): MAE={np.mean(prep_diff_masked):.8f}")
    
    if np.mean(prep_diff_full) > 1e-6 or np.mean(prep_diff_masked) > 1e-6:
        print("❌ PREPROCESSING DIFFERS - Fix this first!")
        return
    else:
        print("✅ Preprocessing is identical")
    
    # === STAGE 2: VAE Encoding ===
    print("\n=== STAGE 2: VAE Encoding ===")
    
    # PyTorch VAE encoding
    with torch.no_grad():
        pytorch_masked_dist = vae.vae.encode(pytorch_masked.to(vae.vae.dtype)).latent_dist
        pytorch_full_dist = vae.vae.encode(pytorch_full.to(vae.vae.dtype)).latent_dist
        
        pytorch_masked_latents = vae.scaling_factor * pytorch_masked_dist.mode()  # Use mode for comparison
        pytorch_full_latents = vae.scaling_factor * pytorch_full_dist.mode()
        
    print(f"PyTorch masked latents: shape={pytorch_masked_latents.shape}, range=[{pytorch_masked_latents.min():.6f}, {pytorch_masked_latents.max():.6f}]")
    print(f"PyTorch full latents: shape={pytorch_full_latents.shape}, range=[{pytorch_full_latents.min():.6f}, {pytorch_full_latents.max():.6f}]")
    
    # ONNX VAE encoding
    onnx_masked_latents = vae_encoder_session.run(['latents'], {'image': onnx_masked})[0]
    onnx_full_latents = vae_encoder_session.run(['latents'], {'image': onnx_full})[0]
    
    print(f"ONNX masked latents: shape={onnx_masked_latents.shape}, range=[{onnx_masked_latents.min():.6f}, {onnx_masked_latents.max():.6f}]")
    print(f"ONNX full latents: shape={onnx_full_latents.shape}, range=[{onnx_full_latents.min():.6f}, {onnx_full_latents.max():.6f}]")
    
    # Compare VAE encoding
    enc_diff_masked = np.abs(pytorch_masked_latents.numpy() - onnx_masked_latents)
    enc_diff_full = np.abs(pytorch_full_latents.numpy() - onnx_full_latents)
    print(f"VAE encoding diff (masked): MAE={np.mean(enc_diff_masked):.6f}")
    print(f"VAE encoding diff (full): MAE={np.mean(enc_diff_full):.6f}")
    
    if np.mean(enc_diff_masked) > 0.01 or np.mean(enc_diff_full) > 0.01:
        print("❌ VAE ENCODING DIFFERS - This might be the issue!")
        
        # Test if using sample() vs mode() explains the difference
        with torch.no_grad():
            pytorch_masked_sample = vae.scaling_factor * pytorch_masked_dist.sample()
            pytorch_full_sample = vae.scaling_factor * pytorch_full_dist.sample()
            
        sample_diff_masked = np.abs(pytorch_masked_sample.numpy() - onnx_masked_latents)
        sample_diff_full = np.abs(pytorch_full_sample.numpy() - onnx_full_latents)
        print(f"PyTorch sample() vs ONNX diff (masked): MAE={np.mean(sample_diff_masked):.6f}")
        print(f"PyTorch sample() vs ONNX diff (full): MAE={np.mean(sample_diff_full):.6f}")
        
        # Use the better matching latents for subsequent stages
        if np.mean(enc_diff_masked) < np.mean(sample_diff_masked):
            pytorch_masked_latents_final = pytorch_masked_latents
            pytorch_full_latents_final = pytorch_full_latents
            print("Using PyTorch mode() latents for comparison")
        else:
            pytorch_masked_latents_final = pytorch_masked_sample
            pytorch_full_latents_final = pytorch_full_sample
            print("Using PyTorch sample() latents for comparison")
    else:
        print("✅ VAE encoding is similar")
        pytorch_masked_latents_final = pytorch_masked_latents
        pytorch_full_latents_final = pytorch_full_latents
    
    # Concatenate latents for UNet
    pytorch_combined_latents = torch.cat([pytorch_masked_latents_final, pytorch_full_latents_final], dim=1)
    onnx_combined_latents = np.concatenate([onnx_masked_latents, onnx_full_latents], axis=1)
    
    print(f"Combined latents shape - PyTorch: {pytorch_combined_latents.shape}, ONNX: {onnx_combined_latents.shape}")
    
    combined_diff = np.abs(pytorch_combined_latents.numpy() - onnx_combined_latents)
    print(f"Combined latents diff: MAE={np.mean(combined_diff):.6f}")
    
    # === STAGE 3: Audio Processing ===
    print("\n=== STAGE 3: Audio Processing ===")
    
    # Create dummy audio (should be identical)
    dummy_audio = np.random.randn(1, 50, 384).astype(np.float32)
    np.random.seed(42)  # Reset seed to ensure same audio
    dummy_audio = np.random.randn(1, 50, 384).astype(np.float32)
    
    # PyTorch audio processing
    pytorch_audio = pe(torch.from_numpy(dummy_audio))
    print(f"PyTorch audio: shape={pytorch_audio.shape}, range=[{pytorch_audio.min():.6f}, {pytorch_audio.max():.6f}]")
    
    # ONNX audio processing
    onnx_audio = pe_session.run(['audio_features_with_pe'], {'audio_features': dummy_audio})[0]
    print(f"ONNX audio: shape={onnx_audio.shape}, range=[{onnx_audio.min():.6f}, {onnx_audio.max():.6f}]")
    
    audio_diff = np.abs(pytorch_audio.numpy() - onnx_audio)
    print(f"Audio processing diff: MAE={np.mean(audio_diff):.6f}")
    
    if np.mean(audio_diff) > 0.01:
        print("❌ AUDIO PROCESSING DIFFERS")
    else:
        print("✅ Audio processing is similar")
    
    # === STAGE 4: UNet Processing ===
    print("\n=== STAGE 4: UNet Processing ===")
    
    timesteps_torch = torch.tensor([0], dtype=torch.long, device=device)
    timesteps_onnx = np.array([0], dtype=np.int64)
    
    # PyTorch UNet
    with torch.no_grad():
        pytorch_unet_out = unet.model(
            sample=pytorch_combined_latents,
            timestep=timesteps_torch,
            encoder_hidden_states=pytorch_audio,
            return_dict=False
        )[0]
    
    print(f"PyTorch UNet out: shape={pytorch_unet_out.shape}, range=[{pytorch_unet_out.min():.6f}, {pytorch_unet_out.max():.6f}]")
    
    # ONNX UNet
    onnx_unet_out = unet_session.run(
        ['noise_prediction'],
        {
            'input_latents': onnx_combined_latents,
            'timesteps': timesteps_onnx,
            'audio_prompts': onnx_audio
        }
    )[0]
    
    print(f"ONNX UNet out: shape={onnx_unet_out.shape}, range=[{onnx_unet_out.min():.6f}, {onnx_unet_out.max():.6f}]")
    
    unet_diff = np.abs(pytorch_unet_out.numpy() - onnx_unet_out)
    print(f"UNet processing diff: MAE={np.mean(unet_diff):.6f}")
    
    if np.mean(unet_diff) > 0.01:
        print("❌ UNET PROCESSING DIFFERS SIGNIFICANTLY")
    else:
        print("✅ UNet processing is similar")
    
    # === STAGE 5: VAE Decoding ===
    print("\n=== STAGE 5: VAE Decoding ===")
    
    # PyTorch VAE decoding
    pytorch_decoded_latents = (1 / vae.scaling_factor) * pytorch_unet_out
    with torch.no_grad():
        pytorch_decoded_image = vae.vae.decode(pytorch_decoded_latents.to(vae.vae.dtype)).sample
    
    print(f"PyTorch decoded: shape={pytorch_decoded_image.shape}, range=[{pytorch_decoded_image.min():.6f}, {pytorch_decoded_image.max():.6f}]")
    
    # ONNX VAE decoding
    onnx_decoded_image = vae_decoder_session.run(['image'], {'latents': onnx_unet_out})[0]
    
    print(f"ONNX decoded: shape={onnx_decoded_image.shape}, range=[{onnx_decoded_image.min():.6f}, {onnx_decoded_image.max():.6f}]")
    
    decode_diff = np.abs(pytorch_decoded_image.numpy() - onnx_decoded_image)
    print(f"VAE decoding diff: MAE={np.mean(decode_diff):.6f}")
    
    if np.mean(decode_diff) > 0.01:
        print("❌ VAE DECODING DIFFERS")
    else:
        print("✅ VAE decoding is similar")
    
    # === STAGE 6: Final Image Conversion ===
    print("\n=== STAGE 6: Final Image Conversion ===")
    
    # PyTorch final image
    pytorch_final = (pytorch_decoded_image / 2 + 0.5).clamp(0, 1)
    pytorch_final = pytorch_final.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    pytorch_final = (pytorch_final * 255).round().astype("uint8")[0]
    pytorch_final_bgr = pytorch_final[...,::-1]  # RGB to BGR
    
    # ONNX final image (using corrected high-precision conversion)
    onnx_image_transposed = np.transpose(onnx_decoded_image, (0, 2, 3, 1))
    onnx_image_float64 = onnx_image_transposed.astype(np.float64)
    onnx_image_normalized = (onnx_image_float64 / 2.0 + 0.5).clip(0.0, 1.0)
    onnx_final = np.round(onnx_image_normalized * 255.0).astype(np.uint8)[0]
    # Keep as RGB for fair comparison
    
    print(f"PyTorch final image: shape={pytorch_final_bgr.shape}, range=[{pytorch_final_bgr.min()}, {pytorch_final_bgr.max()}]")
    print(f"ONNX final image: shape={onnx_final.shape}, range=[{onnx_final.min()}, {onnx_final.max()}]")
    
    # Convert PyTorch BGR back to RGB for comparison
    pytorch_final_rgb = pytorch_final_bgr[...,::-1] 
    
    final_diff = np.abs(pytorch_final_rgb.astype(np.float32) - onnx_final.astype(np.float32))
    print(f"Final image diff: MAE={np.mean(final_diff):.6f}")
    
    # === SUMMARY ===
    print("\n=== SUMMARY: WHERE DO DIFFERENCES START? ===")
    
    stages = [
        ("Preprocessing", max(np.mean(prep_diff_full), np.mean(prep_diff_masked))),
        ("VAE Encoding", max(np.mean(enc_diff_masked), np.mean(enc_diff_full))),
        ("Audio Processing", np.mean(audio_diff)),
        ("UNet Processing", np.mean(unet_diff)),
        ("VAE Decoding", np.mean(decode_diff)),
        ("Final Conversion", np.mean(final_diff))
    ]
    
    print("Stage-by-stage differences:")
    for stage, diff in stages:
        status = "❌ SIGNIFICANT" if diff > 0.01 else "✅ OK"
        print(f"  {stage}: MAE={diff:.6f} {status}")
    
    # Find the first stage with significant differences
    first_problem = None
    for stage, diff in stages:
        if diff > 0.01:
            first_problem = stage
            break
    
    if first_problem:
        print(f"\n🔍 PROBLEM STARTS AT: {first_problem}")
        print("Focus on fixing this stage to resolve blurriness!")
    else:
        print("\n✅ All stages are similar - blurriness might be elsewhere")

if __name__ == "__main__":
    debug_stage_by_stage() 