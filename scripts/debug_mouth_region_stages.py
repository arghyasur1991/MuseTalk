#!/usr/bin/env python3
"""
Mouth Region Stage-by-Stage Debugging: PyTorch vs ONNX
Focus specifically on mouth region quality to identify and fix quality gaps
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

def extract_mouth_region(image, bbox, margin=10):
    """Extract mouth region from image using face bbox"""
    x1, y1, x2, y2 = bbox
    
    # Focus on lower part of face for mouth
    mouth_y1 = y1 + int((y2 - y1) * 0.6)  # Start at 60% down the face
    mouth_y2 = y2 + margin  # Include some margin below
    
    mouth_x1 = x1 + int((x2 - x1) * 0.2)  # Start at 20% from left
    mouth_x2 = x2 - int((x2 - x1) * 0.2)  # End at 80% from left
    
    # Ensure within image bounds
    mouth_y1 = max(0, mouth_y1)
    mouth_y2 = min(image.shape[0], mouth_y2)
    mouth_x1 = max(0, mouth_x1)
    mouth_x2 = min(image.shape[1], mouth_x2)
    
    return image[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

def compare_mouth_regions(pytorch_img, onnx_img, bbox):
    """Compare mouth regions and return detailed metrics"""
    pytorch_mouth = extract_mouth_region(pytorch_img, bbox)
    onnx_mouth = extract_mouth_region(onnx_img, bbox)
    
    # Ensure same size
    if pytorch_mouth.shape != onnx_mouth.shape:
        min_h = min(pytorch_mouth.shape[0], onnx_mouth.shape[0])
        min_w = min(pytorch_mouth.shape[1], onnx_mouth.shape[1])
        pytorch_mouth = pytorch_mouth[:min_h, :min_w]
        onnx_mouth = onnx_mouth[:min_h, :min_w]
    
    # Calculate differences
    diff = np.abs(pytorch_mouth.astype(np.float32) - onnx_mouth.astype(np.float32))
    mae = np.mean(diff)
    max_diff = np.max(diff)
    
    # Calculate correlation per channel
    correlations = []
    for c in range(3):
        pytorch_flat = pytorch_mouth[:, :, c].flatten()
        onnx_flat = onnx_mouth[:, :, c].flatten()
        if len(pytorch_flat) > 1:
            corr = np.corrcoef(pytorch_flat, onnx_flat)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    avg_corr = np.mean(correlations) if correlations else 0
    
    return mae, max_diff, avg_corr, pytorch_mouth, onnx_mouth

def debug_mouth_quality_stages():
    """Debug each stage with focus on mouth region quality"""
    
    print("=== MOUTH REGION STAGE-BY-STAGE DEBUG ===")
    
    # Load PyTorch models
    device = "cpu"
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    
    # Load ONNX models with highest precision settings
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    unet_session = ort.InferenceSession("./models/onnx/unet_v15_hq.onnx", providers=providers, session_options=session_options)
    vae_encoder_session = ort.InferenceSession("./models/onnx/vae_encoder_v15_hq.onnx", providers=providers, session_options=session_options)
    vae_decoder_session = ort.InferenceSession("./models/onnx/vae_decoder_v15_hq.onnx", providers=providers, session_options=session_options)
    pe_session = ort.InferenceSession("./models/onnx/positional_encoding_v15.onnx", providers=providers, session_options=session_options)
    
    # Load test data
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
    
    print(f"Input face crop: {crop_frame.shape}")
    
    # === STAGE 1: VAE Preprocessing ===
    print("\n=== STAGE 1: VAE Preprocessing ===")
    
    # PyTorch preprocessing
    pytorch_full = vae.preprocess_img(crop_frame, half_mask=False)
    pytorch_masked = vae.preprocess_img(crop_frame, half_mask=True)
    
    # ONNX preprocessing (corrected)
    img_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)  # Match PyTorch exactly
    window = [img_rgb]  # Use RGB image like PyTorch
    x = np.asarray(window, dtype=np.float32) / 255.0
    x = np.transpose(x, (3, 0, 1, 2))  # [C, B, H, W]
    x = np.squeeze(x)  # [C, H, W]
    
    # Full image
    x_full = x.copy()
    for c in range(3):
        x_full[c] = (x_full[c] - 0.5) / 0.5
    onnx_full = np.expand_dims(x_full, 0)
    
    # Masked image
    x_masked = x.copy()
    mask_tensor = np.zeros((256, 256), dtype=np.float32)
    mask_tensor[:256//2, :] = 1
    for c in range(3):
        x_masked[c] = x_masked[c] * mask_tensor
    for c in range(3):
        x_masked[c] = (x_masked[c] - 0.5) / 0.5
    onnx_masked = np.expand_dims(x_masked, 0)
    
    prep_diff_full = np.abs(pytorch_full.cpu().numpy() - onnx_full)
    prep_diff_masked = np.abs(pytorch_masked.cpu().numpy() - onnx_masked)
    print(f"Preprocessing diff (full): MAE={np.mean(prep_diff_full):.8f}")
    print(f"Preprocessing diff (masked): MAE={np.mean(prep_diff_masked):.8f}")
    
    # === STAGE 2: VAE Encoding ===
    print("\n=== STAGE 2: VAE Encoding ===")
    
    # PyTorch VAE encoding - TEST BOTH SAMPLE() AND MODE()
    with torch.no_grad():
        # Test with both sample() and mode()
        pytorch_masked_dist = vae.vae.encode(pytorch_masked.to(vae.vae.dtype)).latent_dist
        pytorch_full_dist = vae.vae.encode(pytorch_full.to(vae.vae.dtype)).latent_dist
        
        # Use sample() for exact PyTorch behavior
        pytorch_masked_latents_sample = vae.scaling_factor * pytorch_masked_dist.sample()
        pytorch_full_latents_sample = vae.scaling_factor * pytorch_full_dist.sample()
        
        # Also test mode() for comparison
        pytorch_masked_latents_mode = vae.scaling_factor * pytorch_masked_dist.mode()
        pytorch_full_latents_mode = vae.scaling_factor * pytorch_full_dist.mode()
    
    # ONNX VAE encoding
    onnx_masked_latents = vae_encoder_session.run(['latents'], {'image': onnx_masked})[0]
    onnx_full_latents = vae_encoder_session.run(['latents'], {'image': onnx_full})[0]
    
    # Compare both sample() and mode()
    sample_diff_masked = np.abs(pytorch_masked_latents_sample.numpy() - onnx_masked_latents)
    sample_diff_full = np.abs(pytorch_full_latents_sample.numpy() - onnx_full_latents)
    mode_diff_masked = np.abs(pytorch_masked_latents_mode.numpy() - onnx_masked_latents)
    mode_diff_full = np.abs(pytorch_full_latents_mode.numpy() - onnx_full_latents)
    
    print(f"VAE encoding (sample) diff: masked={np.mean(sample_diff_masked):.6f}, full={np.mean(sample_diff_full):.6f}")
    print(f"VAE encoding (mode) diff: masked={np.mean(mode_diff_masked):.6f}, full={np.mean(mode_diff_full):.6f}")
    
    # Choose the better matching approach
    if np.mean(mode_diff_masked) + np.mean(mode_diff_full) < np.mean(sample_diff_masked) + np.mean(sample_diff_full):
        print("Using PyTorch mode() for better ONNX matching")
        pytorch_masked_latents_final = pytorch_masked_latents_mode
        pytorch_full_latents_final = pytorch_full_latents_mode
        used_method = "mode"
    else:
        print("Using PyTorch sample() for original behavior")
        pytorch_masked_latents_final = pytorch_masked_latents_sample
        pytorch_full_latents_final = pytorch_full_latents_sample
        used_method = "sample"
    
    # === STAGE 3: UNet Processing ===
    print("\n=== STAGE 3: UNet Processing ===")
    
    # Combine latents
    pytorch_combined_latents = torch.cat([pytorch_masked_latents_final, pytorch_full_latents_final], dim=1)
    onnx_combined_latents = np.concatenate([onnx_masked_latents, onnx_full_latents], axis=1)
    
    # Create audio
    np.random.seed(42)  # Fixed seed for reproducibility
    dummy_audio = np.random.randn(1, 50, 384).astype(np.float32)
    
    # PyTorch processing
    pytorch_audio = pe(torch.from_numpy(dummy_audio))
    timesteps_torch = torch.tensor([0], dtype=torch.long, device=device)
    
    with torch.no_grad():
        pytorch_unet_out = unet.model(
            sample=pytorch_combined_latents,
            timestep=timesteps_torch,
            encoder_hidden_states=pytorch_audio,
            return_dict=False
        )[0]
    
    # ONNX processing
    onnx_audio = pe_session.run(['audio_features_with_pe'], {'audio_features': dummy_audio})[0]
    timesteps_onnx = np.array([0], dtype=np.int64)
    
    onnx_unet_out = unet_session.run(
        ['noise_prediction'],
        {
            'input_latents': onnx_combined_latents,
            'timesteps': timesteps_onnx,
            'audio_prompts': onnx_audio
        }
    )[0]
    
    unet_diff = np.abs(pytorch_unet_out.numpy() - onnx_unet_out)
    print(f"UNet processing diff: MAE={np.mean(unet_diff):.6f}")
    
    # === STAGE 4: VAE Decoding ===
    print("\n=== STAGE 4: VAE Decoding ===")
    
    # PyTorch VAE decoding
    pytorch_decoded_latents = (1 / vae.scaling_factor) * pytorch_unet_out
    with torch.no_grad():
        pytorch_decoded_image = vae.vae.decode(pytorch_decoded_latents.to(vae.vae.dtype)).sample
    
    # ONNX VAE decoding
    onnx_decoded_image = vae_decoder_session.run(['image'], {'latents': onnx_unet_out})[0]
    
    decode_diff = np.abs(pytorch_decoded_image.numpy() - onnx_decoded_image)
    print(f"VAE decoding diff: MAE={np.mean(decode_diff):.6f}")
    
    # === STAGE 5: Final Image Conversion & Mouth Region Analysis ===
    print("\n=== STAGE 5: Final Image Conversion & Mouth Analysis ===")
    
    # PyTorch final image
    pytorch_final = (pytorch_decoded_image / 2 + 0.5).clamp(0, 1)
    pytorch_final = pytorch_final.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    pytorch_final = (pytorch_final * 255).round().astype("uint8")[0]
    pytorch_final_bgr = pytorch_final[...,::-1]  # RGB to BGR
    
    # ONNX final image (using highest precision conversion)
    onnx_image_transposed = np.transpose(onnx_decoded_image, (0, 2, 3, 1))
    onnx_image_float64 = onnx_image_transposed.astype(np.float64)
    onnx_image_normalized = (onnx_image_float64 / 2.0 + 0.5).clip(0.0, 1.0)
    onnx_final = np.round(onnx_image_normalized * 255.0).astype(np.uint8)[0]
    # Keep as RGB, then convert to BGR for fair comparison
    onnx_final_bgr = onnx_final[...,::-1]  # RGB to BGR
    
    # Resize both to original face crop size for mouth analysis
    pytorch_resized = cv2.resize(pytorch_final_bgr, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    onnx_resized = cv2.resize(onnx_final_bgr, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    # Create bbox for 256x256 crop (mouth region in lower half)
    crop_bbox = [0, 0, 256, 256]
    
    # Compare mouth regions
    mouth_mae, mouth_max_diff, mouth_corr, pytorch_mouth, onnx_mouth = compare_mouth_regions(
        pytorch_resized, onnx_resized, crop_bbox
    )
    
    print(f"Mouth region analysis:")
    print(f"  MAE: {mouth_mae:.3f} pixels")
    print(f"  Max difference: {mouth_max_diff:.3f} pixels")
    print(f"  Correlation: {mouth_corr:.6f}")
    
    # Save mouth region samples for visual inspection
    os.makedirs("mouth_debug", exist_ok=True)
    cv2.imwrite("mouth_debug/pytorch_mouth.png", pytorch_mouth)
    cv2.imwrite("mouth_debug/onnx_mouth.png", onnx_mouth)
    cv2.imwrite("mouth_debug/pytorch_full.png", pytorch_resized)
    cv2.imwrite("mouth_debug/onnx_full.png", onnx_resized)
    
    # Calculate mouth difference image
    mouth_diff_img = np.abs(pytorch_mouth.astype(np.float32) - onnx_mouth.astype(np.float32))
    mouth_diff_img = (mouth_diff_img / mouth_diff_img.max() * 255).astype(np.uint8)
    cv2.imwrite("mouth_debug/mouth_difference.png", mouth_diff_img)
    
    print(f"Mouth debug images saved to: mouth_debug/")
    
    # === SUMMARY & RECOMMENDATIONS ===
    print("\n=== MOUTH QUALITY ANALYSIS SUMMARY ===")
    
    stage_impacts = [
        ("Preprocessing", max(np.mean(prep_diff_full), np.mean(prep_diff_masked))),
        ("VAE Encoding", max(np.mean(mode_diff_masked), np.mean(mode_diff_full)) if used_method == "mode" else max(np.mean(sample_diff_masked), np.mean(sample_diff_full))),
        ("UNet Processing", np.mean(unet_diff)),
        ("VAE Decoding", np.mean(decode_diff)),
        ("Final Mouth MAE", mouth_mae)
    ]
    
    print(f"Used VAE method: {used_method}")
    print("Stage impact on mouth quality:")
    for stage, impact in stage_impacts:
        print(f"  {stage}: {impact:.6f}")
    
    # Find the most problematic stage
    max_impact_stage = max(stage_impacts[:-1], key=lambda x: x[1])
    print(f"\nMost problematic stage: {max_impact_stage[0]} (impact: {max_impact_stage[1]:.6f})")
    
    if mouth_mae > 1.0:
        print(f"\n❌ MOUTH QUALITY ISSUE: MAE={mouth_mae:.3f} > 1.0 pixels")
        print("Recommendations:")
        if max_impact_stage[1] > 0.001:
            print(f"1. Focus on fixing {max_impact_stage[0]} stage")
        if used_method == "sample":
            print("2. Try using deterministic VAE (mode() instead of sample())")
        print("3. Use higher precision in ONNX models")
        print("4. Check for numerical precision issues in conversion")
    else:
        print(f"\n✅ ACCEPTABLE MOUTH QUALITY: MAE={mouth_mae:.3f} pixels")

if __name__ == "__main__":
    debug_mouth_quality_stages() 