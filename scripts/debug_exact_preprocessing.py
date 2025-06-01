#!/usr/bin/env python3
"""
Debug exact PyTorch preprocessing step by step
"""

import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

# Add the project root to Python path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.utils import load_all_model

def debug_exact_pytorch_preprocessing():
    """Debug the exact PyTorch preprocessing step by step"""
    
    print("=== EXACT PyTorch Preprocessing Debug ===")
    
    # Load VAE
    device = "cpu"
    vae, _, _ = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    
    # Load test image and prepare exactly like inference
    img_path = "temp_comparison_imgs/00000000.png"
    test_image = cv2.imread(img_path)
    
    import pickle
    with open("results/temp_comparison_imgs.pkl", 'rb') as f:
        coord_list = pickle.load(f)
    
    bbox = coord_list[0]
    x1, y1, x2, y2 = bbox
    y2 = y2 + 10  # v15 extra margin
    y2 = min(y2, test_image.shape[0])
    
    # Crop face region exactly like inference
    crop_frame = test_image[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    print(f"Input crop_frame: shape={crop_frame.shape}, dtype={crop_frame.dtype}")
    print(f"Input crop_frame range: [{crop_frame.min()}, {crop_frame.max()}]")
    
    # === Step-by-step PyTorch VAE.preprocess_img() ===
    print("\n--- Step-by-step PyTorch VAE.preprocess_img() ---")
    
    # Step 1: The input is NOT a file path, so it goes to the else branch
    # Step 2: img = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
    print(f"After BGR->RGB: shape={img_rgb.shape}, range=[{img_rgb.min()}, {img_rgb.max()}]")
    
    # Step 3: window.append(img)
    window = [img_rgb]
    print(f"Window: length={len(window)}, first shape={window[0].shape}")
    
    # Step 4: x = np.asarray(window) / 255.
    x = np.asarray(window) / 255.0
    print(f"After normalize: shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
    
    # Step 5: x = np.transpose(x, (3, 0, 1, 2))
    x = np.transpose(x, (3, 0, 1, 2))
    print(f"After transpose: shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
    
    # Step 6: x = torch.squeeze(torch.FloatTensor(x))
    x = torch.squeeze(torch.FloatTensor(x))
    print(f"After squeeze: shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
    
    # Step 7a: Apply mask for half_mask=True
    x_masked = x.clone()
    mask_tensor = vae._mask_tensor
    print(f"Mask tensor shape: {mask_tensor.shape}")
    x_masked = x_masked * (mask_tensor > 0.5)
    print(f"After masking: shape={x_masked.shape}, range=[{x_masked.min():.6f}, {x_masked.max():.6f}]")
    
    # Step 8: x = self.transform(x) - this is the normalization step
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    x_full_normalized = transform(x)
    x_masked_normalized = transform(x_masked)
    print(f"After normalize transform (full): range=[{x_full_normalized.min():.6f}, {x_full_normalized.max():.6f}]")
    print(f"After normalize transform (masked): range=[{x_masked_normalized.min():.6f}, {x_masked_normalized.max():.6f}]")
    
    # Step 9: x = x.unsqueeze(0)
    x_full_final = x_full_normalized.unsqueeze(0)
    x_masked_final = x_masked_normalized.unsqueeze(0)
    print(f"Final PyTorch (full): shape={x_full_final.shape}, range=[{x_full_final.min():.6f}, {x_full_final.max():.6f}]")
    print(f"Final PyTorch (masked): shape={x_masked_final.shape}, range=[{x_masked_final.min():.6f}, {x_masked_final.max():.6f}]")
    
    # === Manual numpy replication ===
    print("\n--- Manual numpy replication ---")
    
    # Replicate exactly the same steps
    img_rgb_manual = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
    window_manual = [img_rgb_manual]
    x_manual = np.asarray(window_manual) / 255.0
    x_manual = np.transpose(x_manual, (3, 0, 1, 2))  # [C, B, H, W]
    x_manual = np.squeeze(x_manual)  # [C, H, W]
    
    # Full image
    x_full_manual = x_manual.copy()
    # Apply normalization: (x - 0.5) / 0.5
    for c in range(3):
        x_full_manual[c] = (x_full_manual[c] - 0.5) / 0.5
    x_full_manual = np.expand_dims(x_full_manual, 0)
    
    # Masked image
    x_masked_manual = x_manual.copy()
    # Apply mask first
    mask_np = np.zeros((256, 256), dtype=np.float32)
    mask_np[:256//2, :] = 1
    for c in range(3):
        x_masked_manual[c] = x_masked_manual[c] * mask_np
    # Then apply normalization
    for c in range(3):
        x_masked_manual[c] = (x_masked_manual[c] - 0.5) / 0.5
    x_masked_manual = np.expand_dims(x_masked_manual, 0)
    
    print(f"Manual numpy (full): shape={x_full_manual.shape}, range=[{x_full_manual.min():.6f}, {x_full_manual.max():.6f}]")
    print(f"Manual numpy (masked): shape={x_masked_manual.shape}, range=[{x_masked_manual.min():.6f}, {x_masked_manual.max():.6f}]")
    
    # === Compare ===
    print("\n--- Compare PyTorch vs Manual ---")
    
    diff_full = np.abs(x_full_final.numpy() - x_full_manual)
    diff_masked = np.abs(x_masked_final.numpy() - x_masked_manual)
    
    print(f"Full image diff: MAE={np.mean(diff_full):.8f}, Max={np.max(diff_full):.8f}")
    print(f"Masked image diff: MAE={np.mean(diff_masked):.8f}, Max={np.max(diff_masked):.8f}")
    
    if np.mean(diff_full) < 1e-6:
        print("✅ Full image preprocessing replicated exactly!")
    else:
        print("❌ Full image preprocessing differs")
        
    if np.mean(diff_masked) < 1e-6:
        print("✅ Masked image preprocessing replicated exactly!")
    else:
        print("❌ Masked image preprocessing differs")

if __name__ == "__main__":
    debug_exact_pytorch_preprocessing() 