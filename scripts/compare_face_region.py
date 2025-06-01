#!/usr/bin/env python3
"""
Compare ONLY the face/mouth region from PyTorch vs ONNX inference
Focuses on the actual animated area rather than the full image
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pickle

def load_face_coordinates():
    """Load the face coordinates used during inference"""
    coord_files = [
        "results/temp_comparison_imgs.pkl",  # Generated during PyTorch inference
        "temp_comparison_imgs.pkl"  # Alternative location
    ]
    
    for coord_file in coord_files:
        if os.path.exists(coord_file):
            with open(coord_file, 'rb') as f:
                coord_list = pickle.load(f)
            print(f"Loaded coordinates from {coord_file}")
            return coord_list
    
    print("Warning: Could not find coordinate file, using manual estimation")
    return None

def get_face_region(image, coord_list, frame_idx):
    """Extract the face region that was actually processed"""
    if coord_list and frame_idx < len(coord_list):
        bbox = coord_list[frame_idx % len(coord_list)]
        x1, y1, x2, y2 = bbox
        
        # Add v15 extra margin like in the inference
        y2 = y2 + 10
        y2 = min(y2, image.shape[0])
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        return face_region, (x1, y1, x2, y2)
    else:
        # Fallback: estimate center face region
        h, w = image.shape[:2]
        # Assume face is in center 40% width, upper 60% height
        x1, x2 = int(w * 0.3), int(w * 0.7)
        y1, y2 = int(h * 0.2), int(h * 0.8)
        face_region = image[y1:y2, x1:x2]
        return face_region, (x1, y1, x2, y2)

def get_mouth_region(face_region):
    """Extract just the mouth/lower face area from the face region"""
    h, w = face_region.shape[:2]
    # Mouth is typically in lower 40% of face
    mouth_y1 = int(h * 0.6)
    mouth_region = face_region[mouth_y1:, :]
    return mouth_region

def compare_face_regions(pytorch_dir, onnx_dir, output_dir):
    """Compare only the face regions from PyTorch and ONNX inference"""
    
    pytorch_frames = sorted(list(Path(pytorch_dir).glob("*.png")))
    onnx_frames = sorted(list(Path(onnx_dir).glob("*.png")))
    
    print(f"PyTorch frames: {len(pytorch_frames)}")
    print(f"ONNX frames: {len(onnx_frames)}")
    
    if len(pytorch_frames) != len(onnx_frames):
        print("WARNING: Different number of frames!")
        return
    
    # Load face coordinates
    coord_list = load_face_coordinates()
    
    os.makedirs(output_dir, exist_ok=True)
    
    face_differences = []
    face_similarities = []
    mouth_differences = []
    mouth_similarities = []
    
    for i, (pt_frame, onnx_frame) in enumerate(zip(pytorch_frames, onnx_frames)):
        # Load full images
        pt_img = cv2.imread(str(pt_frame))
        onnx_img = cv2.imread(str(onnx_frame))
        
        if pt_img.shape != onnx_img.shape:
            print(f"Frame {i}: Shape mismatch - PyTorch: {pt_img.shape}, ONNX: {onnx_img.shape}")
            continue
        
        # Extract face regions
        pt_face, bbox = get_face_region(pt_img, coord_list, i)
        onnx_face, _ = get_face_region(onnx_img, coord_list, i)
        
        # Ensure same size
        if pt_face.shape != onnx_face.shape:
            # Resize to match (shouldn't happen but just in case)
            min_h = min(pt_face.shape[0], onnx_face.shape[0])
            min_w = min(pt_face.shape[1], onnx_face.shape[1])
            pt_face = pt_face[:min_h, :min_w]
            onnx_face = onnx_face[:min_h, :min_w]
        
        # Calculate face region metrics
        face_diff = np.abs(pt_face.astype(np.float32) - onnx_face.astype(np.float32))
        face_mae = np.mean(face_diff)
        face_mse = np.mean(face_diff ** 2)
        face_psnr = 20 * np.log10(255.0 / np.sqrt(face_mse)) if face_mse > 0 else float('inf')
        
        pt_face_flat = pt_face.flatten().astype(np.float32)
        onnx_face_flat = onnx_face.flatten().astype(np.float32)
        face_correlation = np.corrcoef(pt_face_flat, onnx_face_flat)[0, 1]
        
        face_differences.append(face_mae)
        face_similarities.append(face_correlation)
        
        # Extract mouth regions
        pt_mouth = get_mouth_region(pt_face)
        onnx_mouth = get_mouth_region(onnx_face)
        
        # Calculate mouth region metrics
        mouth_diff = np.abs(pt_mouth.astype(np.float32) - onnx_mouth.astype(np.float32))
        mouth_mae = np.mean(mouth_diff)
        mouth_mse = np.mean(mouth_diff ** 2)
        mouth_psnr = 20 * np.log10(255.0 / np.sqrt(mouth_mse)) if mouth_mse > 0 else float('inf')
        
        pt_mouth_flat = pt_mouth.flatten().astype(np.float32)
        onnx_mouth_flat = onnx_mouth.flatten().astype(np.float32)
        mouth_correlation = np.corrcoef(pt_mouth_flat, onnx_mouth_flat)[0, 1]
        
        mouth_differences.append(mouth_mae)
        mouth_similarities.append(mouth_correlation)
        
        print(f"Frame {i+1:02d}:")
        print(f"  Face  - MAE={face_mae:.2f}, PSNR={face_psnr:.2f}dB, Corr={face_correlation:.6f}")
        print(f"  Mouth - MAE={mouth_mae:.2f}, PSNR={mouth_psnr:.2f}dB, Corr={mouth_correlation:.6f}")
        
        # Create detailed comparison for first few frames
        if i < 5:
            # Create face comparison
            face_comparison = np.hstack([
                pt_face, 
                onnx_face, 
                np.clip(face_diff * 5, 0, 255).astype(np.uint8)
            ])
            
            # Create mouth comparison
            mouth_comparison = np.hstack([
                pt_mouth, 
                onnx_mouth, 
                np.clip(mouth_diff * 5, 0, 255).astype(np.uint8)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(face_comparison, "PyTorch Face", (5, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(face_comparison, "ONNX Face", (pt_face.shape[1] + 5, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(face_comparison, "Diff x5", (pt_face.shape[1] * 2 + 5, 20), font, 0.5, (0, 255, 0), 1)
            
            cv2.putText(mouth_comparison, "PyTorch Mouth", (5, 15), font, 0.5, (0, 255, 0), 1)
            cv2.putText(mouth_comparison, "ONNX Mouth", (pt_mouth.shape[1] + 5, 15), font, 0.5, (0, 255, 0), 1)
            cv2.putText(mouth_comparison, "Diff x5", (pt_mouth.shape[1] * 2 + 5, 15), font, 0.5, (0, 255, 0), 1)
            
            # Save comparisons
            cv2.imwrite(os.path.join(output_dir, f"face_comparison_{i+1:02d}.png"), face_comparison)
            cv2.imwrite(os.path.join(output_dir, f"mouth_comparison_{i+1:02d}.png"), mouth_comparison)
            
            # Also save the full image with bbox marked
            full_comparison = np.hstack([pt_img, onnx_img])
            x1, y1, x2, y2 = bbox
            cv2.rectangle(full_comparison, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(full_comparison, (pt_img.shape[1] + x1, y1), (pt_img.shape[1] + x2, y2), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"full_with_bbox_{i+1:02d}.png"), full_comparison)
    
    # Print detailed summary
    print(f"\n=== FACE REGION COMPARISON ===")
    print(f"Face Average MAE: {np.mean(face_differences):.3f} pixels")
    print(f"Face Max MAE: {np.max(face_differences):.3f} pixels")
    print(f"Face Average Correlation: {np.mean(face_similarities):.6f}")
    print(f"Face Min Correlation: {np.min(face_similarities):.6f}")
    
    print(f"\n=== MOUTH REGION COMPARISON ===")
    print(f"Mouth Average MAE: {np.mean(mouth_differences):.3f} pixels")
    print(f"Mouth Max MAE: {np.max(mouth_differences):.3f} pixels")
    print(f"Mouth Average Correlation: {np.mean(mouth_similarities):.6f}")
    print(f"Mouth Min Correlation: {np.min(mouth_similarities):.6f}")
    
    # Assessment
    mouth_avg_mae = np.mean(mouth_differences)
    mouth_avg_corr = np.mean(mouth_similarities)
    
    print(f"\n=== MOUTH ANIMATION QUALITY ASSESSMENT ===")
    if mouth_avg_mae < 3.0 and mouth_avg_corr > 0.98:
        print("✅ MOUTH: PyTorch and ONNX mouth animations are VERY SIMILAR")
    elif mouth_avg_mae < 8.0 and mouth_avg_corr > 0.9:
        print("⚠️  MOUTH: PyTorch and ONNX mouth animations are SIMILAR with visible differences")
    elif mouth_avg_mae < 15.0 and mouth_avg_corr > 0.8:
        print("❌ MOUTH: PyTorch and ONNX mouth animations have NOTICEABLE differences")
    else:
        print("❌ MOUTH: PyTorch and ONNX mouth animations have SIGNIFICANT differences")
        
    if mouth_avg_mae > 5.0 or mouth_avg_corr < 0.95:
        print("\n🔍 ANALYSIS: ONNX mouth region appears blurrier/different than PyTorch")
        print("   Possible causes:")
        print("   - VAE decoder differences in ONNX vs PyTorch")
        print("   - Different interpolation/upsampling in ONNX")
        print("   - Quantization effects in ONNX model")
        print("   - Different precision in UNet output processing")
    
    return {
        'face_avg_mae': np.mean(face_differences),
        'face_avg_correlation': np.mean(face_similarities),
        'mouth_avg_mae': mouth_avg_mae,
        'mouth_avg_correlation': mouth_avg_corr
    }

if __name__ == "__main__":
    pytorch_dir = "comparison_frames/pytorch"
    onnx_dir = "comparison_frames/onnx"
    output_dir = "face_region_comparison"
    
    results = compare_face_regions(pytorch_dir, onnx_dir, output_dir)
    print(f"\nFace region comparison images saved to: {output_dir}") 