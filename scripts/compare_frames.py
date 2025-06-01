#!/usr/bin/env python3
"""
Compare frames from PyTorch vs ONNX inference
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def compare_frames(pytorch_dir, onnx_dir, output_dir):
    """Compare frames from PyTorch and ONNX inference"""
    
    pytorch_frames = sorted(list(Path(pytorch_dir).glob("*.png")))
    onnx_frames = sorted(list(Path(onnx_dir).glob("*.png")))
    
    print(f"PyTorch frames: {len(pytorch_frames)}")
    print(f"ONNX frames: {len(onnx_frames)}")
    
    if len(pytorch_frames) != len(onnx_frames):
        print("WARNING: Different number of frames!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    differences = []
    similarities = []
    
    for i, (pt_frame, onnx_frame) in enumerate(zip(pytorch_frames, onnx_frames)):
        # Load images
        pt_img = cv2.imread(str(pt_frame))
        onnx_img = cv2.imread(str(onnx_frame))
        
        if pt_img.shape != onnx_img.shape:
            print(f"Frame {i}: Shape mismatch - PyTorch: {pt_img.shape}, ONNX: {onnx_img.shape}")
            continue
        
        # Calculate pixel-wise difference
        diff = np.abs(pt_img.astype(np.float32) - onnx_img.astype(np.float32))
        
        # Calculate metrics
        mae = np.mean(diff)  # Mean Absolute Error
        mse = np.mean(diff ** 2)  # Mean Squared Error
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Structural similarity (simple correlation)
        pt_flat = pt_img.flatten().astype(np.float32)
        onnx_flat = onnx_img.flatten().astype(np.float32)
        
        correlation = np.corrcoef(pt_flat, onnx_flat)[0, 1]
        
        differences.append(mae)
        similarities.append(correlation)
        
        print(f"Frame {i+1:02d}: MAE={mae:.2f}, MSE={mse:.2f}, PSNR={psnr:.2f}dB, Correlation={correlation:.6f}")
        
        # Create comparison image for first few frames
        if i < 5:
            # Create side-by-side comparison
            comparison = np.hstack([pt_img, onnx_img, np.clip(diff * 10, 0, 255).astype(np.uint8)])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "PyTorch", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "ONNX", (pt_img.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Diff x10", (pt_img.shape[1] * 2 + 10, 30), font, 1, (0, 255, 0), 2)
            
            cv2.imwrite(os.path.join(output_dir, f"comparison_frame_{i+1:02d}.png"), comparison)
    
    # Print summary statistics
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Average MAE: {np.mean(differences):.3f} pixels")
    print(f"Max MAE: {np.max(differences):.3f} pixels")
    print(f"Min MAE: {np.min(differences):.3f} pixels")
    print(f"Average Correlation: {np.mean(similarities):.6f}")
    print(f"Min Correlation: {np.min(similarities):.6f}")
    
    # Determine if they are essentially identical
    avg_mae = np.mean(differences)
    avg_corr = np.mean(similarities)
    
    if avg_mae < 1.0 and avg_corr > 0.999:
        print("✅ RESULT: PyTorch and ONNX outputs are VIRTUALLY IDENTICAL")
    elif avg_mae < 5.0 and avg_corr > 0.99:
        print("✅ RESULT: PyTorch and ONNX outputs are VERY SIMILAR")
    elif avg_mae < 15.0 and avg_corr > 0.95:
        print("⚠️  RESULT: PyTorch and ONNX outputs are SIMILAR with minor differences")
    else:
        print("❌ RESULT: PyTorch and ONNX outputs have SIGNIFICANT DIFFERENCES")
    
    return {
        'avg_mae': avg_mae,
        'max_mae': np.max(differences),
        'avg_correlation': avg_corr,
        'min_correlation': np.min(similarities)
    }

if __name__ == "__main__":
    pytorch_dir = "comparison_frames/pytorch"
    onnx_dir = "comparison_frames/onnx"
    output_dir = "comparison_results"
    
    results = compare_frames(pytorch_dir, onnx_dir, output_dir)
    print(f"\nComparison images saved to: {output_dir}") 