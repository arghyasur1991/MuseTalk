#!/usr/bin/env python3
"""
Debug script to extract face bbox coordinates for Unity comparison
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder

def debug_face_bbox(image_path):
    """Extract and print face bbox coordinates for debugging"""
    print(f"Processing image: {image_path}")
    
    # Process single image
    img_list = [image_path]
    
    try:
        # Get landmark-based bbox (same as used in inference)
        coords_list, frames = get_landmark_and_bbox(img_list, upperbondrange=0)
        
        print(f"\n=== FACE DETECTION RESULTS ===")
        print(f"Image: {image_path}")
        print(f"Image shape: {frames[0].shape}")
        
        for i, (bbox, frame) in enumerate(zip(coords_list, frames)):
            if bbox == coord_placeholder:
                print(f"Frame {i}: NO FACE DETECTED")
                continue
                
            x1, y1, x2, y2 = bbox
            print(f"Frame {i}: Face bbox = [{x1}, {y1}, {x2}, {y2}]")
            print(f"Frame {i}: Bbox width = {x2 - x1}, height = {y2 - y1}")
            
            # Add v15 extra margin (like in inference)
            y2_with_margin = y2 + 10  # extra_margin for v15
            y2_with_margin = min(y2_with_margin, frame.shape[0])
            print(f"Frame {i}: With v15 margin = [{x1}, {y1}, {x2}, {y2_with_margin}]")
            print(f"Frame {i}: With margin width = {x2 - x1}, height = {y2_with_margin - y1}")
            
            # Show crop region info
            crop_frame = frame[y1:y2_with_margin, x1:x2]
            print(f"Frame {i}: Crop shape before resize = {crop_frame.shape}")
            print(f"Frame {i}: Crop will be resized to (256, 256)")
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    # Process the first avatar image (same as Unity will use)
    avatar_path = "results/v15/avatars/avator_1/full_imgs/00000000.png"
    
    if os.path.exists(avatar_path):
        debug_face_bbox(avatar_path)
    else:
        print(f"Avatar image not found: {avatar_path}") 