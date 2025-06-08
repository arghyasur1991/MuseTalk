#!/usr/bin/env python3
"""
Test script to examine 1k3d68 model output
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.preprocessing import scrfd_detector, landmark_model

def test_landmark_model():
    # Load a test image - try multiple possible paths
    test_paths = [
        "results/v15/avatars/avator_1/full_imgs/00000000.png",
        "assets/demo/man/man.png",
        "assets/demo/yongen/yongen.png", 
        "assets/demo/sun1/sun1.png"
    ]
    
    img = None
    img_path = None
    for path in test_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img_path = path
                break
    
    if img is None:
        print("No test image found. Tried:")
        for path in test_paths:
            print(f"  {path} - exists: {os.path.exists(path)}")
        return
    
    print(f"Loaded image: {img_path}")
    print(f"Image shape: {img.shape}")
    
    # Detect face
    det_results, kps = scrfd_detector.detect(img, max_num=1)
    
    if len(det_results) == 0:
        print("No face detected")
        return
        
    bbox = det_results[0]
    print(f"Face bbox: {bbox}")
    
    # Test our landmark model
    landmarks = landmark_model.get_landmarks(img, bbox)
    print(f"Our landmarks shape: {landmarks.shape}")
    print(f"First 10 landmarks:\n{landmarks[:10]}")
    
    # Examine the raw model behavior
    session = landmark_model.session
    
    # Prepare a simple face crop
    x1, y1, x2, y2 = bbox[:4].astype(int)
    w, h = x2 - x1, y2 - y1
    
    # Extract and resize face region
    face_crop = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (192, 192))
    
    # Save for inspection
    cv2.imwrite("test_face_crop.jpg", face_resized)
    
    # Convert to model input format
    blob = cv2.dnn.blobFromImage(face_resized, 1.0/128.0, (192, 192), 
                                (127.5, 127.5, 127.5), swapRB=True)
    
    # Run raw inference - use actual input/output names
    input_name = landmark_model.input_name
    output_names = landmark_model.output_names
    print(f"Model input name: {input_name}")
    print(f"Model output names: {output_names}")
    
    pred = session.run(output_names, {input_name: blob})[0][0]
    print(f"Raw model output shape: {pred.shape}")
    print(f"Output min/max: {pred.min():.3f}, {pred.max():.3f}")
    
    # Process output
    if pred.shape[0] >= 3000:
        pred = pred.reshape((-1, 3))
        print("Using 3D landmark format")
    else:
        pred = pred.reshape((-1, 2))
        print("Using 2D landmark format")
        
    # Keep only 68 landmarks
    if pred.shape[0] > 68:
        pred = pred[-68:]
        
    print(f"Landmarks after processing: {pred.shape}")
    print(f"Sample landmarks before coordinate fix:\n{pred[:5]}")
    
    # Apply official coordinate processing
    pred[:, 0:2] += 1
    pred[:, 0:2] *= (192 // 2)  # Scale to image size
    if pred.shape[1] == 3:
        pred[:, 2] *= (192 // 2)
        
    print(f"Sample landmarks after coordinate fix:\n{pred[:5]}")
    
    # Draw landmarks on the face crop
    debug_face = face_resized.copy()
    for i, pt in enumerate(pred):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < 192 and 0 <= y < 192:
            color = (0, 255, 0) if i < 68 else (255, 0, 0)
            cv2.circle(debug_face, (x, y), 2, color, -1)
            # Label key landmarks
            if i in [8, 27, 30, 33, 36, 39, 42, 45, 48, 54, 57]:
                cv2.putText(debug_face, str(i), (x+3, y-3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
    cv2.imwrite("test_landmarks_on_crop.jpg", debug_face)
    
    print("Test completed.")
    print("Generated files:")
    print("  test_face_crop.jpg - Original cropped face")
    print("  test_landmarks_on_crop.jpg - Face with landmarks")

if __name__ == "__main__":
    test_landmark_model() 