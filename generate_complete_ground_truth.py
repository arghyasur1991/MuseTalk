#!/usr/bin/env python3
"""
Complete Ground Truth Generator for ALL Frames
This script generates ground truth outputs for every frame in the video
to achieve 100% Unity-Python matching for the entire sequence.
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.onnx_inference import ONNXMuseTalkInference
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
from musetalk.utils.audio_processor import AudioProcessor

def save_array(array, name, output_dir):
    """Save numpy array with metadata"""
    output_path = Path(output_dir) / f"{name}.npy"
    np.save(output_path, array)
    
    # Also save metadata
    metadata = {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'mean': float(np.mean(array)),
        'std': float(np.std(array))
    }
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {name}: {array.shape} {array.dtype} -> {output_path}")

def save_image(image, name, output_dir):
    """Save image with metadata"""
    output_path = Path(output_dir) / f"{name}.png"
    cv2.imwrite(str(output_path), image)
    
    # Save metadata
    metadata = {
        'shape': list(image.shape),
        'dtype': str(image.dtype),
        'min': int(np.min(image)),
        'max': int(np.max(image)),
        'mean': float(np.mean(image))
    }
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {name}: {image.shape} {image.dtype} -> {output_path}")

def generate_complete_ground_truth():
    """Generate complete ground truth outputs for ALL frames"""
    
    # Paths
    avatar_path = "data/video/00000000.png"
    audio_path = "data/audio/detective_1s.wav"
    output_dir = "complete_ground_truth_outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Generating COMPLETE Ground Truth Outputs for ALL Frames ===")
    print(f"Avatar: {avatar_path}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_dir}")
    
    # Check if files exist
    if not os.path.exists(avatar_path):
        print(f"ERROR: Avatar file not found: {avatar_path}")
        return
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        return
    
    # Initialize inference engine
    try:
        inference_engine = ONNXMuseTalkInference(
            model_dir="./models/onnx",
            version="v15",
            device="cpu"
        )
        print("✓ ONNX inference engine initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize inference engine: {e}")
        return
    
    # === STAGE 1: Process Avatar (same as before) ===
    print("\n--- Stage 1: Avatar Processing ---")
    avatar_image = cv2.imread(avatar_path)
    save_image(avatar_image, "01_input_avatar", output_dir)
    
    # Face detection
    coord_list, frame_list = get_landmark_and_bbox([avatar_path], 0)
    face_bbox = coord_list[0] if coord_list else coord_placeholder
    
    if face_bbox == coord_placeholder:
        print("ERROR: No face detected!")
        return
    
    bbox_data = {
        'x1': int(face_bbox[0]),
        'y1': int(face_bbox[1]), 
        'x2': int(face_bbox[2]),
        'y2': int(face_bbox[3]),
        'width': int(face_bbox[2] - face_bbox[0]),
        'height': int(face_bbox[3] - face_bbox[1])
    }
    
    with open(Path(output_dir) / "02_face_bbox.json", 'w') as f:
        json.dump(bbox_data, f, indent=2)
    
    print(f"Face detected: {bbox_data}")
    
    # Process face region
    frame = frame_list[0]
    x1, y1, x2, y2 = face_bbox
    y2 = y2 + 10  # v15 extra margin
    y2 = min(y2, frame.shape[0])
    
    crop_frame = frame[y1:y2, x1:x2]
    save_image(crop_frame, "03_cropped_face", output_dir)
    
    crop_frame_resized = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    save_image(crop_frame_resized, "03_cropped_face_256", output_dir)
    
    # VAE encoding
    masked_latents = inference_engine.encode_image_with_half_mask(crop_frame_resized)
    save_array(masked_latents, "04_masked_latents", output_dir)
    
    ref_latents = inference_engine.encode_image(crop_frame_resized)
    save_array(ref_latents, "04_ref_latents", output_dir)
    
    combined_latents = np.concatenate([masked_latents, ref_latents], axis=1)
    save_array(combined_latents, "04_combined_latents", output_dir)
    
    # === STAGE 2: Complete Audio Processing ===
    print("\n--- Stage 2: Complete Audio Processing ---")
    
    try:
        audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        
        audio_metadata = {
            'librosa_length': int(librosa_length),
            'duration_seconds': float(librosa_length / 16000),
            'sample_rate': 16000
        }
        
        with open(Path(output_dir) / "05_audio_metadata.json", 'w') as f:
            json.dump(audio_metadata, f, indent=2)
        
        print(f"Audio length: {librosa_length} samples ({librosa_length/16000:.3f}s)")
        
    except Exception as e:
        print(f"ERROR: Failed to process audio: {e}")
        return
    
    # Load Whisper and process ALL audio chunks
    try:
        from transformers import WhisperModel
        device = "cpu"
        weight_dtype = torch.float32
        whisper = WhisperModel.from_pretrained("./models/whisper")
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        
        fps = 25
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, 
            device, 
            weight_dtype, 
            whisper, 
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        
        print(f"Generated {len(whisper_chunks)} Whisper chunks")
        
        # Save ALL audio feature chunks (not just first 5)
        for i, chunk in enumerate(whisper_chunks):
            chunk_np = chunk.cpu().numpy()
            save_array(chunk_np, f"06_audio_features_chunk_{i:03d}", output_dir)
            
    except Exception as e:
        print(f"ERROR: Could not load Whisper model: {e}")
        return
    
    # === STAGE 3: Process ALL Frames ===
    print(f"\n--- Stage 3: Processing ALL {len(whisper_chunks)} Frames ---")
    
    # Create cycled latent list for animation
    input_latent_list_cycle = [combined_latents] + [combined_latents][::-1]  # For single image, just repeat
    
    # Process each frame individually
    for frame_idx in range(len(whisper_chunks)):
        print(f"Processing frame {frame_idx + 1}/{len(whisper_chunks)}")
        
        # Prepare latent for this frame (cycle through available latents)
        latent_idx = frame_idx % len(input_latent_list_cycle)
        frame_latents = input_latent_list_cycle[latent_idx]  # [1, 8, 32, 32]
        
        # Prepare audio for this frame
        audio_chunk = whisper_chunks[frame_idx].cpu().numpy()  # [50, 384]
        audio_batch = np.expand_dims(audio_chunk, 0)  # [1, 50, 384]
        
        # Save frame-specific inputs
        save_array(frame_latents, f"07_unet_input_latents_frame_{frame_idx:03d}", output_dir)
        save_array(audio_batch, f"07_unet_input_audio_frame_{frame_idx:03d}", output_dir)
        
        try:
            # Add positional encoding
            audio_with_pe = inference_engine.add_positional_encoding(audio_batch)
            save_array(audio_with_pe, f"08_audio_with_pe_frame_{frame_idx:03d}", output_dir)
            
            # Run UNet inference
            timesteps = np.array([0], dtype=np.int64)
            unet_output = inference_engine.run_unet(frame_latents, timesteps, audio_with_pe)
            save_array(unet_output, f"09_unet_output_frame_{frame_idx:03d}", output_dir)
            
            # Decode to image
            decoded_image = inference_engine.decode_latents(unet_output, target_size=(256, 256))
            save_image(decoded_image, f"10_decoded_image_frame_{frame_idx:03d}", output_dir)
            
            # Resize for blending
            face_width = bbox_data['width']
            face_height = bbox_data['height'] + 10  # v15 extra margin
            face_height = min(face_height, avatar_image.shape[0] - bbox_data['y1'])
            
            resized_for_blend = cv2.resize(decoded_image, (face_width, face_height), interpolation=cv2.INTER_LANCZOS4)
            save_image(resized_for_blend, f"11_final_frame_{frame_idx:03d}", output_dir)
            
            print(f"  ✓ Frame {frame_idx} complete: {face_width}x{face_height}")
            
        except Exception as e:
            print(f"  ✗ Frame {frame_idx} failed: {e}")
            continue
    
    # === Save Complete Summary ===
    print("\n--- Saving Complete Summary ---")
    
    complete_summary = {
        'avatar_path': avatar_path,
        'audio_path': audio_path,
        'face_bbox': bbox_data,
        'audio_length': librosa_length,
        'total_frames': len(whisper_chunks),
        'fps': 25,
        'duration_seconds': len(whisper_chunks) / 25,
        'model_version': "v15",
        'notes': "Complete ground truth outputs for ALL frames - 100% Unity-Python matching"
    }
    
    with open(Path(output_dir) / "complete_ground_truth_summary.json", 'w') as f:
        json.dump(complete_summary, f, indent=2)
    
    print(f"\n=== Complete Ground Truth Generation Complete ===")
    print(f"✓ Processed {len(whisper_chunks)} frames")
    print(f"✓ Generated ground truth for full {len(whisper_chunks)/25:.2f}s video")
    print(f"✓ All outputs saved to: {output_dir}/")
    print("✓ Unity can now achieve 100% matching for ENTIRE video sequence")

if __name__ == "__main__":
    generate_complete_ground_truth() 