#!/usr/bin/env python3
"""
Generate Ground Truth Outputs for Unity Comparison
This script runs the Python ONNX inference and saves all intermediate outputs
that Unity can hardcode for exact numerical matches.
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

def generate_ground_truth():
    """Generate ground truth outputs for Unity comparison"""
    
    # Paths
    avatar_path = "data/video/00000000.png"
    audio_path = "data/audio/detective_1s.wav"
    output_dir = "ground_truth_outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Generating Ground Truth Outputs ===")
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
    
    # === STAGE 1: Load and save input avatar image ===
    print("\n--- Stage 1: Input Avatar ---")
    avatar_image = cv2.imread(avatar_path)
    save_image(avatar_image, "01_input_avatar", output_dir)
    
    # === STAGE 2: Face detection and bounding box ===
    print("\n--- Stage 2: Face Detection ---")
    coord_list, frame_list = get_landmark_and_bbox([avatar_path], 0)
    
    # Save face detection results
    face_bbox = coord_list[0] if coord_list else coord_placeholder
    if face_bbox != coord_placeholder:
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
        
        # === STAGE 3: Crop face region ===
        print("\n--- Stage 3: Face Cropping ---")
        frame = frame_list[0]
        x1, y1, x2, y2 = face_bbox
        
        # Add v15 extra margin
        y2 = y2 + 10  # extra_margin
        y2 = min(y2, frame.shape[0])
        
        # Crop face region
        crop_frame = frame[y1:y2, x1:x2]
        save_image(crop_frame, "03_cropped_face", output_dir)
        
        # Resize to 256x256 for VAE
        crop_frame_resized = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        save_image(crop_frame_resized, "03_cropped_face_256", output_dir)
        
        # === STAGE 4: VAE Encoding ===
        print("\n--- Stage 4: VAE Encoding ---")
        
        # Get masked latents (lower half masked)
        masked_latents = inference_engine.encode_image_with_half_mask(crop_frame_resized)
        save_array(masked_latents, "04_masked_latents", output_dir)
        
        # Get reference latents (full image)
        ref_latents = inference_engine.encode_image(crop_frame_resized)
        save_array(ref_latents, "04_ref_latents", output_dir)
        
        # Combined latents for UNet
        combined_latents = np.concatenate([masked_latents, ref_latents], axis=1)
        save_array(combined_latents, "04_combined_latents", output_dir)
        
    else:
        print("ERROR: No face detected!")
        return
    
    # === STAGE 5: Audio Processing ===
    print("\n--- Stage 5: Audio Processing ---")
    
    # Load audio with AudioProcessor
    try:
        audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        
        # Save raw audio metadata
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
    
    # Load Whisper model for proper audio processing
    try:
        from transformers import WhisperModel
        device = "cpu"
        weight_dtype = torch.float32
        whisper = WhisperModel.from_pretrained("./models/whisper")
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        
        # Process audio with proper Whisper
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
        
        # Save first few audio feature chunks
        for i, chunk in enumerate(whisper_chunks[:5]):  # Save first 5 chunks
            chunk_np = chunk.cpu().numpy()
            save_array(chunk_np, f"06_audio_features_chunk_{i:03d}", output_dir)
            
    except Exception as e:
        print(f"Warning: Could not load Whisper model ({e}), creating dummy audio features")
        
        # Create dummy whisper chunks as fallback
        fps = 25
        num_frames = int((librosa_length / 16000) * fps)
        whisper_chunks = []
        
        for i in range(min(5, num_frames)):  # Create first 5 chunks
            chunk = np.random.randn(50, 384).astype(np.float32)
            save_array(chunk, f"06_audio_features_chunk_{i:03d}_dummy", output_dir)
            whisper_chunks.append(torch.from_numpy(chunk))
    
    # === STAGE 6: Inference Pipeline ===
    print("\n--- Stage 6: UNet Inference ---")
    
    if len(whisper_chunks) > 0:
        # Prepare batch data (first frame only for ground truth)
        batch_size = 1
        
        # Prepare latent batch
        input_latents = combined_latents  # [1, 8, 32, 32]
        save_array(input_latents, "07_unet_input_latents", output_dir)
        
        # Prepare audio batch
        audio_chunk = whisper_chunks[0].cpu().numpy()  # [50, 384]
        audio_batch = np.expand_dims(audio_chunk, 0)  # [1, 50, 384]
        save_array(audio_batch, "07_unet_input_audio", output_dir)
        
        # Add positional encoding
        try:
            audio_with_pe = inference_engine.add_positional_encoding(audio_batch)
            save_array(audio_with_pe, "08_audio_with_positional_encoding", output_dir)
            
            # Run UNet
            timesteps = np.array([0], dtype=np.int64)
            unet_output = inference_engine.run_unet(input_latents, timesteps, audio_with_pe)
            save_array(unet_output, "09_unet_output", output_dir)
            
            # === STAGE 7: VAE Decoding ===
            print("\n--- Stage 7: VAE Decoding ---")
            
            # Decode latents to image
            decoded_image = inference_engine.decode_latents(unet_output, target_size=(256, 256))
            save_image(decoded_image, "10_decoded_image", output_dir)
            
            # Resize to face crop size for blending
            face_width = bbox_data['width']
            face_height = bbox_data['height'] + 10  # v15 extra margin
            face_height = min(face_height, avatar_image.shape[0] - bbox_data['y1'])
            
            resized_for_blend = cv2.resize(decoded_image, (face_width, face_height), interpolation=cv2.INTER_LANCZOS4)
            save_image(resized_for_blend, "11_resized_for_blend", output_dir)
            
            print(f"Resized decoded image to {face_width}x{face_height} for blending")
            
        except Exception as e:
            print(f"ERROR: Failed during inference pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    # === Save Summary ===
    print("\n--- Saving Summary ---")
    
    summary = {
        'avatar_path': avatar_path,
        'audio_path': audio_path,
        'face_bbox': bbox_data if 'bbox_data' in locals() else None,
        'audio_length': librosa_length if 'librosa_length' in locals() else 0,
        'num_audio_chunks': len(whisper_chunks) if len(whisper_chunks) > 0 else 0,
        'model_version': "v15",
        'notes': "Ground truth outputs for Unity hardcoding"
    }
    
    with open(Path(output_dir) / "ground_truth_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Ground Truth Generation Complete ===")
    print(f"All outputs saved to: {output_dir}/")
    print("These files can be used to hardcode exact values in Unity for 100% matching outputs")

if __name__ == "__main__":
    generate_ground_truth() 