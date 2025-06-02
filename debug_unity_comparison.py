#!/usr/bin/env python3
"""
Comprehensive debug script for Unity-Python MuseTalk comparison
Outputs debug information in same format as Unity for easy comparison
"""

import sys
import os
import cv2
import numpy as np
import json
from pathlib import Path
import librosa
import argparse

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
from musetalk.utils.audio_processor import AudioProcessor
import math

class UnityPythonDebugger:
    def __init__(self, output_dir="debug_unity_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create stage directories matching Unity structure
        self.stages = {
            "01_input": "Input avatar images",
            "02_face_detection": "Face detection results",
            "03_cropped_faces": "Cropped face regions (256x256)",
            "04_latents": "VAE encoder outputs",
            "05_audio": "Raw audio data",
            "06_audio_features": "Whisper feature chunks"
        }
        
        for stage in self.stages.keys():
            (self.output_dir / stage).mkdir(exist_ok=True)
            
        print(f"[PythonDebugger] Created debug output directory: {self.output_dir}")
    
    def save_array_debug(self, data, filename, subfolder, description=""):
        """Save array data in Unity-compatible format"""
        folder_path = self.output_dir / subfolder
        folder_path.mkdir(exist_ok=True)
        
        file_path = folder_path / f"{filename}.txt"
        
        with open(file_path, 'w') as f:
            f.write(f"Array Length: {len(data) if hasattr(data, '__len__') else 'N/A'}\n")
            f.write(f"Data Type: {type(data).__name__}\n")
            if description:
                f.write(f"Description: {description}\n")
            f.write("\n")
            
            # Handle different data types
            if isinstance(data, np.ndarray):
                flat_data = data.flatten()
                f.write(f"Array Shape: {data.shape}\n")
                f.write(f"Total Elements: {data.size}\n")
                f.write(f"Sample (first {min(100, len(flat_data))} elements):\n")
                f.write("\n")
                
                for i in range(min(100, len(flat_data))):
                    f.write(f"[{i}] = {flat_data[i]:.6f}\n")
                    
                if len(flat_data) > 100:
                    f.write(f"... and {len(flat_data) - 100} more elements\n")
                    
                # Statistics
                f.write("\nStatistics:\n")
                f.write(f"Min: {flat_data.min():.6f}\n")
                f.write(f"Max: {flat_data.max():.6f}\n")
                f.write(f"Mean: {flat_data.mean():.6f}\n")
                
            elif isinstance(data, list):
                f.write(f"Sample (first {min(100, len(data))} elements):\n")
                f.write("\n")
                
                for i in range(min(100, len(data))):
                    f.write(f"[{i}] = {data[i]:.6f if isinstance(data[i], (int, float)) else data[i]}\n")
                    
                if len(data) > 100:
                    f.write(f"... and {len(data) - 100} more elements\n")
        
        print(f"[PythonDebugger] Saved array debug: {file_path}")
    
    def save_face_detection_results(self, bboxes, filename="face_detection"):
        """Save face detection results in Unity format"""
        folder_path = self.output_dir / "02_face_detection"
        file_path = folder_path / f"{filename}_bboxes.txt"
        
        with open(file_path, 'w') as f:
            f.write("Face Detection Results\n")
            f.write(f"Number of faces: {len(bboxes)}\n")
            f.write("\n")
            
            for i, bbox in enumerate(bboxes):
                if np.array_equal(bbox, coord_placeholder):
                    f.write(f"Face {i}: NO FACE DETECTED\n")
                else:
                    x1, y1, x2, y2 = bbox
                    f.write(f"Face {i}:\n")
                    f.write(f"  Bbox: [{x1}, {y1}, {x2}, {y2}]\n")
                    f.write(f"  Width: {x2 - x1}\n")
                    f.write(f"  Height: {y2 - y1}\n")
                    f.write("\n")
        
        print(f"[PythonDebugger] Saved face detection results: {file_path}")
    
    def debug_face_detection(self, image_path):
        """Debug face detection stage"""
        print(f"\n=== STAGE 1: INPUT PROCESSING ===")
        
        # Load and save input image
        image = cv2.imread(image_path)
        print(f"Input image shape: {image.shape}")
        
        # Save input image (convert BGR to RGB for consistency)
        input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(self.output_dir / "01_input" / "input_avatar_000.png"), 
                   cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR))
        
        print(f"\n=== STAGE 2: FACE DETECTION ===")
        
        # Run face detection
        coords_list, frames = get_landmark_and_bbox([image_path], upperbondrange=0)
        
        # Save face detection results
        self.save_face_detection_results(coords_list)
        
        # Process each detected face
        for i, (bbox, frame) in enumerate(zip(coords_list, frames)):
            if np.array_equal(bbox, coord_placeholder):
                print(f"Frame {i}: NO FACE DETECTED")
                continue
                
            x1, y1, x2, y2 = bbox
            print(f"Frame {i}: Face bbox = [{x1}, {y1}, {x2}, {y2}]")
            
            # Add v15 extra margin
            y2_with_margin = y2 + 10
            y2_with_margin = min(y2_with_margin, frame.shape[0])
            
            print(f"Frame {i}: With v15 margin = [{x1}, {y1}, {x2}, {y2_with_margin}]")
            
            # Crop face region
            crop_frame = frame[y1:y2_with_margin, x1:x2]
            print(f"Frame {i}: Crop shape before resize = {crop_frame.shape}")
            
            # Resize to 256x256 (matching Unity)
            resized_crop = cv2.resize(crop_frame, (256, 256), cv2.INTER_LANCZOS4)
            print(f"Frame {i}: Resized to {resized_crop.shape}")
            
            # Save cropped face (convert BGR to RGB)
            crop_rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(self.output_dir / "03_cropped_faces" / f"cropped_face_{i:03d}.png"),
                       cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            
            # Debug image preprocessing (matching Unity TextureToTensor)
            self.debug_image_preprocessing(resized_crop, i)
        
        return coords_list, frames
    
    def debug_image_preprocessing(self, image, index):
        """Debug image preprocessing to match Unity VAE preprocessing"""
        print(f"\n=== STAGE 3: IMAGE PREPROCESSING (Face {index}) ===")
        
        # Convert BGR to RGB (matching Unity)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1] (matching Unity)
        x = np.asarray(img_rgb, dtype=np.float32) / 255.0
        print(f"Normalized shape: {x.shape}, range: [{x.min():.6f}, {x.max():.6f}]")
        
        # Transpose [H,W,C] -> [C,H,W] (matching Unity)
        x = np.transpose(x, (2, 0, 1))
        print(f"Transposed shape: {x.shape}")
        
        # Normalize to [-1,1] (matching Unity)
        for c in range(3):
            x[c] = (x[c] - 0.5) / 0.5
            
        print(f"Final preprocessing shape: {x.shape}, range: [{x.min():.6f}, {x.max():.6f}]")
        
        # Save preprocessing debug data
        self.save_array_debug(x, f"preprocessed_image_{index:03d}", "04_latents", 
                             "Image after VAE preprocessing (normalized to [-1,1])")
    
    def debug_audio_processing(self, audio_path, duration_seconds=None):
        """Debug audio processing to match Unity"""
        print(f"\n=== STAGE 4: AUDIO PROCESSING ===")
        
        if audio_path and os.path.exists(audio_path):
            # Load actual audio file
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            
            # Resample to 16kHz to match Unity
            target_sr = 16000
            if original_sr != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
            
            duration_seconds = len(audio_data) / target_sr
            sample_rate = target_sr
            
            print(f"Loaded audio from: {audio_path}")
            print(f"Original sample rate: {original_sr}Hz")
            print(f"Resampled to: {target_sr}Hz")
        else:
            # Fallback to dummy data
            if duration_seconds is None:
                duration_seconds = 5.0
            sample_rate = 16000
            num_samples = int(duration_seconds * sample_rate)
            audio_data = np.random.randn(num_samples).astype(np.float32) * 0.1
            print(f"Using dummy audio data (no file provided)")
        
        print(f"Audio duration: {duration_seconds:.2f}s")
        print(f"Audio sample rate: {sample_rate}Hz") 
        print(f"Audio data length: {len(audio_data)} samples")
        
        # Save audio debug info
        self.save_array_debug(audio_data, "raw_audio_data", "05_audio", 
                             f"Raw audio data ({sample_rate}Hz, {duration_seconds:.2f}s)")
        
        # Calculate frame count (matching Unity exactly)
        num_frames = math.floor((len(audio_data) / sample_rate) * 25)  # fps=25
        print(f"Calculated frames (Python method): {num_frames}")
        
        # Generate audio features (matching Unity format)
        time_steps = 50
        feature_dim = 384
        chunk_size = time_steps * feature_dim
        
        print(f"\n=== STAGE 5: AUDIO FEATURES ===")
        print(f"Generating {num_frames} audio feature chunks")
        print(f"Each chunk: [{time_steps}, {feature_dim}] = {chunk_size} elements")
        
        for i in range(min(num_frames, 10)):  # Save first 10 chunks
            # Generate features with variation (matching Unity)
            chunk = np.random.randn(chunk_size).astype(np.float32) * 0.1
            
            # Add some audio-based variation
            if len(audio_data) > 0:
                audio_sample_index = int((i / num_frames) * (len(audio_data) - 1))
                audio_variation = audio_data[audio_sample_index] * 0.1
                chunk += audio_variation
            
            self.save_array_debug(chunk, f"audio_features_chunk_{i:03d}", "06_audio_features",
                                 f"Whisper feature chunk {i} [{time_steps}, {feature_dim}]")
        
        return num_frames
    
    def run_complete_debug(self, image_path, audio_path=None, audio_duration=None):
        """Run complete debugging pipeline"""
        print("="*80)
        print("PYTHON MUSETALK DEBUG - UNITY COMPARISON")
        print("="*80)
        
        # Debug face detection and image processing
        coords_list, frames = self.debug_face_detection(image_path)
        
        # Debug audio processing
        num_frames = self.debug_audio_processing(audio_path, audio_duration)
        
        print(f"\n=== SUMMARY ===")
        print(f"Input image: {image_path}")
        if audio_path:
            print(f"Input audio: {audio_path}")
        print(f"Detected faces: {len([c for c in coords_list if not np.array_equal(c, coord_placeholder)])}")
        print(f"Audio frames: {num_frames}")
        print(f"Debug output saved to: {self.output_dir}")
        print(f"\nCompare with Unity debug output folder!")

def main():
    parser = argparse.ArgumentParser(description="Generate Unity-compatible Python debug outputs for MuseTalk")
    parser.add_argument("--image", default="results/v15/avatars/avator_1/full_imgs/00000000.png", 
                       help="Path to avatar image")
    parser.add_argument("--audio", default=None, 
                       help="Path to audio file")
    parser.add_argument("--duration", type=float, default=None,
                       help="Audio duration in seconds (if not loading from file)")
    
    args = parser.parse_args()
    
    # Process the test avatar image
    avatar_path = args.image
    
    if not os.path.exists(avatar_path):
        print(f"Avatar image not found: {avatar_path}")
        print("Please ensure you have the test data available")
        return
    
    if args.audio and not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        print("Will use dummy audio data")
        args.audio = None
    
    # Create debugger and run comparison
    debugger = UnityPythonDebugger()
    debugger.run_complete_debug(avatar_path, args.audio, args.duration)

if __name__ == "__main__":
    main() 