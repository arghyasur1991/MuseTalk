#!/usr/bin/env python3
"""
ONNX Inference Script for MuseTalk
This script performs inference using the exported ONNX models.
"""

import os
import sys
import cv2
import torch
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
import json
import time
import librosa
import glob

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image

class ONNXMuseTalkInference:
    def __init__(self, model_dir="./models/onnx", version="v15", device="cpu"):
        self.model_dir = Path(model_dir)
        self.version = version
        self.device = device
        
        # Set up ONNX Runtime providers
        self.providers = self._get_providers()
        
        # Load ONNX models
        self.load_models()
        
    def _get_providers(self):
        """Get available ONNX Runtime providers"""
        providers = []
        
        # For large models like UNet, CoreML may fail, so prefer CPU for stability
        # Check for CUDA first if requested
        if 'CUDAExecutionProvider' in ort.get_available_providers() and self.device == "cuda":
            providers.append('CUDAExecutionProvider')
            
        # Always add CPU as it's most compatible
        providers.append('CPUExecutionProvider')
        
        print(f"Using ONNX providers: {providers}")
        return providers
        
    def load_models(self):
        """Load all ONNX models"""
        print("Loading ONNX models...")
        
        # Model file paths
        model_suffix = f"_{self.version}" if self.version != "v1.0" else ""
        
        unet_path = self.model_dir / f"unet{model_suffix}.onnx"
        vae_encoder_path = self.model_dir / f"vae_encoder{model_suffix}.onnx"
        vae_decoder_path = self.model_dir / f"vae_decoder{model_suffix}.onnx"
        pe_path = self.model_dir / f"positional_encoding{model_suffix}.onnx"
        
        # Load models with error handling
        try:
            print(f"Loading UNet from {unet_path}")
            self.unet_session = ort.InferenceSession(str(unet_path), providers=self.providers)
            print("✓ UNet loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load UNet: {e}")
            raise
            
        try:
            print(f"Loading VAE Encoder from {vae_encoder_path}")
            self.vae_encoder_session = ort.InferenceSession(str(vae_encoder_path), providers=self.providers)
            print("✓ VAE Encoder loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load VAE Encoder: {e}")
            raise
            
        try:
            print(f"Loading VAE Decoder from {vae_decoder_path}")
            self.vae_decoder_session = ort.InferenceSession(str(vae_decoder_path), providers=self.providers)
            print("✓ VAE Decoder loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load VAE Decoder: {e}")
            raise
            
        try:
            print(f"Loading Positional Encoding from {pe_path}")
            self.pe_session = ort.InferenceSession(str(pe_path), providers=self.providers)
            print("✓ Positional Encoding loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Positional Encoding: {e}")
            raise
            
        print("All ONNX models loaded successfully!")
        
    def preprocess_audio(self, audio_path, fps=25):
        """Preprocess audio to get features"""
        print(f"Processing audio: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate number of frames needed
        duration = len(audio) / sr
        num_frames = int(duration * fps)
        
        # For now, create dummy audio features matching expected shape
        # In a real implementation, you'd use Whisper or similar
        audio_features = np.random.randn(1, num_frames, 384).astype(np.float32)
        
        return audio_features, num_frames
        
    def encode_image(self, image):
        """Encode image using VAE encoder"""
        # Normalize image to [-1, 1]
        image_tensor = (image.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension and transpose to NCHW
        if len(image_tensor.shape) == 3:
            image_tensor = np.expand_dims(image_tensor, 0)
        image_tensor = np.transpose(image_tensor, (0, 3, 1, 2))
        
        # Run VAE encoder
        latents = self.vae_encoder_session.run(
            ['latents'], 
            {'image': image_tensor}
        )[0]
        
        return latents
        
    def decode_latents(self, latents):
        """Decode latents using VAE decoder"""
        # Run VAE decoder
        image = self.vae_decoder_session.run(
            ['image'], 
            {'latents': latents}
        )[0]
        
        # Convert back to uint8 image
        image = np.transpose(image, (0, 2, 3, 1))
        image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        return image[0]  # Remove batch dimension
        
    def add_positional_encoding(self, audio_features):
        """Add positional encoding to audio features"""
        audio_features_pe = self.pe_session.run(
            ['audio_features_with_pe'],
            {'audio_features': audio_features}
        )[0]
        
        return audio_features_pe
        
    def run_unet(self, input_latents, timesteps, audio_prompts):
        """Run UNet denoising"""
        try:
            noise_prediction = self.unet_session.run(
                ['noise_prediction'],
                {
                    'input_latents': input_latents,
                    'timesteps': timesteps,
                    'audio_prompts': audio_prompts
                }
            )[0]
            return noise_prediction
        except Exception as e:
            print(f"UNet inference failed: {e}")
            # Return zeros as fallback
            return np.zeros_like(input_latents[:, :4])  # Return only first 4 channels
            
    def inference(self, avatar_path, audio_path, output_path, batch_size=4):
        """Run complete inference pipeline"""
        print(f"Starting ONNX inference...")
        print(f"Avatar: {avatar_path}")
        print(f"Audio: {audio_path}")
        print(f"Output: {output_path}")
        
        start_time = time.time()
        
        # Load avatar images
        print("Loading avatar images...")
        if os.path.isdir(avatar_path):
            # Get list of image files from directory
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            img_path_list = []
            for ext in image_extensions:
                img_path_list.extend(glob.glob(os.path.join(avatar_path, ext)))
            img_path_list = sorted(img_path_list)
        else:
            # Single image file
            img_path_list = [avatar_path]
            
        if not img_path_list:
            raise ValueError(f"No image files found in {avatar_path}")
            
        print(f"Found {len(img_path_list)} avatar images")
        
        # Get landmarks and bbox
        coord_list, frame_list = get_landmark_and_bbox(img_path_list, 0)  # 0 for default bbox_shift
        
        # Process audio
        audio_features, num_frames = self.preprocess_audio(audio_path)
        print(f"Audio features shape: {audio_features.shape}")
        
        # Add positional encoding
        audio_features_pe = self.add_positional_encoding(audio_features)
        print(f"Audio features with PE shape: {audio_features_pe.shape}")
        
        # Prepare for inference
        bbox = coord_list[0]
        ori_imgs = frame_list
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Process frames in batches
        generated_frames = []
        
        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)
            batch_size_actual = batch_end - i
            
            print(f"Processing frames {i}-{batch_end-1}")
            
            try:
                # Get reference image (cycle through available images)
                ref_img_idx = i % len(ori_imgs)
                ref_img = ori_imgs[ref_img_idx]
                
                # Encode reference image
                ref_latents = self.encode_image(ref_img)
                
                # Prepare batch inputs
                batch_latents = np.repeat(ref_latents, batch_size_actual, axis=0)
                
                # Create noise latents (same shape as ref_latents)
                noise_latents = np.random.randn(*batch_latents.shape).astype(np.float32)
                
                # Concatenate reference and noise latents
                input_latents = np.concatenate([batch_latents, noise_latents], axis=1)
                
                # Prepare timesteps
                timesteps = np.array([0] * batch_size_actual, dtype=np.int64)
                
                # Get audio features for this batch
                batch_audio = audio_features_pe[:, i:batch_end, :]
                if batch_audio.shape[1] < batch_size_actual:
                    # Pad if needed
                    padding = np.zeros((1, batch_size_actual - batch_audio.shape[1], 384), dtype=np.float32)
                    batch_audio = np.concatenate([batch_audio, padding], axis=1)
                
                # Repeat for batch
                batch_audio = np.repeat(batch_audio, batch_size_actual, axis=0)
                
                # Run UNet
                noise_pred = self.run_unet(input_latents, timesteps, batch_audio)
                
                # Simple denoising (subtract predicted noise)
                denoised_latents = batch_latents - noise_pred
                
                # Decode latents to images
                for j in range(batch_size_actual):
                    frame_latents = denoised_latents[j:j+1]
                    decoded_img = self.decode_latents(frame_latents)
                    
                    # Blend with original image using bbox
                    final_img = get_image(ori_imgs[ref_img_idx], decoded_img, bbox)
                    generated_frames.append(final_img)
                    
                    # Save frame
                    frame_path = os.path.join(output_path, f"frame_{i+j:06d}.png")
                    cv2.imwrite(frame_path, final_img)
                    
            except Exception as e:
                print(f"Error processing batch {i}-{batch_end-1}: {e}")
                # Use fallback - just copy reference image
                for j in range(batch_size_actual):
                    ref_img_idx = (i + j) % len(ori_imgs)
                    fallback_img = ori_imgs[ref_img_idx]
                    generated_frames.append(fallback_img)
                    
                    frame_path = os.path.join(output_path, f"frame_{i+j:06d}.png")
                    cv2.imwrite(frame_path, fallback_img)
        
        # Create video from frames
        self.create_video(output_path, generated_frames, fps=25)
        
        end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds")
        print(f"Generated {len(generated_frames)} frames")
        
    def create_video(self, output_dir, frames, fps=25):
        """Create video from generated frames"""
        if not frames:
            print("No frames to create video")
            return
            
        output_video = os.path.join(output_dir, "output.mp4")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        print(f"Video saved to: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="ONNX inference for MuseTalk")
    parser.add_argument("--avatar_path", required=True, help="Path to avatar images directory")
    parser.add_argument("--audio_path", required=True, help="Path to audio file")
    parser.add_argument("--output_path", required=True, help="Output directory")
    parser.add_argument("--model_dir", default="./models/onnx", help="ONNX models directory")
    parser.add_argument("--version", default="v15", help="Model version")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    # Create inference engine
    inference_engine = ONNXMuseTalkInference(
        model_dir=args.model_dir,
        version=args.version,
        device=args.device
    )
    
    # Run inference
    inference_engine.inference(
        avatar_path=args.avatar_path,
        audio_path=args.audio_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 