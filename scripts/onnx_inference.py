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
import copy

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing

class ONNXMuseTalkInference:
    def __init__(self, model_dir="./models/onnx", version="v15", device="cpu"):
        self.model_dir = Path(model_dir)
        self.version = version
        self.device = device
        
        # Set up ONNX Runtime providers
        self.providers = self._get_providers()
        
        # Initialize face parsing model
        print("Initializing face parsing model...")
        if version == "v15":
            self.fp = FaceParsing()  # v15 uses default parameters
        else:
            self.fp = FaceParsing()  # v1 also uses default for now
        
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
        
    def encode_image(self, image):
        """Encode image using VAE encoder - FIXED: match PyTorch normalization"""
        # Resize image to expected VAE input size (256x256 for the exported model)
        target_size = 256
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size))
        
        # Normalize like PyTorch VAE: first to [0,1], then transform (x - 0.5) / 0.5 = 2*x - 1
        image_normalized = image.astype(np.float32) / 255.0
        image_tensor = 2.0 * image_normalized - 1.0
        
        # Add batch dimension and transpose to NCHW
        if len(image_tensor.shape) == 3:
            image_tensor = np.expand_dims(image_tensor, 0)
        image_tensor = np.transpose(image_tensor, (0, 3, 1, 2))
        
        # Run VAE encoder
        latents = self.vae_encoder_session.run(
            ['latents'], 
            {'image': image_tensor}
        )[0]
        
        # No resizing needed - VAE produces correct 32x32 latents
        
        return latents
    
    def encode_image_with_half_mask(self, image):
        """Encode image with lower half masked (like get_latents_for_unet) - FIXED: zero out lower half"""
        # Resize image to expected VAE input size
        target_size = 256
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size))
        
        # Normalize image to [0, 1] first (like PyTorch)
        image_normalized = image.astype(np.float32) / 255.0
        
        # Apply half mask - ZERO OUT lower half (like PyTorch VAE)
        mask = np.ones((target_size, target_size), dtype=np.float32)
        mask[target_size//2:, :] = 0  # Set lower half to 0 (matching PyTorch)
        
        # Apply mask to all channels
        for c in range(3):
            image_normalized[:, :, c] *= mask
        
        # Now apply the transform normalization: (x - 0.5) / 0.5 = 2*x - 1
        image_tensor = 2.0 * image_normalized - 1.0
        
        # Add batch dimension and transpose to NCHW
        if len(image_tensor.shape) == 3:
            image_tensor = np.expand_dims(image_tensor, 0)
        image_tensor = np.transpose(image_tensor, (0, 3, 1, 2))
        
        # Run VAE encoder
        latents = self.vae_encoder_session.run(
            ['latents'], 
            {'image': image_tensor}
        )[0]
        
        # No resizing needed - VAE produces correct 32x32 latents
        
        return latents
    
    def get_latents_for_unet(self, image):
        """Replicate the exact behavior of VAE.get_latents_for_unet - FIXED: mask lower half"""
        # Get masked latents (lower half masked)
        masked_latents = self.encode_image_with_half_mask(image)
        
        # Get reference latents (full image)
        ref_latents = self.encode_image(image)
        
        # Concatenate as in original implementation
        latent_model_input = np.concatenate([masked_latents, ref_latents], axis=1)
        
        # print(f"Masked latents shape: {masked_latents.shape}")
        # print(f"Ref latents shape: {ref_latents.shape}")
        # print(f"Combined latents shape: {latent_model_input.shape}")
        
        return latent_model_input
        
    def decode_latents(self, latents, target_size=(256, 256)):
        """Decode latents using VAE decoder - FIXED: proper normalization and color space"""
        # Run VAE decoder
        image = self.vae_decoder_session.run(
            ['image'], 
            {'latents': latents}
        )[0]
        
        # Convert back to uint8 image - FIXED: use correct normalization like PyTorch
        image = np.transpose(image, (0, 2, 3, 1))
        image = (image / 2 + 0.5).clip(0, 1)  # Correct normalization: (image / 2 + 0.5)
        image = (image * 255).round().astype(np.uint8)
        image = image[0]  # Remove batch dimension
        
        # Keep as RGB - get_image function handles color conversion internally
        # DO NOT convert RGB→BGR here as it causes confusion
        
        # Resize to target size if specified
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, target_size)
            print(f"Resized decoded image to {target_size}")
        
        return image
        
    def add_positional_encoding(self, audio_features):
        """Add positional encoding to audio features"""
        audio_features_pe = self.pe_session.run(
            ['audio_features_with_pe'],
            {'audio_features': audio_features}
        )[0]
        
        return audio_features_pe
        
    def run_unet(self, input_latents, timesteps, audio_prompts):
        """Run UNet denoising"""
        noise_prediction = self.unet_session.run(
            ['noise_prediction'],
            {
                'input_latents': input_latents,
                'timesteps': timesteps,
                'audio_prompts': audio_prompts
            }
        )[0]
        return noise_prediction
        
    def inference(self, avatar_path, audio_path, output_path, batch_size=4, max_images=10):
        """Run complete inference pipeline matching PyTorch exactly"""
        print(f"Starting ONNX inference...")
        print(f"Avatar: {avatar_path}")
        print(f"Audio: {audio_path}")
        print(f"Output: {output_path}")
        print(f"Max images for debugging: {max_images}")
        
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
            
            # Use only a subset for debugging
            if max_images > 0 and len(img_path_list) > max_images:
                img_path_list = img_path_list[:max_images]
                print(f"Using first {max_images} images for faster debugging")
        else:
            # Single image file
            img_path_list = [avatar_path]
            
        if not img_path_list:
            raise ValueError(f"No image files found in {avatar_path}")
            
        print(f"Found {len(img_path_list)} avatar images")
        
        # Get landmarks and bbox
        coord_list, frame_list = get_landmark_and_bbox(img_path_list, 0)  # 0 for default bbox_shift
        
        # Process each frame to create input latents (matching PyTorch exactly)
        print("Processing frames to create latents...")
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            # Add extra margin for v15 (matching PyTorch)
            if self.version == "v15":
                y2 = y2 + 10  # extra_margin from PyTorch args
                y2 = min(y2, frame.shape[0])
            
            # Crop face region (matching PyTorch exactly)
            crop_frame = frame[y1:y2, x1:x2]
            # Resize to 256x256 (matching PyTorch exactly)
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # Get latents for UNet (matching PyTorch exactly)
            latents = self.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        print(f"Created {len(input_latent_list)} latent tensors")
        
        # Process audio properly with AudioProcessor
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel
        import torch
        
        # Set global timesteps (matching PyTorch)
        timesteps = np.array([0], dtype=np.int64)
        
        audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        
        print("Processing audio with Whisper...")
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        
        # Load Whisper model for proper audio processing
        try:
            # Try to load Whisper model - this might fail in ONNX environment
            device = "cpu"  # Force CPU for ONNX
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
            num_frames = min(len(input_latent_list), len(whisper_chunks))
            print(f"Processed {len(whisper_chunks)} audio chunks with Whisper")
            
        except Exception as e:
            print(f"Warning: Could not load Whisper model ({e}), using dummy audio features")
            # Fallback to dummy features if Whisper loading fails
            fps = 25
            num_frames = min(len(input_latent_list), int((librosa_length / 16000) * fps))
            
            # Create dummy whisper chunks for now (proper audio processing needs full Whisper model)
            whisper_chunks = []
            for i in range(num_frames):
                # Each chunk is [50, 384] - 50 time steps, 384 features
                chunk = np.random.randn(50, 384).astype(np.float32)
                whisper_chunks.append(torch.from_numpy(chunk))
        
        print(f"Created {len(whisper_chunks)} audio chunks")
        
        # Cycle latents like PyTorch (for smoothing)
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Process frames in batches (matching PyTorch exactly)
        res_frame_list = []
        total_batches = int(np.ceil(float(len(whisper_chunks)) / batch_size))
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(whisper_chunks))
            batch_size_actual = end_idx - start_idx
            
            print(f"Processing batch {batch_idx+1}/{total_batches} (frames {start_idx}-{end_idx-1})")
            
            # Prepare whisper batch
            whisper_batch = []
            latent_batch = []
            
            for i in range(start_idx, end_idx):
                # Get whisper chunk
                w = whisper_chunks[i]
                whisper_batch.append(w)
                
                # Get corresponding latent (with cycling)
                idx = i % len(input_latent_list_cycle)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            
            # Stack batches (matching PyTorch datagen)
            if isinstance(whisper_batch[0], torch.Tensor):
                whisper_batch = torch.stack(whisper_batch).numpy()  # Convert to numpy for ONNX
            else:
                whisper_batch = np.stack(whisper_batch)  # [batch_size, 50, 384]
            latent_batch = np.concatenate(latent_batch, axis=0)  # [batch_size, 8, 32, 32]
            
            # Apply positional encoding to audio (matching PyTorch)
            audio_feature_batch = self.add_positional_encoding(whisper_batch)
            
            # Prepare timesteps
            batch_timesteps = np.repeat(timesteps, batch_size_actual)
            
            # Run UNet inference
            pred_latents = self.run_unet(latent_batch, batch_timesteps, audio_feature_batch)
            
            # Decode latents to images (matching PyTorch)
            for j in range(batch_size_actual):
                frame_latents = pred_latents[j:j+1]  # [1, 4, 32, 32]
                
                # Decode using VAE decoder
                decoded_img = self.decode_latents(frame_latents, target_size=None)  # Let it be natural size
                
                # Resize to face crop size (matching PyTorch)
                frame_idx = start_idx + j
                bbox = coord_list[frame_idx % len(coord_list)]
                x1, y1, x2, y2 = bbox
                if self.version == "v15":
                    y2 = y2 + 10
                    y2 = min(y2, frame_list[frame_idx % len(frame_list)].shape[0])
                
                target_width = x2 - x1
                target_height = y2 - y1
                res_frame = cv2.resize(decoded_img.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Blend with original image (matching PyTorch)
                ori_frame = frame_list[frame_idx % len(frame_list)].copy()
                if self.version == "v15":
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode='jaw', fp=self.fp)
                else:
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=self.fp)
                
                res_frame_list.append(combine_frame)
                
                # Save frame
                frame_path = os.path.join(output_path, f"{str(len(res_frame_list)-1).zfill(8)}.png")
                cv2.imwrite(frame_path, combine_frame)
        
        # Create video from frames
        self.create_video(output_path, res_frame_list, fps=25, audio_path=audio_path)
        
        end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds")
        print(f"Generated {len(res_frame_list)} frames")
        
    def create_video(self, output_dir, frames, fps=25, audio_path=None):
        """Create video from generated frames and add audio"""
        if not frames:
            print("No frames to create video")
            return
            
        output_video = os.path.join(output_dir, "output.mp4")
        temp_video = os.path.join(output_dir, "temp_video.mp4")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer for temporary video without audio
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        
        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            print(f"Adding audio from {audio_path}")
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_video} {output_video}"
            print(f"Audio combination command: {cmd_combine_audio}")
            os.system(cmd_combine_audio)
            
            # Clean up temporary video
            os.remove(temp_video)
        else:
            # No audio, just rename temp video
            os.rename(temp_video, output_video)
        
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
    parser.add_argument("--max_images", type=int, default=10, help="Max images for debugging (0 for all)")
    
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
        batch_size=args.batch_size,
        max_images=args.max_images
    )

if __name__ == "__main__":
    main() 