#!/usr/bin/env python3
"""
Deterministic PyTorch Inference Script for MuseTalk
This script uses deterministic VAE (mode) to match ONNX behavior exactly
"""

import os
import sys
import cv2
import torch
import argparse
import time
from pathlib import Path
import librosa
import numpy as np
import glob

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

def run_deterministic_pytorch_inference(avatar_path, audio_path, output_path, max_images=10):
    """Run PyTorch inference with deterministic VAE to match ONNX"""
    
    print(f"Starting DETERMINISTIC PyTorch inference...")
    print(f"Avatar: {avatar_path}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_path}")
    print(f"Max images for debugging: {max_images}")
    
    start_time = time.time()
    
    # Load models with deterministic VAE
    device = "cpu"
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    
    # Make VAE deterministic (use mode() instead of sample())
    vae._deterministic = True
    print("✓ Using DETERMINISTIC VAE (mode instead of sample)")
    
    # Initialize face parsing
    fp = FaceParsing()
    
    # Load avatar images
    print("Loading avatar images...")
    if os.path.isdir(avatar_path):
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        img_path_list = []
        for ext in image_extensions:
            img_path_list.extend(glob.glob(os.path.join(avatar_path, ext)))
        img_path_list = sorted(img_path_list)
        
        if max_images > 0 and len(img_path_list) > max_images:
            img_path_list = img_path_list[:max_images]
            print(f"Using first {max_images} images for faster debugging")
    else:
        img_path_list = [avatar_path]
        
    if not img_path_list:
        raise ValueError(f"No image files found in {avatar_path}")
        
    print(f"Found {len(img_path_list)} avatar images")
    
    # Get landmarks and bbox
    coord_list, frame_list = get_landmark_and_bbox(img_path_list, 0)
    
    # Process each frame to create input latents
    print("Processing frames to create latents...")
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        # Add extra margin for v15
        y2 = y2 + 10
        y2 = min(y2, frame.shape[0])
        
        # Crop face region
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        # Get latents for UNet (now deterministic)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    
    print(f"Created {len(input_latent_list)} latent tensors")
    
    # Process audio
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    
    print("Processing audio with Whisper...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    
    # Load Whisper model
    weight_dtype = torch.float32
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # Process audio with Whisper
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
    
    # Set global timesteps
    timesteps = torch.tensor([0], dtype=torch.long, device=device)
    
    # Cycle latents
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process frames in batches
    res_frame_list = []
    batch_size = 4
    total_batches = int(np.ceil(float(len(whisper_chunks)) / batch_size))
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(whisper_chunks))
        batch_size_actual = end_idx - start_idx
        
        print(f"Processing batch {batch_idx+1}/{total_batches} (frames {start_idx}-{end_idx-1})")
        
        # Prepare batch
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
        
        # Stack batches
        whisper_batch = torch.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)
        
        # Apply positional encoding to audio
        audio_feature_batch = pe(whisper_batch)
        
        # Prepare timesteps
        batch_timesteps = timesteps.repeat(batch_size_actual)
        
        # Run UNet inference
        with torch.no_grad():
            pred_latents = unet.model(
                sample=latent_batch,
                timestep=batch_timesteps,
                encoder_hidden_states=audio_feature_batch,
                return_dict=False
            )[0]
        
        # Decode latents to images
        for j in range(batch_size_actual):
            frame_latents = pred_latents[j:j+1]
            
            # Decode using VAE decoder (deterministic)
            decoded_img = vae.decode_latents(frame_latents)
            
            # Resize to face crop size
            frame_idx = start_idx + j
            bbox = coord_list[frame_idx % len(coord_list)]
            x1, y1, x2, y2 = bbox
            y2 = y2 + 10
            y2 = min(y2, frame_list[frame_idx % len(frame_list)].shape[0])
            
            target_width = x2 - x1
            target_height = y2 - y1
            res_frame = cv2.resize(decoded_img[0], (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Blend with original image
            ori_frame = frame_list[frame_idx % len(frame_list)].copy()
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode='jaw', fp=fp)
            
            res_frame_list.append(combine_frame)
            
            # Save frame
            frame_path = os.path.join(output_path, f"{str(len(res_frame_list)-1).zfill(8)}.png")
            cv2.imwrite(frame_path, combine_frame)
    
    # Create video from frames
    create_video(output_path, res_frame_list, fps=25, audio_path=audio_path)
    
    end_time = time.time()
    print(f"Deterministic PyTorch inference completed in {end_time - start_time:.2f} seconds")
    print(f"Generated {len(res_frame_list)} frames")

def create_video(output_dir, frames, fps=25, audio_path=None):
    """Create video from generated frames and add audio"""
    if not frames:
        print("No frames to create video")
        return
        
    output_video = os.path.join(output_dir, "output.mp4")
    temp_video = os.path.join(output_dir, "temp_video.mp4")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
        
    out.release()
    
    # Add audio if provided
    if audio_path and os.path.exists(audio_path):
        print(f"Adding audio from {audio_path}")
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_video} {output_video}"
        os.system(cmd_combine_audio)
        os.remove(temp_video)
    else:
        os.rename(temp_video, output_video)
    
    print(f"Video saved to: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Deterministic PyTorch inference for MuseTalk")
    parser.add_argument("--avatar_path", required=True, help="Path to avatar images directory")
    parser.add_argument("--audio_path", required=True, help="Path to audio file")
    parser.add_argument("--output_path", required=True, help="Output directory")
    parser.add_argument("--max_images", type=int, default=10, help="Max images for debugging (0 for all)")
    
    args = parser.parse_args()
    
    run_deterministic_pytorch_inference(
        avatar_path=args.avatar_path,
        audio_path=args.audio_path,
        output_path=args.output_path,
        max_images=args.max_images
    )

if __name__ == "__main__":
    main() 