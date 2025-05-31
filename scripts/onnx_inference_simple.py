#!/usr/bin/env python3
"""
Simplified ONNX Inference Script for MuseTalk
This script performs inference using simplified ONNX models.
"""

import os
import cv2
import glob
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
import sys
import onnxruntime as ort
from pathlib import Path
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import get_file_type, get_video_fps
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

class ONNXMuseTalkSimpleInference:
    """Simplified ONNX-based MuseTalk inference engine"""
    
    def __init__(self, onnx_model_dir, version="v1.5", use_gpu=True):
        self.onnx_model_dir = Path(onnx_model_dir)
        self.version = version
        self.model_suffix = "_v15" if version == "v15" else "_v1"
        
        # Setup ONNX Runtime providers
        providers = ['CPUExecutionProvider']
        if use_gpu and ort.get_available_providers():
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
            elif 'CoreMLExecutionProvider' in available_providers:
                providers.insert(0, 'CoreMLExecutionProvider')
        
        self.providers = providers
        print(f"Using ONNX Runtime providers: {providers}")
        
        # Load ONNX models
        self.load_onnx_models()
        
        # Load configuration
        config_path = self.onnx_model_dir / f"onnx_config{self.model_suffix}_simple.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def load_onnx_models(self):
        """Load all simplified ONNX models"""
        try:
            # Load UNet
            unet_path = self.onnx_model_dir / f"unet{self.model_suffix}_simple.onnx"
            if unet_path.exists():
                self.unet_session = ort.InferenceSession(str(unet_path), providers=self.providers)
                print(f"✓ Loaded UNet ONNX model: {unet_path}")
            else:
                raise FileNotFoundError(f"UNet ONNX model not found: {unet_path}")
            
            # Load VAE Encoder
            vae_encoder_path = self.onnx_model_dir / f"vae_encoder{self.model_suffix}_simple.onnx"
            if vae_encoder_path.exists():
                self.vae_encoder_session = ort.InferenceSession(str(vae_encoder_path), providers=self.providers)
                print(f"✓ Loaded VAE Encoder ONNX model: {vae_encoder_path}")
            else:
                raise FileNotFoundError(f"VAE Encoder ONNX model not found: {vae_encoder_path}")
            
            # Load VAE Decoder
            vae_decoder_path = self.onnx_model_dir / f"vae_decoder{self.model_suffix}_simple.onnx"
            if vae_decoder_path.exists():
                self.vae_decoder_session = ort.InferenceSession(str(vae_decoder_path), providers=self.providers)
                print(f"✓ Loaded VAE Decoder ONNX model: {vae_decoder_path}")
            else:
                raise FileNotFoundError(f"VAE Decoder ONNX model not found: {vae_decoder_path}")
            
            # Load Positional Encoding
            pe_path = self.onnx_model_dir / f"positional_encoding{self.model_suffix}.onnx"
            if pe_path.exists():
                self.pe_session = ort.InferenceSession(str(pe_path), providers=self.providers)
                print(f"✓ Loaded Positional Encoding ONNX model: {pe_path}")
            else:
                print("Positional Encoding ONNX model not found, will use identity")
                self.pe_session = None
                
        except Exception as e:
            print(f"Error loading ONNX models: {e}")
            raise
    
    def preprocess_image(self, img, half_mask=False):
        """Preprocess image for VAE encoder"""
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply half mask if requested
        if half_mask:
            img[128:, :] = 0  # Mask bottom half
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # Add batch dimension and rearrange to CHW
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        
        return img
    
    def encode_image(self, img):
        """Encode image using VAE encoder"""
        preprocessed = self.preprocess_image(img)
        inputs = {'image': preprocessed}
        outputs = self.vae_encoder_session.run(None, inputs)
        return outputs[0]  # Return latents
    
    def decode_latents(self, latents):
        """Decode latents using VAE decoder"""
        inputs = {'latents': latents}
        outputs = self.vae_decoder_session.run(None, inputs)
        image = outputs[0]  # [B, C, H, W]
        
        # Convert from [-1, 1] to [0, 255]
        image = (image + 1.0) / 2.0
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Convert from CHW to HWC
        image = np.transpose(image[0], (1, 2, 0))
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def get_latents_for_unet(self, img):
        """Prepare latents for UNet"""
        # Get masked latents (bottom half masked)
        masked_img = self.preprocess_image(img, half_mask=True)
        masked_inputs = {'image': masked_img}
        masked_latents = self.vae_encoder_session.run(None, masked_inputs)[0]
        
        # Get reference latents (full image)
        ref_img = self.preprocess_image(img, half_mask=False)
        ref_inputs = {'image': ref_img}
        ref_latents = self.vae_encoder_session.run(None, ref_inputs)[0]
        
        # Concatenate along channel dimension
        latent_model_input = np.concatenate([masked_latents, ref_latents], axis=1)
        
        return latent_model_input
    
    def apply_positional_encoding(self, audio_features):
        """Apply positional encoding to audio features"""
        if self.pe_session is not None:
            inputs = {'audio_features': audio_features}
            outputs = self.pe_session.run(None, inputs)
            return outputs[0]
        else:
            # Return input unchanged if no PE model
            return audio_features
    
    def run_unet(self, latent_input, audio_embedding):
        """Run UNet inference (simplified version without timesteps)"""
        inputs = {
            'latent_input': latent_input,
            'audio_embedding': audio_embedding
        }
        outputs = self.unet_session.run(None, inputs)
        return outputs[0]
    
    def process_batch(self, latent_batch, audio_batch):
        """Process a batch of latents and audio features"""
        # Apply positional encoding to audio features
        audio_with_pe = self.apply_positional_encoding(audio_batch)
        
        # Run UNet (simplified model doesn't need timesteps)
        latents_pred = self.run_unet(latent_batch, audio_with_pe)
        
        return latents_pred

def fast_check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def create_dummy_audio_features(num_frames, seq_len=50, audio_dim=384):
    """Create dummy audio features for testing"""
    # Create simple sinusoidal audio features
    audio_features = []
    for i in range(num_frames):
        # Create time-varying features
        t = np.linspace(0, 2 * np.pi, seq_len)
        feature = np.sin(t[:, np.newaxis] * (i + 1)) * np.cos(np.linspace(0, np.pi, audio_dim))
        feature = feature.astype(np.float32)
        audio_features.append(feature)
    
    return audio_features

def datagen(audio_features, vae_encode_latents, batch_size=8, delay_frame=0):
    """Data generator for batched inference"""
    audio_batch, latent_batch = [], []
    for i, w in enumerate(audio_features):
        idx = (i + delay_frame) % len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        audio_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            audio_batch = np.stack(audio_batch)
            latent_batch = np.concatenate(latent_batch, axis=0)
            yield audio_batch, latent_batch
            audio_batch, latent_batch = [], []

    # Handle the last batch
    if len(latent_batch) > 0:
        audio_batch = np.stack(audio_batch)
        latent_batch = np.concatenate(latent_batch, axis=0)
        yield audio_batch, latent_batch

def main(args):
    """Main inference function"""
    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
    
    # Initialize ONNX inference engine
    onnx_engine = ONNXMuseTalkSimpleInference(
        onnx_model_dir=args.onnx_model_dir,
        version=args.version,
        use_gpu=args.use_gpu
    )
    
    # Initialize face parser
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:
        fp = FaceParsing()
    
    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)
    
    # Process each task
    for task_id in inference_config:
        try:
            # Get task configuration
            video_path = inference_config[task_id]["video_path"]
            audio_path = inference_config[task_id]["audio_path"]
            if "result_name" in inference_config[task_id]:
                args.output_vid_name = inference_config[task_id]["result_name"]
            
            # Set bbox_shift
            if args.version == "v15":
                bbox_shift = 0
            else:
                bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
            
            # Set output paths
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}_{audio_basename}"
            
            # Create directories
            temp_dir = os.path.join(args.result_dir, f"{args.version}_onnx_simple")
            os.makedirs(temp_dir, exist_ok=True)
            
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename + ".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)
            
            # Set output video paths
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            
            # Extract frames from source video
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(temp_dir, input_basename)
                os.makedirs(save_dir_full, exist_ok=True)
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                os.system(cmd)
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                fps = get_video_fps(video_path)
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                fps = args.fps
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

            # Preprocess input images
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                print("Using saved coordinates")
                with open(crop_coord_save_path, 'rb') as f:
                    coord_list = pickle.load(f)
                frame_list = read_imgs(input_img_list)
            else:
                print("Extracting landmarks...")
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)
            
            print(f"Number of frames: {len(frame_list)}")
            
            # Process each frame with ONNX
            print("Processing frames with ONNX...")
            input_latent_list = []
            for bbox, frame in tqdm(zip(coord_list, frame_list), desc="Encoding frames"):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                
                # Use ONNX VAE encoder
                latents = onnx_engine.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
            
            # Create cycle for smooth inference
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            
            print("Starting ONNX inference...")
            
            # Create dummy audio features for demonstration
            video_num = len(input_latent_list_cycle)
            dummy_audio_features = create_dummy_audio_features(video_num)
            
            batch_size = args.batch_size
            gen = datagen(
                audio_features=dummy_audio_features,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
            )
            
            res_frame_list = []
            total = int(np.ceil(float(video_num) / batch_size))
            
            # Execute ONNX inference
            for i, (audio_batch, latent_batch) in enumerate(tqdm(gen, total=total, desc="ONNX Inference")):
                try:
                    # Run ONNX inference
                    latents_pred = onnx_engine.process_batch(latent_batch, audio_batch)
                    
                    # Decode latents to images
                    for j in range(latents_pred.shape[0]):
                        single_latent = np.expand_dims(latents_pred[j], 0)
                        pred_img = onnx_engine.decode_latents(single_latent)
                        res_frame_list.append(pred_img)
                        
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    # Use original frame as fallback
                    for j in range(latent_batch.shape[0]):
                        idx = i * batch_size + j
                        if idx < len(frame_list_cycle):
                            res_frame_list.append(frame_list_cycle[idx])
            
            # Blend and save results
            print("Blending and saving results...")
            for i, (res_frame, coord, frame) in enumerate(zip(res_frame_list[:len(frame_list)], 
                                                              coord_list, frame_list)):
                if coord == coord_placeholder:
                    result = frame
                else:
                    x1, y1, x2, y2 = coord
                    if args.version == "v15":
                        y2 = y2 + args.extra_margin
                        y2 = min(y2, frame.shape[0])
                    
                    # Resize result to match crop size
                    crop_size = (x2 - x1, y2 - y1)
                    res_frame_resized = cv2.resize(res_frame, crop_size, interpolation=cv2.INTER_LANCZOS4)
                    
                    # Blend with original frame
                    result = frame.copy()
                    result[y1:y2, x1:x2] = res_frame_resized
                
                # Save frame
                cv2.imwrite(os.path.join(result_img_save_path, f"{i:08d}.png"), result)
            
            # Generate video
            print("Generating final video...")
            temp_video = os.path.join(temp_dir, f"temp_{output_basename}.mp4")
            cmd = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_video}"
            os.system(cmd)
            
            # Combine with audio
            cmd = f"ffmpeg -y -v warning -i {audio_path} -i {temp_video} {output_vid_name}"
            os.system(cmd)
            
            print(f"Results saved to {output_vid_name}")
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified ONNX-based MuseTalk Inference")
    
    # Model paths
    parser.add_argument("--onnx_model_dir", default="./models/onnx", 
                       help="Directory containing ONNX models")
    parser.add_argument("--version", choices=["v1", "v15"], default="v15",
                       help="MuseTalk version")
    
    # Inference config
    parser.add_argument("--inference_config", default="./configs/inference/test.yaml",
                       help="Path to inference configuration")
    parser.add_argument("--result_dir", default="./results/onnx_simple_test",
                       help="Directory to save results")
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--fps", type=int, default=25,
                       help="FPS for output video")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Use GPU for ONNX inference")
    
    # Face processing
    parser.add_argument("--bbox_shift", type=int, default=0,
                       help="Bounding box shift")
    parser.add_argument("--extra_margin", type=int, default=30,
                       help="Extra margin for v1.5")
    parser.add_argument("--left_cheek_width", type=int, default=30,
                       help="Left cheek width for v1.5")
    parser.add_argument("--right_cheek_width", type=int, default=30,
                       help="Right cheek width for v1.5")
    parser.add_argument("--use_saved_coord", action="store_true", default=True,
                       help="Use saved coordinates")
    
    # System
    parser.add_argument("--ffmpeg_path", default="",
                       help="Path to FFmpeg")
    parser.add_argument("--output_vid_name", default=None,
                       help="Output video name")
    
    args = parser.parse_args()
    
    main(args) 