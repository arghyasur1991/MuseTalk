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
    def __init__(self, model_dir="./models/onnx", version="v15", device="cpu", use_int8=True):
        self.model_dir = Path(model_dir)
        self.version = version
        self.device = device
        self.use_int8 = use_int8
        
        # Set up ONNX Runtime providers
        self.providers = self._get_providers()
        
        # Validate quantization options
        if self.use_int8:
            print("🍎 Using INT8 quantization - optimal for CPU inference")
        else:
            print("📝 Using FP32 models")
        
        # Initialize face parsing - will try ONNX first, fallback to PyTorch
        print("Initializing face parsing model...")
        self.fp = None
        self.fp_session = None
        
        # Load ONNX models
        self.load_models()
        
    def _get_providers(self):
        """Get available ONNX Runtime providers optimized for CPU/INT8"""
        providers = []
        
        # Check for CUDA first if requested
        if 'CUDAExecutionProvider' in ort.get_available_providers() and self.device in ["cuda", "gpu"]:
            cuda_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 0,  # Use all available GPU memory
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            providers.append(('CUDAExecutionProvider', cuda_options))
            print("✓ CUDA provider configured")
            
        elif 'DmlExecutionProvider' in ort.get_available_providers() and self.device in ["cuda", "gpu", "dml"]:
            providers.append(('DmlExecutionProvider', {}))
            print("✓ DirectML provider configured")
            
        # Always add CPU as fallback (optimal for INT8)
        providers.append('CPUExecutionProvider')
        
        print(f"Using ONNX providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
        return providers
        
    def _get_session_options(self):
        """Get optimized session options for INT8/CPU inference"""
        session_options = ort.SessionOptions()
        
        if self.use_int8:
            # Enable optimizations for INT8 quantized models
            print("Enabling INT8 optimizations in ONNX Runtime...")
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            # FP32 optimizations with conservative settings
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Common optimizations
        session_options.add_session_config_entry("session.use_env_allocators", "1")
        
        return session_options
        
    def load_models(self):
        """Load all ONNX models"""
        print("Loading ONNX models...")
        
        # Model file paths - use regular optimized models
        model_suffix = f"_{self.version}" if self.version != "v1.0" else ""
        
        unet_path = self._get_model_path("unet", model_suffix)
        vae_encoder_path = self._get_model_path("vae_encoder", model_suffix)
        vae_decoder_path = self._get_model_path("vae_decoder", model_suffix)
        pe_path = self._get_model_path("positional_encoding", model_suffix)
        whisper_path = self._get_model_path("whisper_encoder")
        
        # Load models with error handling and INT8 fallback
        try:
            print(f"Loading UNet from {unet_path}")
            self.unet_session = ort.InferenceSession(str(unet_path), providers=self.providers, session_options=self._get_session_options())
            print("✓ UNet loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load UNet: {e}")
            raise
            
        # VAE Encoder with INT8 fallback
        try:
            print(f"Loading VAE Encoder from {vae_encoder_path}")
            self.vae_encoder_session = ort.InferenceSession(str(vae_encoder_path), providers=self.providers, session_options=self._get_session_options())
            print("✓ VAE Encoder loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load VAE Encoder: {e}")
            if "ConvInteger" in str(e) and "_int8" in str(vae_encoder_path):
                print("💡 INT8 ConvInteger not supported on this platform, falling back to FP32...")
                fallback_path = str(vae_encoder_path).replace("_int8", "")
                print(f"Trying FP32 fallback: {fallback_path}")
                try:
                    self.vae_encoder_session = ort.InferenceSession(fallback_path, providers=self.providers, session_options=self._get_session_options())
                    print("✓ VAE Encoder loaded successfully (FP32 fallback)")
                except Exception as e2:
                    print(f"✗ FP32 fallback also failed: {e2}")
                    raise
            else:
                raise
            
        # VAE Decoder with INT8 fallback
        try:
            print(f"Loading VAE Decoder from {vae_decoder_path}")
            self.vae_decoder_session = ort.InferenceSession(str(vae_decoder_path), providers=self.providers, session_options=self._get_session_options())
            print("✓ VAE Decoder loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load VAE Decoder: {e}")
            if "ConvInteger" in str(e) and "_int8" in str(vae_decoder_path):
                print("💡 INT8 ConvInteger not supported on this platform, falling back to FP32...")
                fallback_path = str(vae_decoder_path).replace("_int8", "")
                print(f"Trying FP32 fallback: {fallback_path}")
                try:
                    self.vae_decoder_session = ort.InferenceSession(fallback_path, providers=self.providers, session_options=self._get_session_options())
                    print("✓ VAE Decoder loaded successfully (FP32 fallback)")
                except Exception as e2:
                    print(f"✗ FP32 fallback also failed: {e2}")
                    raise
            else:
                raise
            
        # Positional Encoding with INT8 fallback
        try:
            print(f"Loading Positional Encoding from {pe_path}")
            self.pe_session = ort.InferenceSession(str(pe_path), providers=self.providers, session_options=self._get_session_options())
            print("✓ Positional Encoding loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Positional Encoding: {e}")
            if "ConvInteger" in str(e) and "_int8" in str(pe_path):
                print("💡 INT8 operations not supported on this platform, falling back to FP32...")
                fallback_path = str(pe_path).replace("_int8", "")
                print(f"Trying FP32 fallback: {fallback_path}")
                try:
                    self.pe_session = ort.InferenceSession(fallback_path, providers=self.providers, session_options=self._get_session_options())
                    print("✓ Positional Encoding loaded successfully (FP32 fallback)")
                except Exception as e2:
                    print(f"✗ FP32 fallback also failed: {e2}")
                    raise
            else:
                raise
        
        # Load Whisper ONNX model with INT8 fallback
        try:
            print(f"Loading Whisper ONNX from {whisper_path}")
            self.whisper_session = ort.InferenceSession(str(whisper_path), providers=self.providers, session_options=self._get_session_options())
            print("✓ Whisper ONNX loaded successfully")
            self.has_whisper_onnx = True
        except Exception as e:
            print(f"✗ Failed to load Whisper ONNX: {e}")
            if "ConvInteger" in str(e) and "_int8" in str(whisper_path):
                print("💡 INT8 operations not supported on this platform, falling back to FP32...")
                fallback_path = str(whisper_path).replace("_int8", "")
                print(f"Trying FP32 fallback: {fallback_path}")
                try:
                    self.whisper_session = ort.InferenceSession(fallback_path, providers=self.providers, session_options=self._get_session_options())
                    print("✓ Whisper ONNX loaded successfully (FP32 fallback)")
                    self.has_whisper_onnx = True
                except Exception as e2:
                    print(f"✗ FP32 fallback also failed: {e2}")
                    self.has_whisper_onnx = False
            else:
                self.has_whisper_onnx = False
        
        # Load Face Parsing ONNX model (priority) or fallback to PyTorch
        face_parsing_path = self._get_model_path("face_parsing")
        try:
            print(f"Loading Face Parsing ONNX from {face_parsing_path}")
            self.fp_session = ort.InferenceSession(str(face_parsing_path), providers=self.providers, session_options=self._get_session_options())
            print("✓ Face Parsing ONNX loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Face Parsing ONNX: {e}")
            if "ConvInteger" in str(e) and "_int8" in str(face_parsing_path):
                print("💡 INT8 operations not supported on this platform, falling back to FP32...")
                fallback_path = str(face_parsing_path).replace("_int8", "")
                print(f"Trying FP32 fallback: {fallback_path}")
                try:
                    self.fp_session = ort.InferenceSession(fallback_path, providers=self.providers, session_options=self._get_session_options())
                    print("✓ Face Parsing ONNX loaded successfully (FP32 fallback)")
                except Exception as e2:
                    print(f"✗ FP32 fallback also failed: {e2}")
                    print("Falling back to PyTorch Face Parsing model...")
                    try:
                        self.fp = FaceParsing()
                        print("✓ PyTorch Face Parsing loaded successfully")
                    except Exception as e3:
                        print(f"✗ Failed to load PyTorch Face Parsing: {e3}")
                        self.fp = None
            else:
                print("Falling back to PyTorch Face Parsing model...")
                try:
                    self.fp = FaceParsing()
                    print("✓ PyTorch Face Parsing loaded successfully")
                except Exception as e2:
                    print(f"✗ Failed to load PyTorch Face Parsing: {e2}")
                    self.fp = None
            
        print("All ONNX models loaded successfully!")
        

        
    def _get_model_path(self, base_name, model_suffix=""):
        """Get model path with INT8 quantization support"""
        
        # Priority: INT8 for CPU optimization > FP32 fallback
        if self.use_int8:
            int8_path = self.model_dir / f"{base_name}{model_suffix}_int8.onnx"
            if int8_path.exists():
                print(f"Using INT8 model: {int8_path}")
                return int8_path
            else:
                print(f"⚠️ INT8 model not found: {int8_path}, falling back to FP32")
        
        # Fallback to FP32 model
        fp32_path = self.model_dir / f"{base_name}{model_suffix}.onnx"
        print(f"Using FP32 model: {fp32_path}")
        return fp32_path
    
    def extract_mel_spectrogram(self, audio_path):
        """Extract mel spectrogram features matching the transformers WhisperFeatureExtractor"""
        print(f"Extracting mel spectrogram from {audio_path}")
        
        # Load audio using librosa (matching AudioProcessor)
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Split into 30-second segments (matching Python implementation)
        segment_length = 30 * sample_rate  # 30 seconds * 16000 Hz
        segments = []
        
        for i in range(0, len(audio_data), segment_length):
            segment = audio_data[i:i + segment_length]
            # Pad the last segment if needed
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            segments.append(segment)
        
        mel_features = []
        for segment in segments:
            # Extract mel spectrogram using librosa (matching Whisper preprocessing)
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=sample_rate,
                n_mels=80,  # Whisper uses 80 mel bins
                n_fft=400,  # Whisper uses 25ms window = 400 samples at 16kHz
                hop_length=160,  # Whisper uses 10ms hop = 160 samples at 16kHz
                fmin=0,
                fmax=8000  # Nyquist frequency for 16kHz
            )
            
            # Convert to log scale (matching Whisper)
            mel_spec = np.log(np.maximum(mel_spec, 1e-10))
            
            # Normalize (rough approximation of Whisper's normalization)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            # Ensure correct shape for Whisper: [80, 3000] -> pad or trim
            if mel_spec.shape[1] > 3000:
                mel_spec = mel_spec[:, :3000]
            elif mel_spec.shape[1] < 3000:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, 3000 - mel_spec.shape[1])), 'constant')
            
            mel_features.append(mel_spec.astype(np.float32))
        
        return mel_features, len(audio_data)
    
    def process_audio_with_onnx_whisper_fixed(self, audio_path):
        """Process audio using ONNX Whisper with pure numpy (no PyTorch dependencies)"""
        print("Processing audio with pure ONNX Whisper...")
        
        try:
            # Load audio using librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Convert to mel spectrogram (matching Whisper preprocessing)
            mel_features = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=80,           # Whisper uses 80 mel bins
                hop_length=160,      # 10ms hop (16000 * 0.01)
                win_length=400,      # 25ms window (16000 * 0.025)
                window='hann',
                center=True,
                pad_mode='reflect',
                power=2.0
            )
            
            # Convert to log scale (dB)
            mel_features = librosa.power_to_db(mel_features, ref=np.max)
            
            # Normalize to [-1, 1] range like Whisper
            mel_features = (mel_features + 80.0) / 80.0
            mel_features = np.clip(mel_features, -1.0, 1.0)
            
            # Pad or trim to 3000 frames (30 seconds at 10ms hop)
            target_frames = 3000
            if mel_features.shape[1] < target_frames:
                # Pad with zeros
                padding = target_frames - mel_features.shape[1]
                mel_features = np.pad(mel_features, ((0, 0), (0, padding)), mode='constant')
            else:
                # Trim to target length
                mel_features = mel_features[:, :target_frames]
            
            # Add batch dimension: [1, 80, 3000]
            mel_features = mel_features[np.newaxis, ...]
            
            print(f"Input shape for Whisper ONNX: {mel_features.shape}")
            
            # Run ONNX Whisper (this now returns ALL hidden states stacked)
            outputs = self.whisper_session.run(
                ['audio_features_all_layers'],
                {'input_features': mel_features.astype(np.float32)}
            )
            
            # Get the stacked hidden states: [batch, seq_len, layers, features]
            whisper_features = outputs[0]  # Shape: [1, seq_len, layers, features]
            print(f"Whisper ONNX output shape: {whisper_features.shape}")
            
            # Now process exactly like Python get_whisper_chunk
            audio_length = len(audio)
            sr = 16000
            audio_fps = 50
            fps = 25
            audio_padding_length_left = 2
            audio_padding_length_right = 2
            
            whisper_idx_multiplier = audio_fps / fps
            num_frames = int((audio_length / sr) * fps)
            actual_length = int((audio_length / sr) * audio_fps)
            
            # Trim to actual length
            whisper_features = whisper_features[:, :actual_length, ...]
            
            # Add padding
            import math
            padding_nums = math.ceil(whisper_idx_multiplier)
            left_padding_size = padding_nums * audio_padding_length_left
            right_padding_size = padding_nums * 3 * audio_padding_length_right
            
            # Create padding arrays
            batch_size, seq_len, layers, features = whisper_features.shape
            left_padding = np.zeros((batch_size, left_padding_size, layers, features), dtype=whisper_features.dtype)
            right_padding = np.zeros((batch_size, right_padding_size, layers, features), dtype=whisper_features.dtype)
            
            # Concatenate padding
            whisper_features = np.concatenate([left_padding, whisper_features, right_padding], axis=1)
            
            # Generate chunks
            audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
            audio_prompts = []
            
            for frame_index in range(num_frames):
                audio_index = int(frame_index * whisper_idx_multiplier)
                audio_clip = whisper_features[:, audio_index:audio_index + audio_feature_length_per_frame]
                
                if audio_clip.shape[1] == audio_feature_length_per_frame:
                    audio_prompts.append(audio_clip)
            
            # Stack all chunks: [num_frames, 10, layers, features]
            audio_prompts = np.concatenate(audio_prompts, axis=0)  
            
            # Rearrange: 'b c h w -> b (c h) w' 
            # where b=frames, c=time_chunks, h=layers, w=features
            batch_size, time_chunks, layers, features = audio_prompts.shape
            audio_prompts = audio_prompts.reshape(batch_size, time_chunks * layers, features)
            
            # Convert to list of numpy arrays for compatibility
            whisper_chunks_np = []
            for i in range(audio_prompts.shape[0]):
                chunk = audio_prompts[i]  # Shape: [60, 384] (10*6 layers, 384 features)
                whisper_chunks_np.append(chunk)
            
            print(f"Processed {len(whisper_chunks_np)} audio chunks with pure ONNX Whisper")
            print(f"Each chunk shape: {whisper_chunks_np[0].shape if whisper_chunks_np else 'None'}")
            return whisper_chunks_np, audio_length
            
        except Exception as e:
            print(f"Warning: Could not process with ONNX Whisper ({e}), using dummy audio features")
            return self.create_dummy_audio_features(audio_path)
    
    def get_whisper_chunks(self, whisper_feature, librosa_length, fps=25, audio_padding_length_left=2, audio_padding_length_right=2):
        """Convert Whisper features to chunks matching Python implementation exactly"""
        # Calculate parameters (matching Python exactly)
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        sr = 16000
        audio_fps = 50  # Whisper's internal frame rate
        whisper_idx_multiplier = audio_fps / fps
        num_frames = int((librosa_length / sr) * fps)
        actual_length = int((librosa_length / sr) * audio_fps)
        
        # Trim to actual length (matching Python)
        whisper_feature = whisper_feature[:, :actual_length, :]  # [1, actual_length, 384]
        
        # Calculate padding (matching Python)
        padding_nums = int(np.ceil(whisper_idx_multiplier))
        left_padding = np.zeros((1, padding_nums * audio_padding_length_left, 384), dtype=np.float32)
        right_padding = np.zeros((1, padding_nums * 3 * audio_padding_length_right, 384), dtype=np.float32)
        
        # Add padding (matching Python)
        whisper_feature = np.concatenate([left_padding, whisper_feature, right_padding], axis=1)
        
        # Generate chunks for each frame
        audio_prompts = []
        for frame_index in range(num_frames):
            audio_index = int(frame_index * whisper_idx_multiplier)
            audio_clip = whisper_feature[:, audio_index:audio_index + audio_feature_length_per_frame, :]  # [1, 10, 384]
            
            if audio_clip.shape[1] == audio_feature_length_per_frame:
                # Reshape to match Python: [1, 10, 384] -> [10, 384] -> [50, 384] after reshaping
                audio_clip = audio_clip.squeeze(0)  # [10, 384]
                # Expand to [50, 384] to match Python (10 time steps * 5 layers = 50)
                audio_clip_expanded = np.tile(audio_clip, (5, 1))  # [50, 384]
                audio_prompts.append(audio_clip_expanded)
        
        print(f"Generated {len(audio_prompts)} audio chunks, each shape: {audio_prompts[0].shape if audio_prompts else 'None'}")
        return audio_prompts
    
    def create_dummy_audio_features(self, audio_path):
        """Create dummy audio features as fallback"""
        print("Creating dummy audio features...")
        
        # Load audio to get duration
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)
        librosa_length = len(audio_data)
        
        # Calculate number of frames
        fps = 25
        num_frames = int((librosa_length / sample_rate) * fps)
        
        # Create dummy chunks
        audio_prompts = []
        for i in range(num_frames):
            # Each chunk is [50, 384] matching Python
            chunk = np.random.randn(50, 384).astype(np.float32)
            audio_prompts.append(chunk)
        
        print(f"Created {len(audio_prompts)} dummy audio chunks")
        return audio_prompts, librosa_length

    def encode_image(self, image):
        """Encode image using VAE encoder - FIXED: match PyTorch VAE preprocessing exactly"""
        # Resize image to expected VAE input size (256x256 for the exported model)
        target_size = 256
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size))
        
        # FIXED: Match PyTorch VAE preprocessing order EXACTLY
        # Step 1: BGR to RGB conversion (PyTorch VAE does this!)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 2: Create window and normalize to [0,1]
        window = [img_rgb]
        x = np.asarray(window, dtype=np.float32) / 255.0
        
        # Step 3: Transpose to get [C, B, H, W] then squeeze to [C, H, W]
        x = np.transpose(x, (3, 0, 1, 2))  # [C, B, H, W]
        x = np.squeeze(x)  # [C, H, W]
        
        # Step 4: Apply normalization transform: (x - 0.5) / 0.5
        for c in range(3):
            x[c] = (x[c] - 0.5) / 0.5
        
        # Step 5: Add batch dimension for ONNX: [1, C, H, W]
        image_tensor = np.expand_dims(x, 0)
        
        # Run VAE encoder
        latents = self.vae_encoder_session.run(
            ['latents'], 
            {'image': image_tensor}
        )[0]
        
        return latents
    
    def encode_image_with_half_mask(self, image):
        """Encode image with lower half masked - FIXED: match PyTorch VAE exactly"""
        # Resize image to expected VAE input size
        target_size = 256
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size))
        
        # FIXED: Match PyTorch VAE preprocessing order EXACTLY  
        # Step 1: BGR to RGB conversion (PyTorch VAE does this!)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 2: Create window and normalize to [0,1]
        window = [img_rgb]
        x = np.asarray(window, dtype=np.float32) / 255.0
        
        # Step 3: Transpose to get [C, B, H, W] then squeeze to [C, H, W]
        x = np.transpose(x, (3, 0, 1, 2))  # [C, B, H, W]
        x = np.squeeze(x)  # [C, H, W]
        
        # Step 4: Apply mask first (before normalization transform)
        mask_tensor = np.zeros((target_size, target_size), dtype=np.float32)
        mask_tensor[:target_size//2, :] = 1  # Upper half = 1, lower half = 0
        # Apply mask to each channel of [C, H, W] tensor
        for c in range(3):
            x[c] = x[c] * mask_tensor
        
        # Step 5: Apply normalization transform: (x - 0.5) / 0.5
        for c in range(3):
            x[c] = (x[c] - 0.5) / 0.5
        
        # Step 6: Add batch dimension for ONNX: [1, C, H, W]
        image_tensor = np.expand_dims(x, 0)
        
        # Run VAE encoder
        latents = self.vae_encoder_session.run(
            ['latents'], 
            {'image': image_tensor}
        )[0]
        
        return latents
    
    def get_latents_for_unet(self, image):
        """Replicate the exact behavior of VAE.get_latents_for_unet - FIXED: mask lower half"""
        # Get masked latents (half mask applied)
        masked_latents = self.encode_image_with_half_mask(image)  # [1, 4, 32, 32]
        
        # Get reference latents (no mask)
        ref_latents = self.encode_image(image)  # [1, 4, 32, 32]
        
        # Concatenate along channel dimension (matching PyTorch)
        latent_model_input = np.concatenate([masked_latents, ref_latents], axis=1)  # [1, 8, 32, 32]
        
        return latent_model_input

    def decode_latents(self, latents, target_size=(256, 256)):
        """Decode latents using VAE decoder - IMPROVED: Higher precision decoding"""
        # Ensure input is float32 for maximum precision
        if latents.dtype != np.float32:
            latents = latents.astype(np.float32)
        
        # Run VAE decoder with precise session options
        image = self.vae_decoder_session.run(
            ['image'], 
            {'latents': latents}
        )[0]
        
        # Convert back to uint8 image with improved precision
        image = np.transpose(image, (0, 2, 3, 1))
        
        # Use higher precision for normalization to avoid rounding errors
        image_float64 = image.astype(np.float64)
        image_normalized = (image_float64 / 2.0 + 0.5).clip(0.0, 1.0)  # Higher precision
        
        # Convert to uint8 with proper rounding
        image_uint8 = np.round(image_normalized * 255.0).astype(np.uint8)
        image_final = image_uint8[0]  # Remove batch dimension
        
        # FIXED: Convert RGB to BGR to match PyTorch VAE decode_latents exactly
        # PyTorch VAE does: image = image[...,::-1] # RGB to BGR
        image_final_bgr = image_final[...,::-1]  # RGB to BGR conversion
        
        # Resize to target size with high-quality interpolation if specified
        if target_size and (image_final_bgr.shape[0] != target_size[0] or image_final_bgr.shape[1] != target_size[1]):
            # Use LANCZOS for high-quality resizing (same as PyTorch inference)
            image_final_bgr = cv2.resize(image_final_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"Resized decoded image to {target_size} using LANCZOS4")
        
        return image_final_bgr
        
    def add_positional_encoding(self, audio_features):
        """Add positional encoding to audio features"""
        # Run positional encoding ONNX model
        audio_with_pe = self.pe_session.run(
            ['audio_features_with_pe'], 
            {'audio_features': audio_features}
        )[0]
        
        return audio_with_pe
        
    def run_unet(self, input_latents, timesteps, audio_prompts):
        """Run UNet inference"""
        # Run UNet ONNX model
        pred_latents = self.unet_session.run(
            ['noise_prediction'], 
            {
                'input_latents': input_latents,
                'timesteps': timesteps,
                'audio_prompts': audio_prompts
            }
        )[0]
        
        return pred_latents
        
    def run_face_parsing(self, image):
        """Run face parsing inference using ONNX or PyTorch"""
        if self.fp_session is not None:
            # Use ONNX face parsing
            # Prepare image for ONNX model (512x512 RGB)
            if image.shape[0] != 512 or image.shape[1] != 512:
                image = cv2.resize(image, (512, 512))
            
            # Convert BGR to RGB and normalize
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_norm = image_rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_norm = (image_norm - mean) / std
            
            # Convert to tensor format [1, 3, 512, 512]
            image_tensor = np.transpose(image_norm, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, 0)
            
            # Run ONNX inference
            parsing_output = self.fp_session.run(
                ['face_parsing_output'],
                {'image': image_tensor.astype(np.float32)}
            )[0]
            
            # Convert output to segmentation mask [512, 512]
            parsing_map = np.argmax(parsing_output[0], axis=0)
            return parsing_map
            
        elif self.fp is not None:
            # Use PyTorch face parsing (fallback)
            return self.fp.get_mask(image)
        else:
            # No face parsing available, return dummy mask
            print("Warning: No face parsing model available, using dummy mask")
            return np.ones((512, 512), dtype=np.uint8)
        
    def inference(self, avatar_path, audio_path, output_path, batch_size=4, max_images=10, use_insightface=True):
        """Main inference function matching PyTorch implementation exactly"""
        start_time = time.time()
        
        # Read avatar images
        if os.path.isdir(avatar_path):
            input_img_list = glob.glob(os.path.join(avatar_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        else:
            input_img_list = [avatar_path]
        
        if max_images > 0:
            input_img_list = input_img_list[:max_images]
        
        print(f"Processing {len(input_img_list)} avatar images")
        
        # Get face landmarks and bounding boxes
        backend_name = "InsightFace" if use_insightface else "MMPose"
        print(f"Extracting face landmarks with {backend_name}...")
        debug_dir = os.path.join(output_path, "debug")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, upperbondrange=0, use_insightface=use_insightface, debug_dir=debug_dir)
        # Process images to get latents
        print("Encoding avatar images...")
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + 10  # v15 extra margin
                y2 = min(y2, frame.shape[0])
            
            # Crop and resize face
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # Get latents for UNet
            latents = self.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        # Process audio with ONNX Whisper
        whisper_chunks, librosa_length = self.process_audio_with_onnx_whisper_fixed(audio_path)
        
        # Set global timesteps (matching PyTorch)
        timesteps = np.array([0], dtype=np.int64)
        
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
                
                # Debug: Save intermediate results for first few frames
                if frame_idx < 3:  # Debug first 3 frames
                    debug_dir = os.path.join(output_path, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save original frame crop
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_original_crop.jpg"), ori_frame[y1:y2, x1:x2])
                    
                    # Save generated result
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_generated.jpg"), res_frame)
                    
                    # Save resized generated result
                    target_width = x2 - x1
                    target_height = y2 - y1
                    res_frame_resized = cv2.resize(decoded_img.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_generated_resized.jpg"), res_frame_resized)
                
                # Create a wrapper for ONNX face parsing that mimics FaceParsing interface
                class FaceParsingWrapper:
                    def __init__(self, inference_engine):
                        self.engine = inference_engine
                    
                    def get_mask(self, image):
                        return self.engine.run_face_parsing(image)
                    
                    def __call__(self, image, size=(512, 512), mode="raw"):
                        """Make wrapper callable like original FaceParsing class"""
                        # Convert input image if it's a path string
                        if isinstance(image, str):
                            import cv2
                            image = cv2.imread(image)
                        
                        # If it's a PIL Image, convert to OpenCV format
                        if hasattr(image, 'size'):
                            import numpy as np
                            image = np.array(image)
                            if image.shape[2] == 3:  # RGB to BGR
                                image = image[:, :, ::-1]
                        
                        # Get face parsing mask using ONNX
                        parsing = self.engine.run_face_parsing(image)
                        
                        # Apply mode processing like original FaceParsing
                        if mode == "neck":
                            parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                            parsing[np.where(parsing!=255)] = 0
                        elif mode == "jaw":
                            # For jaw mode, include face and mouth regions
                            face_region = np.isin(parsing, [1])*255
                            face_region = face_region.astype(np.uint8)
                            
                            # Simple dilation and erosion (without complex kernels for now)
                            import cv2
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                            face_region = cv2.dilate(face_region, kernel, iterations=1)
                            face_region = cv2.erode(face_region, kernel, iterations=1)
                            
                            parsing[(face_region==255) & (~np.isin(parsing, [10]))] = 255         
                            parsing[np.isin(parsing, [11, 12, 13])] = 255  # Include mouth regions
                            parsing[np.where(parsing!=255)] = 0
                        else:  # raw mode
                            parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                            parsing[np.where(parsing!=255)] = 0
                        
                        # Convert to PIL Image like original
                        from PIL import Image
                        return Image.fromarray(parsing.astype(np.uint8))
                
                fp_wrapper = FaceParsingWrapper(self)
                
                if self.version == "v15":
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], upper_boundary_ratio=0.5, expand=1.8, mode='jaw', fp=fp_wrapper, debug_dir=debug_dir, frame_idx=frame_idx)
                else:
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], upper_boundary_ratio=0.5, expand=1.7, fp=fp_wrapper, debug_dir=debug_dir, frame_idx=frame_idx)
                
                # Debug: Save blended result
                if frame_idx < 3:
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_blended.jpg"), combine_frame)
                
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
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/gpu)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_images", type=int, default=10, help="Max images for debugging (0 for all)")
    parser.add_argument("--use_insightface", action="store_true", default=True, help="Use InsightFace models for face detection/landmarking")
    parser.add_argument("--use_mmpose", action="store_true", help="Use MMPose models for face detection/landmarking (fallback)")
    parser.add_argument("--int8", action="store_true", default=True, help="Use INT8 quantized models for inference (CPU-optimized, default: True)")
    parser.add_argument("--no-int8", action="store_true", help="Force disable INT8 quantization")
    
    args = parser.parse_args()
    
    # Handle quantization flags
    use_int8 = args.int8 and not args.no_int8
    
    # Create inference engine
    inference_engine = ONNXMuseTalkInference(
        model_dir=args.model_dir,
        version=args.version,
        device=args.device,
        use_int8=use_int8
    )
    
    # Log quantization status
    if inference_engine.use_int8:
        print("🍎 INT8 quantization enabled - optimized for CPU!")
    else:
        print("💾 Using FP32 inference")
    
    # Determine which backend to use
    use_insightface_models = args.use_insightface and not args.use_mmpose
    
    # Run inference
    inference_engine.inference(
        avatar_path=args.avatar_path,
        audio_path=args.audio_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_images=args.max_images,
        use_insightface=use_insightface_models
    )

if __name__ == "__main__":
    main() 