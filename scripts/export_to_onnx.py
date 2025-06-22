#!/usr/bin/env python3
"""
ONNX Export Script for MuseTalk Models
This script exports the UNet, VAE, and PositionalEncoding models to ONNX format.
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import json
from pathlib import Path
import shutil

# INT8 quantization support
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    from onnxruntime.quantization import quantize_static, CalibrationMethod
    INT8_AVAILABLE = True
    print("✓ INT8 quantization support available")
except ImportError:
    INT8_AVAILABLE = False
    print("⚠️ INT8 quantization not available. Install with: pip install onnxruntime")

# More aggressive disabling of attention optimizations
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# Try to disable scaled_dot_product_attention globally
import torch.nn.functional as F
original_sdpa = F.scaled_dot_product_attention

def disabled_sdpa(*args, **kwargs):
    """Fallback to manual attention computation"""
    query, key, value = args[:3]
    
    # Manual attention computation
    scale = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    if 'attn_mask' in kwargs and kwargs['attn_mask'] is not None:
        attn_weights += kwargs['attn_mask']
        
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    if 'dropout_p' in kwargs and kwargs['dropout_p'] > 0.0:
        attn_weights = F.dropout(attn_weights, p=kwargs['dropout_p'])
    
    output = torch.matmul(attn_weights, value)
    return output

# Patch the function
F.scaled_dot_product_attention = disabled_sdpa

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.utils import load_all_model
from musetalk.models.unet import PositionalEncoding
from transformers import WhisperModel
from musetalk.utils.face_parsing import FaceParsing

def export_unet_to_onnx(unet, output_path, device='cpu', opset_version=18):
    """Export UNet model to ONNX format with external data support"""
    print(f"Exporting UNet to {output_path} with opset {device}")
    
    # Convert path to string to handle both string and Path objects
    output_path = str(output_path)
    
    # Ensure device is CPU for export to avoid CUDA issues
    device = torch.device('cpu')
    
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet_model):
            super().__init__()
            # UNet class is a wrapper around UNet2DConditionModel
            # Access the actual model
            self.unet = unet_model.model
            
            # Disable attention optimizations for ONNX export
            # try:
            #     # Disable flash attention and memory efficient attention
            #     self.unet.set_attention_slice(None)
            #     if hasattr(self.unet, 'set_use_memory_efficient_attention_xformers'):
            #         self.unet.set_use_memory_efficient_attention_xformers(False)
            #     if hasattr(self.unet, 'set_attn_processor'):
            #         from diffusers.models.attention_processor import AttnProcessor
            #         self.unet.set_attn_processor(AttnProcessor())
            # except Exception as e:
            #     print(f"Warning: Could not disable UNet attention optimizations: {e}")
            
        def forward(self, input_latents, timesteps, audio_prompts):
            # Call the UNet model directly
            result = self.unet(
                sample=input_latents,
                timestep=timesteps,
                encoder_hidden_states=audio_prompts,
                return_dict=False
            )
            return result[0]  # Return just the noise prediction
    
    # Ensure model and inputs are on CPU to avoid CUDA issues
    wrapper = UNetWrapper(unet).to(device)
    wrapper.eval()
    
    # Create dummy inputs matching the expected format (always on CPU for export)
    input_latents = torch.randn(1, 8, 32, 32, device=device)  # Concatenated latents
    timesteps = torch.tensor([0], device=device, dtype=torch.long)
    audio_prompts = torch.randn(1, 50, 384, device=device)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # For very large models, we'll use a different approach
    if output_path.endswith('.onnx'):
        temp_path = output_path[:-5] + '_temp.onnx'
    else:
        temp_path = output_path + '_temp.onnx'
    
    print("Attempting to export large UNet model with external data support...")
    
    try:
        # First, try the export
        with torch.no_grad():
            # Use BytesIO to capture the model in memory first
            import io
            f = io.BytesIO()
            
            torch.onnx.export(
                wrapper,
                (input_latents, timesteps, audio_prompts),
                f,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_latents', 'timesteps', 'audio_prompts'],
                output_names=['noise_prediction'],
                # dynamic_axes={
                #     'input_latents': {0: 'batch_size'},
                #     'timesteps': {0: 'batch_size'},
                #     'audio_prompts': {0: 'batch_size', 1: 'sequence_length'},
                #     'noise_prediction': {0: 'batch_size'}
                # },
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL
            )
            
            # Load the model from memory and save with external data
            f.seek(0)
            import onnx
            model = onnx.load(f)
            
            # Save with external data format
            onnx.save_model(
                model, 
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{os.path.basename(output_path)}.data",
                size_threshold=1024  # Save all tensors > 1KB externally
            )
            print(f"UNet saved with external data to {output_path}")
            print(f"External data file: {os.path.dirname(output_path)}/{os.path.basename(output_path)}.data")
            
    except Exception as e:
        print(f"Memory-based export failed: {e}")
        print("Trying direct file export with external data format...")
        
        try:
            # Direct export to file
            torch.onnx.export(
                wrapper,
                (input_latents, timesteps, audio_prompts),
                temp_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_latents', 'timesteps', 'audio_prompts'],
                output_names=['noise_prediction'],
                # dynamic_axes={
                #     'input_latents': {0: 'batch_size'},
                #     'timesteps': {0: 'batch_size'},
                #     'audio_prompts': {0: 'batch_size', 1: 'sequence_length'},
                #     'noise_prediction': {0: 'batch_size'}
                # },
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL
            )
            
            # Load and convert to external data format
            import onnx
            model = onnx.load(temp_path)
            onnx.save_model(
                model, 
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{os.path.basename(output_path)}.data",
                size_threshold=1024
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            print(f"UNet exported successfully with external data")
            
        except Exception as e2:
            print(f"All export methods failed: {e2}")
            print("UNet model is too large for current ONNX export capabilities")
            return None
    
    print(f"UNet exported successfully to {output_path}")
    return wrapper

def export_vae_encoder_to_onnx(vae_model, output_path, device="cpu", opset_version=18):
    """Export VAE encoder to ONNX"""
    print(f"Exporting VAE Encoder to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    
    # Set model to evaluation mode
    vae_model.vae.eval()
    
    # Create dummy input with dynamic dimensions
    batch_size = 1
    channels = 3
    height, width = 256, 256  # Use larger base size for better compatibility
    
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    
    # Create a wrapper for the encoder
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, x):
            latent_dist = self.vae.encode(x).latent_dist
            # Use mode() for deterministic results (matching PyTorch VAE.encode_latents)
            # instead of sample() which adds stochastic noise
            return latent_dist.mode() * self.vae.config.scaling_factor
    
    encoder_wrapper = VAEEncoderWrapper(vae_model.vae).to(device)

    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['latents'],
        # No dynamic_axes --> static shape model
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    
    # Export to ONNX with dynamic axes for height and width
    # torch.onnx.export(
    #     encoder_wrapper,
    #     dummy_input,
    #     output_path,
    #     export_params=True,
    #     opset_version=opset_version,
    #     do_constant_folding=True,
    #     input_names=['image'],
    #     output_names=['latents'],
    #     dynamic_axes={
    #         'image': {0: 'batch_size', 2: 'height', 3: 'width'},
    #         'latents': {0: 'batch_size', 2: 'latent_height', 3: 'latent_width'}
    #     },us
    #     verbose=True,
    #     training=torch.onnx.TrainingMode.EVAL
    # )
    
    print(f"VAE Encoder exported successfully to {output_path}")
    return True

def export_vae_decoder_to_onnx(vae_model, output_path, device="cpu", opset_version=18):
    """Export VAE decoder to ONNX"""
    print(f"Exporting VAE Decoder to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    
    # Set model to evaluation mode
    vae_model.vae.eval()
    
    # Create dummy input with dynamic dimensions
    batch_size = 1
    latent_channels = 4
    latent_height, latent_width = 32, 32  # Use larger base size for better compatibility
    
    dummy_latents = torch.randn(batch_size, latent_channels, latent_height, latent_width).to(device)
    
    # Create a wrapper for the decoder
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, latents):
            latents = latents / self.vae.config.scaling_factor
            return self.vae.decode(latents).sample
    
    decoder_wrapper = VAEDecoderWrapper(vae_model.vae).to(device)
    
    # Export to ONNX with dynamic axes for latent dimensions
    torch.onnx.export(
        decoder_wrapper,
        dummy_latents,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['latents'],
        output_names=['image'],
        # dynamic_axes={
        #     'latents': {0: 'batch_size', 2: 'latent_height', 3: 'latent_width'},
        #     'image': {0: 'batch_size', 2: 'height', 3: 'width'}
        # },
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    
    print(f"VAE Decoder exported successfully to {output_path}")
    return True

def export_positional_encoding_to_onnx(pe_model, output_path, device="cpu", opset_version=18):
    """Export Positional Encoding to ONNX"""
    print(f"Exporting Positional Encoding to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)

    pe_model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 50
    d_model = 384
    
    dummy_input = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        pe_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_features'],
        output_names=['audio_features_with_pe'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'seq_len'},
            'audio_features_with_pe': {0: 'batch_size', 1: 'seq_len'}
        },
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    
    print(f"Positional Encoding exported successfully to {output_path}")
    return True

def export_whisper_to_onnx(whisper_model, output_path, device="cpu", opset_version=18):
    """Export Whisper encoder to ONNX with ALL hidden states stacked (fixed for Python compatibility)"""
    print(f"Exporting Whisper to {output_path}")
    
    whisper_model = whisper_model.to(device).eval()
    
    # Standard Whisper input: [batch_size, 80, 3000] for mel spectrogram
    batch_size = 1
    feature_dim = 80
    seq_len = 3000
    
    # Create dummy input matching Whisper's expected format
    dummy_input = torch.randn(batch_size, feature_dim, seq_len).to(device)
    
    # Create a wrapper for whisper encoder
    class WhisperEncoderWrapper(torch.nn.Module):
        def __init__(self, whisper):
            super().__init__()
            self.whisper = whisper
            
        def forward(self, input_features):
            outputs = self.whisper.encoder(input_features, output_hidden_states=True)
            # Stack ALL hidden states like Python implementation
            # This is the key fix - Python uses torch.stack(outputs.hidden_states, dim=2)
            hidden_states = torch.stack(outputs.hidden_states, dim=2)  # [batch, seq_len, layers, features]
            return hidden_states
    
    whisper_wrapper = WhisperEncoderWrapper(whisper_model).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        whisper_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['audio_features_all_layers'],  # More descriptive name
        dynamic_axes={
            'input_features': {0: 'batch_size', 2: 'seq_len'},
            'audio_features_all_layers': {0: 'batch_size', 1: 'seq_len', 2: 'num_layers'}  # Updated for stacked layers
        },
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    
    print(f"Whisper exported successfully to {output_path}")
    return True

def export_face_parsing_to_onnx(fp_model, output_path, device="cpu", opset_version=18):
    """Export BiSeNet face parsing model to ONNX format with fixed input size"""
    print(f"Exporting Face Parsing (BiSeNet) to {output_path}")
    
    # Access the actual BiSeNet model from FaceParsing wrapper
    bisenet_model = fp_model.net.to(device).eval()
    
    # Face parsing expects 512x512 RGB input (normalized)
    batch_size = 1
    channels = 3
    height = 512
    width = 512
    
    # Create dummy input (already normalized like FaceParsing preprocessing)
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    
    # Create an ONNX-friendly wrapper that fixes the interpolation issue
    class ONNXFriendlyFaceParsingWrapper(torch.nn.Module):
        def __init__(self, bisenet_model):
            super().__init__()
            self.bisenet = bisenet_model
            
        def forward(self, x):
            # BiSeNet forward pass but with fixed input size to avoid dynamic shape issues
            # Assuming input is always 512x512 for face parsing
            H, W = 512, 512  # Fixed input size
            
            # Get features from context path  
            feat_res8, feat_cp8, feat_cp16 = self.bisenet.cp(x)
            
            # Use res3b1 feature to replace spatial path feature
            feat_sp = feat_res8
            
            # Feature fusion
            feat_fuse = self.bisenet.ffm(feat_sp, feat_cp8)
            
            # Get outputs
            feat_out = self.bisenet.conv_out(feat_fuse)
            feat_out16 = self.bisenet.conv_out16(feat_cp8)
            feat_out32 = self.bisenet.conv_out32(feat_cp16)
            
            # Fixed interpolation to 512x512 (since input is always 512x512)
            import torch.nn.functional as F
            feat_out = F.interpolate(feat_out, size=(512, 512), mode='bilinear', align_corners=True)
            feat_out16 = F.interpolate(feat_out16, size=(512, 512), mode='bilinear', align_corners=True)
            feat_out32 = F.interpolate(feat_out32, size=(512, 512), mode='bilinear', align_corners=True)
            
            # Return only the main output for face parsing
            return feat_out  # [batch, 19, 512, 512] - 19 face parsing classes
    
    fp_wrapper = ONNXFriendlyFaceParsingWrapper(bisenet_model).to(device)
    
    # Export to ONNX with more stable options
    try:
        # Use TorchScript tracing first for better ONNX compatibility
        with torch.no_grad():
            traced_model = torch.jit.trace(fp_wrapper, dummy_input)
        
        torch.onnx.export(
            traced_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=False,  # Disable to avoid dynamic shape issues
            input_names=['image'],
            output_names=['face_parsing_output'],
            # Remove dynamic axes since we're fixing the input size
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )
        
        print(f"Face Parsing exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to export Face Parsing model with tracing, trying direct export: {e}")
        
        # Fallback to direct export without tracing
        try:
            fp_wrapper = ONNXFriendlyFaceParsingWrapper(bisenet_model).to(device)
            
            torch.onnx.export(
                fp_wrapper,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,  # Use older opset for better compatibility
                do_constant_folding=False,
                input_names=['image'],
                output_names=['face_parsing_output'],
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL
            )
            
            print(f"Face Parsing exported successfully to {output_path} (fallback mode)")
            return True
            
        except Exception as e2:
            print(f"Failed to export Face Parsing model (both methods): {e2}")
            return False

def verify_onnx_model(onnx_path, input_shapes=None):
    """Verify the exported ONNX model"""
    print(f"Verifying ONNX model: {onnx_path}")
    
    try:
        # For large models with external data, skip the onnx.checker and go straight to runtime
        file_size = os.path.getsize(onnx_path)
        is_large_model = file_size > 100 * 1024 * 1024  # > 100MB likely has external data
        
        if is_large_model:
            print(f"Large model detected ({file_size / 1024 / 1024:.1f}MB), skipping protobuf check...")
        else:
            # Load and check the model (only for smaller models)
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session to verify it can load
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
            
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"✓ ONNX model {onnx_path} is valid")
        print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Output names: {[out.name for out in session.get_outputs()]}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "2GiB limit" in error_msg or "too large" in error_msg:
            print(f"Model has external data (>2GB), trying runtime verification only...")
            try:
                # Try to create session without full model checking
                providers = ['CPUExecutionProvider']
                session = ort.InferenceSession(onnx_path, providers=providers)
                print(f"✓ ONNX model {onnx_path} is valid (runtime verification)")
                print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")
                print(f"  Output names: {[out.name for out in session.get_outputs()]}")
                return True
            except Exception as e2:
                print(f"✗ Runtime verification also failed: {e2}")
                return False
        else:
            print(f"✗ ONNX model verification failed: {e}")
            return False

def convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type="general"):
    """Convert FP32 ONNX model to INT8 using static quantization with QDQ format (Mac-compatible)"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False
        
    try:
        print(f"Converting {fp32_model_path} to INT8 using static QDQ quantization (Mac-compatible)...")
        
        # Import static quantization functions
        from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat
        from onnxruntime.quantization.calibrate import CalibrationDataReader
        
        # Create a simple calibration data reader
        class DummyCalibrationDataReader(CalibrationDataReader):
            def __init__(self, model_path):
                self.model_path = model_path
                self.data_generated = False
                
                # Load model to get input shapes and types
                import onnx
                model = onnx.load(model_path)
                self.input_names = [inp.name for inp in model.graph.input]
                self.input_shapes = {}
                self.input_types = {}
                
                for inp in model.graph.input:
                    # Get shape
                    shape = []
                    for dim in inp.type.tensor_type.shape.dim:
                        if dim.dim_value > 0:
                            shape.append(dim.dim_value)
                        else:
                            # Use realistic defaults for dynamic dimensions based on input name
                            if inp.name == 'image':
                                # VAE Encoder image input: use typical 256x256 RGB
                                if len(shape) == 0:  # batch dimension
                                    shape.append(1)
                                elif len(shape) == 1:  # channels
                                    shape.append(3)
                                elif len(shape) == 2:  # height
                                    shape.append(256)
                                elif len(shape) == 3:  # width
                                    shape.append(256)
                                else:
                                    shape.append(1)
                            elif inp.name == 'latents':
                                # VAE Decoder latent input: use typical 32x32
                                if len(shape) == 0:  # batch dimension
                                    shape.append(1)
                                elif len(shape) == 1:  # channels
                                    shape.append(4)
                                elif len(shape) == 2:  # height
                                    shape.append(32)
                                elif len(shape) == 3:  # width
                                    shape.append(32)
                                else:
                                    shape.append(1)
                            elif 'latent' in inp.name.lower():
                                # UNet latent input: use 32x32
                                if len(shape) >= 2:
                                    shape.append(32)
                                else:
                                    shape.append(1)
                            elif 'audio' in inp.name.lower():
                                # Audio input: use realistic audio dimensions
                                if len(shape) == 1:  # sequence length
                                    shape.append(50)
                                elif len(shape) == 2:  # features
                                    shape.append(384)
                                else:
                                    shape.append(1)
                            else:
                                shape.append(1)  # Default fallback
                    self.input_shapes[inp.name] = shape
                    
                    # Get data type
                    elem_type = inp.type.tensor_type.elem_type
                    if elem_type == onnx.TensorProto.FLOAT:
                        self.input_types[inp.name] = np.float32
                    elif elem_type == onnx.TensorProto.INT64:
                        self.input_types[inp.name] = np.int64
                    elif elem_type == onnx.TensorProto.INT32:
                        self.input_types[inp.name] = np.int32
                    else:
                        # Default to float32 for unknown types
                        self.input_types[inp.name] = np.float32
                        print(f"Warning: Unknown data type {elem_type} for input {inp.name}, using float32")
                
            def get_next(self):
                if not self.data_generated:
                    self.data_generated = True
                    # Generate dummy calibration data with proper types
                    calibration_data = {}
                    for name, shape in self.input_shapes.items():
                        dtype = self.input_types[name]
                        if dtype == np.int64 or dtype == np.int32:
                            # For integer types, generate small positive values
                            calibration_data[name] = np.zeros(shape, dtype=dtype)
                        else:
                            # For float types, generate random data
                            calibration_data[name] = np.random.randn(*shape).astype(dtype)
                    
                    print(f"Generated calibration data with types: {[(name, self.input_types[name]) for name in self.input_names]}")
                    return calibration_data
                else:
                    return None
        
        # Create calibration data reader
        calibration_reader = DummyCalibrationDataReader(fp32_model_path)
        
        # Determine if we need external data format (conservative threshold for quantized models)
        import os
        model_size = os.path.getsize(fp32_model_path)
        # Use much higher threshold since quantized models are typically 3-4x smaller
        # Only really large models (like UNet ~130MB+ FP32) should use external data
        use_external_data =model_type == "unet" or (model_size > 1024 * 1024 * 200)  # > 200MB threshold
        
        if use_external_data:
            print(f"Large model detected ({model_size / 1024 / 1024:.1f}MB), using external data format")
        else:
            print(f"Small model ({model_size / 1024 / 1024:.1f}MB), using embedded format")
        
        # Use static quantization with QDQ format (avoids ConvInteger)
        quantize_static(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,  # Use QDQ format instead of QOperator
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            use_external_data_format=use_external_data,  # Only for large models
            calibrate_method=CalibrationMethod.MinMax
        )
        
        print(f"✓ INT8 QDQ model saved to {int8_model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to INT8 QDQ: {e}")
        return False

def convert_model_to_int8_alternative(fp32_model_path, int8_model_path, model_type="general"):
    """Convert FP32 ONNX model to INT8 using QDQ format (avoids ConvInteger operations)"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False
        
    try:
        # First try static QDQ quantization (most Mac-compatible)
        if convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type):
            return True
        
        print(f"Static QDQ quantization failed, trying minimal dynamic quantization...")
        
        # Fallback to minimal dynamic quantization
        model_size = os.path.getsize(fp32_model_path)
        use_external_data = model_size > 1024 * 1024 * 200  # > 200MB threshold (conservative for quantized models)
        
        quantize_dynamic(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            weight_type=QuantType.QInt8,
            use_external_data_format=use_external_data  # Only for large models
        )
        
        print(f"✓ INT8 model (minimal dynamic) saved to {int8_model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to INT8 (all methods): {e}")
        return False

def convert_model_to_int8(fp32_model_path, int8_model_path, model_type="general"):
    """Convert FP32 ONNX model to INT8 using best approach for Mac compatibility"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False
        
    try:
        print(f"Converting {fp32_model_path} to INT8 (Mac-compatible approach)...")
        
        # Choose quantization type based on model
        if model_type in ["unet", "vae_encoder", "vae_decoder"]:
            print(f"Using QInt8 quantization for {model_type}")
        else:
            print(f"Using conservative quantization for {model_type}")
        
        # Try approaches in order of Mac compatibility:
        # 1. Static QDQ quantization (most compatible)
        # 2. Alternative dynamic quantization
        success = convert_model_to_int8_alternative(fp32_model_path, int8_model_path, model_type)
        
        if success:
            print(f"✓ INT8 model saved to {int8_model_path}")
            return True
        else:
            print(f"✗ All quantization methods failed for {fp32_model_path}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to INT8: {e}")
        return False

def copy_to_streaming_assets(source_dir, model_suffix="_v15"):
    """Copy exported models to Unity StreamingAssets folder"""
    # Define StreamingAssets path
    unity_streaming_assets = Path("../MysteryAI/Assets/StreamingAssets/LiveTalk")
    
    if not unity_streaming_assets.exists():
        print(f"Creating StreamingAssets directory: {unity_streaming_assets}")
        unity_streaming_assets.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    copied_files = []
    
    # Models to copy (FP32 and INT8 versions, VAE decoder FP32 only for quality)
    models_to_copy = [
        f"unet{model_suffix}.onnx",
        f"unet{model_suffix}_int8.onnx",
        f"vae_encoder{model_suffix}.onnx",  # FP32 version
        f"vae_encoder{model_suffix}_int8.onnx",  # INT8 version for performance testing
        f"vae_decoder{model_suffix}.onnx",  # FP32 only for quality
        f"positional_encoding{model_suffix}.onnx",
        f"positional_encoding{model_suffix}_int8.onnx",
        "whisper_encoder.onnx",
        "whisper_encoder_int8.onnx",
        "face_parsing.onnx",
        "face_parsing_int8.onnx",
        f"onnx_config{model_suffix}.json"
    ]
    
    # Copy external data files for large models (only UNet needs external data now)
    external_data_files = [
        f"unet{model_suffix}.onnx.data",
        f"unet{model_suffix}_int8.onnx.data"
    ]
    
    for model_file in models_to_copy:
        source_file = source_path / model_file
        dest_file = unity_streaming_assets / model_file
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, dest_file)
                copied_files.append(model_file)
                print(f"✓ Copied {model_file} to StreamingAssets")
            except Exception as e:
                print(f"✗ Failed to copy {model_file}: {e}")
        else:
            print(f"⚠️ Model file not found: {source_file}")
    
    # Copy external data files for large models
    for data_file in external_data_files:
        source_file = source_path / data_file
        dest_file = unity_streaming_assets / data_file
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, dest_file)
                copied_files.append(data_file)
                print(f"✓ Copied external data {data_file} to StreamingAssets")
            except Exception as e:
                print(f"✗ Failed to copy external data {data_file}: {e}")
    
    print(f"\n✓ Successfully copied {len(copied_files)} files to StreamingAssets")
    return copied_files

def export_model_with_quantization(export_func, model, output_path, model_name, export_int8=True, device="cpu", opset_version=18, **kwargs):
    """Export model in FP32 and INT8 formats"""
    # Convert path to string and create variant paths
    fp32_path = str(output_path)
    int8_path = fp32_path.replace('.onnx', '_int8.onnx')
    
    success_count = 0
    
    # Extract model type for INT8 quantization
    model_type = model_name.lower().replace(" ", "_")
    
    # Skip INT8 only for VAE decoder (most sensitive to quality loss)
    # VAE encoder can handle INT8 better, so we'll export both versions
    if model_type == "vae_decoder":
        print(f"✓ Exporting both FP32 and INT8 for {model_name} (runtime will choose best)")
        # export_int8 = False
    elif model_type == "vae_encoder":
        print(f"✓ Exporting both FP32 and INT8 for {model_name} (runtime will choose best)")
        # export_int8 remains True
    
    # Export FP32 model
    try:
        print(f"\n=== Exporting {model_name} (FP32) ===")
        if export_func(model, fp32_path, device, opset_version, **kwargs):
            if verify_onnx_model(fp32_path):
                success_count += 1
                print(f"✓ {model_name} FP32 export successful")
                
                # Convert to INT8 if requested (CPU-optimized)
                if export_int8:
                    if convert_model_to_int8(fp32_path, int8_path, model_type):
                        if verify_onnx_model(int8_path):
                            success_count += 1
                            print(f"✓ {model_name} INT8 quantization successful")
                        else:
                            print(f"✗ {model_name} INT8 model verification failed")
                    else:
                        print(f"✗ {model_name} INT8 quantization failed")
                else:
                    print(f"⚠️ Skipping INT8 quantization for {model_name}")
            else:
                print(f"✗ {model_name} FP32 model verification failed")
        else:
            print(f"✗ {model_name} FP32 export failed")
    except Exception as e:
        print(f"✗ Failed to export {model_name}: {e}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Export MuseTalk models to ONNX")
    parser.add_argument("--version", choices=["v1.0", "v1.5"], default="v1.5", 
                       help="MuseTalk version to export")
    parser.add_argument("--output_dir", default="./models/onnx", 
                       help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", 
                       help="Device to use for export (cpu/cuda)")
    parser.add_argument("--models", nargs="+", 
                       choices=["unet", "vae_encoder", "vae_decoder", "pe", "whisper", "face_parsing", "all"],
                       default=["all"], help="Models to export")
    parser.add_argument("--opset_version", type=int, default=18,
                       help="ONNX opset version to use (11-18)")
    parser.add_argument("--int8", action="store_true", default=True,
                       help="Export INT8 quantized models (CPU-optimized, default: True)")
    parser.add_argument("--no-int8", action="store_true", 
                       help="Disable INT8 quantization")
    parser.add_argument("--copy-to-unity", action="store_true", default=True,
                       help="Copy exported models to Unity StreamingAssets (default: True)")
    parser.add_argument("--no-copy-unity", action="store_true", 
                       help="Disable copying to Unity StreamingAssets")
    
    args = parser.parse_args()
    
    # Handle quantization and Unity copy flags
    export_int8 = args.int8 and not args.no_int8 and INT8_AVAILABLE
    copy_to_unity = args.copy_to_unity and not args.no_copy_unity
    
    if args.int8 and not INT8_AVAILABLE:
        print("⚠️ INT8 quantization requested but onnxruntime quantization not available")
        export_int8 = False
    
    # Recommend INT8 for CPU-only setups
    if export_int8:
        print("🍎 Using INT8 quantization - optimal for CPU inference (especially on Mac)")
    else:
        print("📝 Exporting FP32 models only")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Using ONNX opset version: {args.opset_version}")
    
    # Load models based on version
    if args.version == "v1.0":
        unet_model_path = "./models/musetalk/pytorch_model.bin"
        unet_config_path = "./models/musetalk/musetalk.json"
        model_suffix = "_v1"
    else:  # v1.5
        unet_model_path = "./models/musetalkV15/unet.pth"
        unet_config_path = "./models/musetalkV15/musetalk.json"
        model_suffix = "_v15"
    
    models_to_export = args.models
    if "all" in models_to_export:
        models_to_export = ["unet", "vae_encoder", "vae_decoder", "pe", "whisper", "face_parsing"]
    
    try:
        # Load MuseTalk models
        print("Loading MuseTalk models...")
        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type="sd-vae",
            unet_config=unet_config_path,
            device=device
        )
        
        # Load Whisper model
        whisper_dir = "./models/whisper"
        if os.path.exists(whisper_dir):
            print("Loading Whisper model...")
            whisper = WhisperModel.from_pretrained(whisper_dir)
            whisper = whisper.to(device).eval()
        else:
            print("Whisper model not found, skipping...")
            if "whisper" in models_to_export:
                models_to_export.remove("whisper")
        
        # Load Face Parsing model
        if "face_parsing" in models_to_export:
            try:
                print("Loading Face Parsing (BiSeNet) model...")
                fp = FaceParsing()
                print("Face Parsing model loaded successfully")
            except Exception as e:
                print(f"Face Parsing model not found or failed to load: {e}")
                models_to_export.remove("face_parsing")
        
        # Export models
        success_count = 0
        
        if "unet" in models_to_export:
            unet_path = output_dir / f"unet{model_suffix}.onnx"
            try:
                success_count += export_model_with_quantization(export_unet_to_onnx, unet, unet_path, "UNet", export_int8, device, args.opset_version)
            except Exception as e:
                print(f"Failed to export UNet: {e}")
        
        if "vae_encoder" in models_to_export:
            vae_encoder_path = output_dir / f"vae_encoder{model_suffix}.onnx"
            try:
                success_count += export_model_with_quantization(export_vae_encoder_to_onnx, vae, vae_encoder_path, "VAE Encoder", export_int8, device, args.opset_version)
            except Exception as e:
                print(f"Failed to export VAE Encoder: {e}")
        
        if "vae_decoder" in models_to_export:
            vae_decoder_path = output_dir / f"vae_decoder{model_suffix}.onnx"
            try:
                success_count += export_model_with_quantization(export_vae_decoder_to_onnx, vae, vae_decoder_path, "VAE Decoder", export_int8, device, args.opset_version)
            except Exception as e:
                print(f"Failed to export VAE Decoder: {e}")
        
        if "pe" in models_to_export:
            pe_path = output_dir / f"positional_encoding{model_suffix}.onnx"
            try:
                success_count += export_model_with_quantization(export_positional_encoding_to_onnx, pe, pe_path, "Positional Encoding", export_int8, device, args.opset_version)
            except Exception as e:
                print(f"Failed to export Positional Encoding: {e}")
        
        if "whisper" in models_to_export and 'whisper' in locals():
            whisper_path = output_dir / "whisper_encoder.onnx"
            try:
                success_count += export_model_with_quantization(export_whisper_to_onnx, whisper, whisper_path, "Whisper", export_int8, device, args.opset_version)
            except Exception as e:
                print(f"Failed to export Whisper: {e}")
        
        if "face_parsing" in models_to_export and 'fp' in locals():
            face_parsing_path = output_dir / "face_parsing.onnx"
            try:
                success_count += export_model_with_quantization(export_face_parsing_to_onnx, fp, face_parsing_path, "Face Parsing", export_int8, device, args.opset_version)
            except Exception as e:
                print(f"Failed to export Face Parsing: {e}")
        
        print(f"\n✓ Successfully exported {success_count} models to ONNX format")
        print(f"Output directory: {output_dir}")
        
        # Save model configuration
        config = {
            "version": args.version,
            "model_suffix": model_suffix,
            "exported_models": models_to_export,
            "device": str(device),
            "opset_version": args.opset_version,
            "int8_exported": export_int8,
            "int8_available": INT8_AVAILABLE,
            "copied_to_unity": copy_to_unity,
            "success_count": success_count
        }
        
        config_path = output_dir / f"onnx_config{model_suffix}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
        
        # Copy models to StreamingAssets
        if copy_to_unity:
            copied_files = copy_to_streaming_assets(output_dir, model_suffix)
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 