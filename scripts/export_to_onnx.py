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

def export_unet_to_onnx(unet, output_path, opset_version=18, device='cpu'):
    """Export UNet model to ONNX format with external data support"""
    print(f"Exporting UNet to {output_path} with opset {opset_version}")
    
    # Convert path to string to handle both string and Path objects
    output_path = str(output_path)
    
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet_model):
            super().__init__()
            # UNet class is a wrapper around UNet2DConditionModel
            # Access the actual model
            self.unet = unet_model.model
            
            # Disable attention optimizations for ONNX export
            try:
                # Disable flash attention and memory efficient attention
                self.unet.set_attention_slice(None)
                if hasattr(self.unet, 'set_use_memory_efficient_attention_xformers'):
                    self.unet.set_use_memory_efficient_attention_xformers(False)
                if hasattr(self.unet, 'set_attn_processor'):
                    from diffusers.models.attention_processor import AttnProcessor
                    self.unet.set_attn_processor(AttnProcessor())
            except Exception as e:
                print(f"Warning: Could not disable UNet attention optimizations: {e}")
            
        def forward(self, input_latents, timesteps, audio_prompts):
            # Call the UNet model directly
            result = self.unet(
                sample=input_latents,
                timestep=timesteps,
                encoder_hidden_states=audio_prompts,
                return_dict=False
            )
            return result[0]  # Return just the noise prediction
    
    wrapper = UNetWrapper(unet).to(device)
    wrapper.eval()
    
    # Create dummy inputs matching the expected format
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
                dynamic_axes={
                    'input_latents': {0: 'batch_size'},
                    'timesteps': {0: 'batch_size'},
                    'audio_prompts': {0: 'batch_size', 1: 'sequence_length'},
                    'noise_prediction': {0: 'batch_size'}
                },
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
                dynamic_axes={
                    'input_latents': {0: 'batch_size'},
                    'timesteps': {0: 'batch_size'},
                    'audio_prompts': {0: 'batch_size', 1: 'sequence_length'},
                    'noise_prediction': {0: 'batch_size'}
                },
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
    
    # Disable flash attention for ONNX export
    try:
        # Disable attention optimizations
        vae_model.vae.set_attention_slice(None)
        if hasattr(vae_model.vae, 'set_use_memory_efficient_attention_xformers'):
            vae_model.vae.set_use_memory_efficient_attention_xformers(False)
        if hasattr(vae_model.vae, 'set_attn_processor'):
            from diffusers.models.attention_processor import AttnProcessor
            vae_model.vae.set_attn_processor(AttnProcessor())
    except Exception as e:
        print(f"Warning: Could not disable VAE attention optimizations: {e}")
    
    # Create dummy input with dynamic dimensions
    batch_size = 1
    channels = 3
    height, width = 512, 512  # Use larger base size for better compatibility
    
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    
    # Create a wrapper for the encoder
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, x):
            latent_dist = self.vae.encode(x).latent_dist
            return latent_dist.sample() * self.vae.config.scaling_factor
    
    encoder_wrapper = VAEEncoderWrapper(vae_model.vae).to(device)
    
    # Export to ONNX with dynamic axes for height and width
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['latents'],
        dynamic_axes={
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'latents': {0: 'batch_size', 2: 'latent_height', 3: 'latent_width'}
        },
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    
    print(f"VAE Encoder exported successfully to {output_path}")
    return True

def export_vae_decoder_to_onnx(vae_model, output_path, device="cpu", opset_version=18):
    """Export VAE decoder to ONNX"""
    print(f"Exporting VAE Decoder to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    
    # Set model to evaluation mode
    vae_model.vae.eval()
    
    # Disable flash attention for ONNX export
    try:
        # Disable attention optimizations
        vae_model.vae.set_attention_slice(None)
        if hasattr(vae_model.vae, 'set_use_memory_efficient_attention_xformers'):
            vae_model.vae.set_use_memory_efficient_attention_xformers(False)
        if hasattr(vae_model.vae, 'set_attn_processor'):
            from diffusers.models.attention_processor import AttnProcessor
            vae_model.vae.set_attn_processor(AttnProcessor())
    except Exception as e:
        print(f"Warning: Could not disable VAE attention optimizations: {e}")
    
    # Create dummy input with dynamic dimensions
    batch_size = 1
    latent_channels = 4
    latent_height, latent_width = 64, 64  # Use larger base size for better compatibility
    
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
        dynamic_axes={
            'latents': {0: 'batch_size', 2: 'latent_height', 3: 'latent_width'},
            'image': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
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
    """Export Whisper model to ONNX"""
    print(f"Exporting Whisper to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)

    whisper_model.eval()
    
    # Create dummy input for audio features
    batch_size = 1
    seq_len = 3000  # Typical whisper input length
    feature_dim = 80   # Mel spectrogram features
    
    dummy_input = torch.randn(batch_size, feature_dim, seq_len).to(device)
    
    # Create a wrapper for whisper encoder
    class WhisperEncoderWrapper(torch.nn.Module):
        def __init__(self, whisper):
            super().__init__()
            self.whisper = whisper
            
        def forward(self, input_features):
            outputs = self.whisper.encoder(input_features, output_hidden_states=True)
            return outputs.hidden_states[-1]  # Use last hidden state
    
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
        output_names=['audio_features'],
        dynamic_axes={
            'input_features': {0: 'batch_size', 2: 'seq_len'},
            'audio_features': {0: 'batch_size', 1: 'seq_len'}
        },
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    
    print(f"Whisper exported successfully to {output_path}")
    return True

def verify_onnx_model(onnx_path, input_shapes=None):
    """Verify the exported ONNX model"""
    print(f"Verifying ONNX model: {onnx_path}")
    
    try:
        # Load and check the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
            
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"✓ ONNX model {onnx_path} is valid")
        print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Output names: {[out.name for out in session.get_outputs()]}")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Export MuseTalk models to ONNX")
    parser.add_argument("--version", choices=["v1.0", "v1.5"], default="v1.5", 
                       help="MuseTalk version to export")
    parser.add_argument("--output_dir", default="./models/onnx", 
                       help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", 
                       help="Device to use for export (cpu/cuda)")
    parser.add_argument("--models", nargs="+", 
                       choices=["unet", "vae_encoder", "vae_decoder", "pe", "whisper", "all"],
                       default=["all"], help="Models to export")
    parser.add_argument("--opset_version", type=int, default=18,
                       help="ONNX opset version to use (11-18)")
    
    args = parser.parse_args()
    
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
        models_to_export = ["unet", "vae_encoder", "vae_decoder", "pe", "whisper"]
    
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
        
        # Export models
        success_count = 0
        
        if "unet" in models_to_export:
            unet_path = output_dir / f"unet{model_suffix}.onnx"
            try:
                if export_unet_to_onnx(unet, unet_path, args.opset_version, device):
                    if verify_onnx_model(unet_path):
                        success_count += 1
            except Exception as e:
                print(f"Failed to export UNet: {e}")
        
        if "vae_encoder" in models_to_export:
            vae_encoder_path = output_dir / f"vae_encoder{model_suffix}.onnx"
            try:
                if export_vae_encoder_to_onnx(vae, vae_encoder_path, device, args.opset_version):
                    if verify_onnx_model(vae_encoder_path):
                        success_count += 1
            except Exception as e:
                print(f"Failed to export VAE Encoder: {e}")
        
        if "vae_decoder" in models_to_export:
            vae_decoder_path = output_dir / f"vae_decoder{model_suffix}.onnx"
            try:
                if export_vae_decoder_to_onnx(vae, vae_decoder_path, device, args.opset_version):
                    if verify_onnx_model(vae_decoder_path):
                        success_count += 1
            except Exception as e:
                print(f"Failed to export VAE Decoder: {e}")
        
        if "pe" in models_to_export:
            pe_path = output_dir / f"positional_encoding{model_suffix}.onnx"
            try:
                if export_positional_encoding_to_onnx(pe, pe_path, device, args.opset_version):
                    if verify_onnx_model(pe_path):
                        success_count += 1
            except Exception as e:
                print(f"Failed to export Positional Encoding: {e}")
        
        if "whisper" in models_to_export and 'whisper' in locals():
            whisper_path = output_dir / "whisper_encoder.onnx"
            try:
                if export_whisper_to_onnx(whisper, whisper_path, device, args.opset_version):
                    if verify_onnx_model(whisper_path):
                        success_count += 1
            except Exception as e:
                print(f"Failed to export Whisper: {e}")
        
        print(f"\n✓ Successfully exported {success_count} models to ONNX format")
        print(f"Output directory: {output_dir}")
        
        # Save model configuration
        config = {
            "version": args.version,
            "model_suffix": model_suffix,
            "exported_models": models_to_export,
            "device": str(device),
            "opset_version": args.opset_version
        }
        
        config_path = output_dir / f"onnx_config{model_suffix}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 