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

# Disable scaled_dot_product_attention for ONNX export
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.utils import load_all_model
from musetalk.models.unet import PositionalEncoding
from transformers import WhisperModel

def export_unet_to_onnx(unet_model, output_path, device="cpu"):
    """Export UNet model to ONNX"""
    print(f"Exporting UNet to {output_path}")
    
    # Set model to evaluation mode
    unet_model.model.eval()
    
    # Create dummy inputs matching the expected input shapes
    batch_size = 1
    latent_channels = 8  # 4 (masked) + 4 (reference)
    latent_height, latent_width = 32, 32
    audio_seq_len = 50
    audio_dim = 384
    
    # Dummy inputs
    latent_input = torch.randn(batch_size, latent_channels, latent_height, latent_width).to(device)
    timesteps = torch.tensor([0]).to(device)
    audio_embedding = torch.randn(batch_size, audio_seq_len, audio_dim).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        unet_model.model,
        (latent_input, timesteps, audio_embedding),
        output_path,
        export_params=True,
        opset_version=11,  # Use lower opset version
        do_constant_folding=True,
        input_names=['latent_input', 'timesteps', 'audio_embedding'],
        output_names=['output'],
        dynamic_axes={
            'latent_input': {0: 'batch_size'},
            'timesteps': {0: 'batch_size'},
            'audio_embedding': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"UNet exported successfully to {output_path}")
    return True

def export_vae_encoder_to_onnx(vae_model, output_path, device="cpu"):
    """Export VAE encoder to ONNX"""
    print(f"Exporting VAE Encoder to {output_path}")
    
    # Set model to evaluation mode
    vae_model.vae.eval()
    
    # Disable flash attention for ONNX export
    try:
        # Disable attention slicing and set attention to basic mode
        vae_model.vae.set_attention_slice(None)
        if hasattr(vae_model.vae, 'set_use_memory_efficient_attention_xformers'):
            vae_model.vae.set_use_memory_efficient_attention_xformers(False)
    except:
        pass
    
    # Create dummy input
    batch_size = 1
    channels = 3
    height, width = 256, 256
    
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
    
    # Export to ONNX with lower opset version
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Use lower opset version
        do_constant_folding=True,
        input_names=['image'],
        output_names=['latents'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'latents': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"VAE Encoder exported successfully to {output_path}")
    return True

def export_vae_decoder_to_onnx(vae_model, output_path, device="cpu"):
    """Export VAE decoder to ONNX"""
    print(f"Exporting VAE Decoder to {output_path}")
    
    # Set model to evaluation mode
    vae_model.vae.eval()
    
    # Disable flash attention for ONNX export
    try:
        # Disable attention slicing and set attention to basic mode
        vae_model.vae.set_attention_slice(None)
        if hasattr(vae_model.vae, 'set_use_memory_efficient_attention_xformers'):
            vae_model.vae.set_use_memory_efficient_attention_xformers(False)
    except:
        pass
    
    # Create dummy input
    batch_size = 1
    latent_channels = 4
    latent_height, latent_width = 32, 32
    
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
    
    # Export to ONNX with lower opset version
    torch.onnx.export(
        decoder_wrapper,
        dummy_latents,
        output_path,
        export_params=True,
        opset_version=11,  # Use lower opset version
        do_constant_folding=True,
        input_names=['latents'],
        output_names=['image'],
        dynamic_axes={
            'latents': {0: 'batch_size'},
            'image': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"VAE Decoder exported successfully to {output_path}")
    return True

def export_positional_encoding_to_onnx(pe_model, output_path, device="cpu"):
    """Export Positional Encoding to ONNX"""
    print(f"Exporting Positional Encoding to {output_path}")
    
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
        opset_version=11,  # Use lower opset version
        do_constant_folding=True,
        input_names=['audio_features'],
        output_names=['audio_features_with_pe'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'seq_len'},
            'audio_features_with_pe': {0: 'batch_size', 1: 'seq_len'}
        },
        verbose=False
    )
    
    print(f"Positional Encoding exported successfully to {output_path}")
    return True

def export_whisper_to_onnx(whisper_model, output_path, device="cpu"):
    """Export Whisper model to ONNX"""
    print(f"Exporting Whisper to {output_path}")
    
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
        opset_version=11,  # Use lower opset version
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['audio_features'],
        dynamic_axes={
            'input_features': {0: 'batch_size', 2: 'seq_len'},
            'audio_features': {0: 'batch_size', 1: 'seq_len'}
        },
        verbose=False
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
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
            if export_unet_to_onnx(unet, unet_path, device):
                if verify_onnx_model(unet_path):
                    success_count += 1
        
        if "vae_encoder" in models_to_export:
            vae_encoder_path = output_dir / f"vae_encoder{model_suffix}.onnx"
            if export_vae_encoder_to_onnx(vae, vae_encoder_path, device):
                if verify_onnx_model(vae_encoder_path):
                    success_count += 1
        
        if "vae_decoder" in models_to_export:
            vae_decoder_path = output_dir / f"vae_decoder{model_suffix}.onnx"
            if export_vae_decoder_to_onnx(vae, vae_decoder_path, device):
                if verify_onnx_model(vae_decoder_path):
                    success_count += 1
        
        if "pe" in models_to_export:
            pe_path = output_dir / f"positional_encoding{model_suffix}.onnx"
            if export_positional_encoding_to_onnx(pe, pe_path, device):
                if verify_onnx_model(pe_path):
                    success_count += 1
        
        if "whisper" in models_to_export and 'whisper' in locals():
            whisper_path = output_dir / "whisper_encoder.onnx"
            if export_whisper_to_onnx(whisper, whisper_path, device):
                if verify_onnx_model(whisper_path):
                    success_count += 1
        
        print(f"\n✓ Successfully exported {success_count} models to ONNX format")
        print(f"Output directory: {output_dir}")
        
        # Save model configuration
        config = {
            "version": args.version,
            "model_suffix": model_suffix,
            "exported_models": models_to_export,
            "device": str(device),
            "opset_version": 11  # Updated to reflect the actual opset version used
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