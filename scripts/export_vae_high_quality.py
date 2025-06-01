#!/usr/bin/env python3
"""
High-Quality VAE Export Script for MuseTalk
This script exports VAE models with improved quality settings to fix blurriness issues
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
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

def export_vae_decoder_high_quality(vae_model, output_path, device="cpu", opset_version=18):
    """Export VAE decoder with high quality settings to reduce blurriness"""
    print(f"Exporting High-Quality VAE Decoder to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    
    # Set model to evaluation mode and disable optimizations that might cause blurriness
    vae_model.vae.eval()
    
    # Force high precision mode
    vae_model.vae = vae_model.vae.float()  # Ensure float32
    
    # Disable all attention optimizations that might reduce quality
    try:
        vae_model.vae.set_attention_slice(None)
        if hasattr(vae_model.vae, 'set_use_memory_efficient_attention_xformers'):
            vae_model.vae.set_use_memory_efficient_attention_xformers(False)
        if hasattr(vae_model.vae, 'set_attn_processor'):
            from diffusers.models.attention_processor import AttnProcessor
            vae_model.vae.set_attn_processor(AttnProcessor())
    except Exception as e:
        print(f"Warning: Could not disable VAE attention optimizations: {e}")
    
    # Create dummy input - use exact size matching inference (32x32 latents -> 256x256 image)
    batch_size = 1
    latent_channels = 4
    latent_height, latent_width = 32, 32  # Match inference exactly
    
    dummy_latents = torch.randn(batch_size, latent_channels, latent_height, latent_width, dtype=torch.float32).to(device)
    
    # Create a wrapper for the decoder with exact PyTorch behavior
    class VAEDecoderHighQualityWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, latents):
            # Exact same operations as PyTorch VAE
            latents = latents / self.vae.config.scaling_factor
            
            # Force no gradient computation
            with torch.no_grad():
                # Use the exact same decode path
                decoded = self.vae.decode(latents, return_dict=False)[0]
                
            return decoded
    
    decoder_wrapper = VAEDecoderHighQualityWrapper(vae_model.vae).to(device)
    
    # Test the wrapper to ensure it works correctly
    print("Testing wrapper...")
    with torch.no_grad():
        test_output = decoder_wrapper(dummy_latents)
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    
    # Export with higher precision settings
    print("Exporting with high-quality settings...")
    torch.onnx.export(
        decoder_wrapper,
        dummy_latents,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,  # Disable constant folding to preserve precision
        input_names=['latents'],
        output_names=['image'],
        dynamic_axes={
            'latents': {0: 'batch_size'},  # Remove dynamic height/width for better optimization
            'image': {0: 'batch_size'}
        },
        verbose=False,  # Disable verbose to avoid cluttering
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=False  # Optimize model size
    )
    
    print(f"High-Quality VAE Decoder exported successfully to {output_path}")
    return True

def export_vae_encoder_high_quality(vae_model, output_path, device="cpu", opset_version=18):
    """Export VAE encoder with high quality settings"""
    print(f"Exporting High-Quality VAE Encoder to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    
    # Set model to evaluation mode
    vae_model.vae.eval()
    vae_model.vae = vae_model.vae.float()  # Ensure float32
    
    # Disable flash attention for ONNX export
    try:
        vae_model.vae.set_attention_slice(None)
        if hasattr(vae_model.vae, 'set_use_memory_efficient_attention_xformers'):
            vae_model.vae.set_use_memory_efficient_attention_xformers(False)
        if hasattr(vae_model.vae, 'set_attn_processor'):
            from diffusers.models.attention_processor import AttnProcessor
            vae_model.vae.set_attn_processor(AttnProcessor())
    except Exception as e:
        print(f"Warning: Could not disable VAE attention optimizations: {e}")
    
    # Create dummy input - exact size matching inference (256x256)
    batch_size = 1
    channels = 3
    height, width = 256, 256  # Match inference exactly
    
    dummy_input = torch.randn(batch_size, channels, height, width, dtype=torch.float32).to(device)
    
    # Create a wrapper for the encoder with exact PyTorch behavior
    class VAEEncoderHighQualityWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, x):
            with torch.no_grad():
                latent_dist = self.vae.encode(x).latent_dist
                # Use mode() for deterministic results (matching PyTorch VAE.encode_latents)
                return latent_dist.mode() * self.vae.config.scaling_factor
    
    encoder_wrapper = VAEEncoderHighQualityWrapper(vae_model.vae).to(device)
    
    # Test the wrapper
    print("Testing encoder wrapper...")
    with torch.no_grad():
        test_output = encoder_wrapper(dummy_input)
        print(f"Test encoder output shape: {test_output.shape}")
        print(f"Test encoder output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    
    # Export with high precision settings
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,  # Preserve precision
        input_names=['image'],
        output_names=['latents'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'latents': {0: 'batch_size'}
        },
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=False
    )
    
    print(f"High-Quality VAE Encoder exported successfully to {output_path}")
    return True

def verify_quality_improvement(old_model_path, new_model_path):
    """Compare old and new VAE models for quality"""
    print(f"\n=== Quality Verification ===")
    print(f"Old model: {old_model_path}")
    print(f"New model: {new_model_path}")
    
    # Create test inputs
    test_latents = np.random.randn(1, 4, 32, 32).astype(np.float32)
    
    providers = ['CPUExecutionProvider']
    
    try:
        # Load old model
        old_session = ort.InferenceSession(old_model_path, providers=providers)
        old_output = old_session.run(['image'], {'latents': test_latents})[0]
        
        # Load new model  
        new_session = ort.InferenceSession(new_model_path, providers=providers)
        new_output = new_session.run(['image'], {'latents': test_latents})[0]
        
        # Calculate difference
        diff = np.abs(old_output - new_output)
        mae = np.mean(diff)
        
        print(f"Output shape: {new_output.shape}")
        print(f"Old output range: [{old_output.min():.3f}, {old_output.max():.3f}]")
        print(f"New output range: [{new_output.min():.3f}, {new_output.max():.3f}]") 
        print(f"MAE difference: {mae:.6f}")
        
        if mae < 0.001:
            print("✅ Models produce nearly identical outputs")
        elif mae < 0.01:
            print("⚠️ Models have minor differences")
        else:
            print("❌ Models have significant differences")
            
        return mae
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Export high-quality VAE models")
    parser.add_argument("--version", choices=["v1.0", "v1.5"], default="v1.5", 
                       help="MuseTalk version to export")
    parser.add_argument("--output_dir", default="./models/onnx", 
                       help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", 
                       help="Device to use for export (cpu/cuda)")
    parser.add_argument("--opset_version", type=int, default=18,
                       help="ONNX opset version to use")
    parser.add_argument("--compare_old", action="store_true",
                       help="Compare with old models for quality verification")
    
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
    
    try:
        # Load MuseTalk VAE model
        print("Loading MuseTalk models...")
        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type="sd-vae",
            unet_config=unet_config_path,
            device=device
        )
        
        # Export high-quality models
        encoder_path = output_dir / f"vae_encoder{model_suffix}_hq.onnx"
        decoder_path = output_dir / f"vae_decoder{model_suffix}_hq.onnx"
        
        print("\n=== Exporting High-Quality VAE Models ===")
        
        # Export encoder
        if export_vae_encoder_high_quality(vae, encoder_path, device, args.opset_version):
            print(f"✅ High-quality VAE encoder exported")
        
        # Export decoder
        if export_vae_decoder_high_quality(vae, decoder_path, device, args.opset_version):
            print(f"✅ High-quality VAE decoder exported")
        
        # Verify quality if requested
        if args.compare_old:
            old_decoder_path = output_dir / f"vae_decoder{model_suffix}.onnx"
            if old_decoder_path.exists():
                verify_quality_improvement(str(old_decoder_path), str(decoder_path))
            else:
                print("⚠️ Old model not found for comparison")
        
        print(f"\n✅ High-quality models exported to: {output_dir}")
        print("Use these models in your ONNX inference for improved quality!")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 