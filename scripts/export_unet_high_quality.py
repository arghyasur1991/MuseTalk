#!/usr/bin/env python3
"""
High-Quality UNet Export Script for MuseTalk
This script exports UNet with improved precision settings to fix output differences
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

def export_unet_high_quality(unet, output_path, opset_version=18, device='cpu'):
    """Export UNet model with high quality settings to preserve precision"""
    print(f"Exporting High-Quality UNet to {output_path} with opset {opset_version}")
    
    # Convert path to string to handle both string and Path objects
    output_path = str(output_path)
    
    class UNetHighQualityWrapper(torch.nn.Module):
        def __init__(self, unet_model):
            super().__init__()
            # UNet class is a wrapper around UNet2DConditionModel
            # Access the actual model
            self.unet = unet_model.model
            
            # Force high precision mode
            self.unet = self.unet.float()  # Ensure float32
            
            # Disable attention optimizations for maximum precision
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
            # Ensure all inputs are float32 for maximum precision
            input_latents = input_latents.float()
            audio_prompts = audio_prompts.float()
            
            # Call the UNet model directly with precise computation
            with torch.no_grad():
                result = self.unet(
                    sample=input_latents,
                    timestep=timesteps,
                    encoder_hidden_states=audio_prompts,
                    return_dict=False
                )
            return result[0]  # Return just the noise prediction
    
    wrapper = UNetHighQualityWrapper(unet).to(device)
    wrapper.eval()
    
    # Create dummy inputs matching the expected format - use exact sizes from debug
    input_latents = torch.randn(1, 8, 32, 32, dtype=torch.float32, device=device)  # Concatenated latents
    timesteps = torch.tensor([0], device=device, dtype=torch.long)
    audio_prompts = torch.randn(1, 50, 384, dtype=torch.float32, device=device)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Test the wrapper first
    print("Testing UNet wrapper...")
    with torch.no_grad():
        test_output = wrapper(input_latents, timesteps, audio_prompts)
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output range: [{test_output.min():.6f}, {test_output.max():.6f}]")
    
    print("Exporting UNet with high-precision settings...")
    
    try:
        # Export with high precision settings
        torch.onnx.export(
            wrapper,
            (input_latents, timesteps, audio_prompts),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=False,  # Disable constant folding to preserve precision
            input_names=['input_latents', 'timesteps', 'audio_prompts'],
            output_names=['noise_prediction'],
            dynamic_axes={
                'input_latents': {0: 'batch_size'},  # Only batch size dynamic
                'timesteps': {0: 'batch_size'},
                'audio_prompts': {0: 'batch_size'},
                'noise_prediction': {0: 'batch_size'}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False
        )
        
        print(f"High-Quality UNet exported successfully to {output_path}")
        return True
            
    except Exception as e:
        print(f"UNet export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_unet_quality(pytorch_unet, onnx_path, device='cpu'):
    """Compare PyTorch and ONNX UNet outputs for quality verification"""
    print(f"\n=== UNet Quality Verification ===")
    
    try:
        # Load ONNX model
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Create test inputs
        input_latents = torch.randn(1, 8, 32, 32, dtype=torch.float32, device=device)
        timesteps = torch.tensor([0], device=device, dtype=torch.long)
        audio_prompts = torch.randn(1, 50, 384, dtype=torch.float32, device=device)
        
        # PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_unet.model(
                sample=input_latents,
                timestep=timesteps,
                encoder_hidden_states=audio_prompts,
                return_dict=False
            )[0]
        
        # ONNX output
        onnx_output = session.run(
            ['noise_prediction'],
            {
                'input_latents': input_latents.numpy(),
                'timesteps': timesteps.numpy(),
                'audio_prompts': audio_prompts.numpy()
            }
        )[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output.numpy() - onnx_output)
        mae = np.mean(diff)
        max_diff = np.max(diff)
        correlation = np.corrcoef(pytorch_output.flatten().numpy(), onnx_output.flatten())[0, 1]
        
        print(f"PyTorch output range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        print(f"ONNX output range: [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")
        print(f"MAE: {mae:.6f}")
        print(f"Max Diff: {max_diff:.6f}")
        print(f"Correlation: {correlation:.6f}")
        
        if mae < 0.01:
            print("✅ EXCELLENT: UNet outputs are nearly identical")
            quality = "excellent"
        elif mae < 0.05:
            print("✅ GOOD: UNet outputs are very similar")
            quality = "good"
        elif mae < 0.1:
            print("⚠️ FAIR: UNet outputs have minor differences")
            quality = "fair"
        else:
            print("❌ POOR: UNet outputs differ significantly")
            quality = "poor"
            
        return quality, mae
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return "failed", None

def main():
    parser = argparse.ArgumentParser(description="Export high-quality UNet model")
    parser.add_argument("--version", choices=["v1.0", "v1.5"], default="v1.5", 
                       help="MuseTalk version to export")
    parser.add_argument("--output_dir", default="./models/onnx", 
                       help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", 
                       help="Device to use for export (cpu/cuda)")
    parser.add_argument("--opset_version", type=int, default=18,
                       help="ONNX opset version to use")
    parser.add_argument("--verify", action="store_true",
                       help="Verify quality against PyTorch model")
    
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
        # Load MuseTalk models
        print("Loading MuseTalk models...")
        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type="sd-vae",
            unet_config=unet_config_path,
            device=device
        )
        
        # Export high-quality UNet
        unet_path = output_dir / f"unet{model_suffix}_hq.onnx"
        
        print("\n=== Exporting High-Quality UNet ===")
        
        if export_unet_high_quality(unet, unet_path, args.opset_version, device):
            print(f"✅ High-quality UNet exported")
            
            # Verify quality if requested
            if args.verify:
                quality, mae = verify_unet_quality(unet, str(unet_path), device)
                print(f"\nQuality Assessment: {quality.upper()}")
                if mae is not None:
                    print(f"Mean Absolute Error: {mae:.6f}")
        else:
            print("❌ UNet export failed")
            return 1
        
        print(f"\n✅ High-quality UNet exported to: {output_dir}")
        print("Use this model in your ONNX inference for improved quality!")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 