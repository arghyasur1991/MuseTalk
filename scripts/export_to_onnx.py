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
from onnxsim import simplify
from onnx import numpy_helper

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

from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model

@torch.no_grad()
def tune_model(
    model_path: str,
    model_type: str,
    fp16: bool
):
    model_dir=os.path.dirname(model_path)
    
    # First we set our optimisation to the ORT Optimizer defaults for the provided type
    optimization_options = FusionOptions(model_type)
    # The ORT optimizer is designed for ORT GPU and CUDA
    # To make things work with ORT DirectML, we disable some options
    # The GroupNorm op has a very negative effect on VRAM and CPU use
    optimization_options.enable_group_norm = False
    # On by default in ORT optimizer, turned off as it causes performance issues
    optimization_options.enable_nhwc_conv = False
    # On by default in ORT optimizer, turned off because it has no effect
    optimization_options.enable_qordered_matmul = False
    optimization_options.enable_bias_splitgelu = False
    optimization_options.enable_bias_add = False
    optimization_options.enable_skip_layer_norm = model_type != "unet"
    optimization_options.enable_gelu = model_type != "unet"
    optimizer = optimize_model(
        input = model_path,
        model_type = model_type,
        opt_level = 0,
        optimization_options = optimization_options,
        use_gpu = False,
        only_onnxruntime = False
    )
    if fp16:
        optimizer.convert_float_to_float16(
        keep_io_types=True, disable_shape_infer=True, op_block_list=['RandomNormalLike']
    )
    optimizer.topological_sort()

    data_location = f"{model_path}.data"
    if os.path.exists(data_location):
        os.remove(data_location)
        
    # shutil.rmtree(model_dir)
    # os.mkdir(model_dir)
    # collate external tensor files into one
    onnx.save_model(
        optimizer.model,
        model_path,
        save_as_external_data=(model_type == "unet" and not fp16),
        all_tensors_to_one_file=True,
        location=f"{os.path.basename(model_path)}.data",
        convert_attribute=False,
    )

def export_unet_to_onnx(unet, output_path, device='cpu', opset_version=18, use_timesteps=False, fixed_timestep=0):
    """Export UNet model to ONNX format with external data support
    
    Args:
        unet: UNet model to export
        output_path: Path to save the ONNX model
        device: Device to use for export
        opset_version: ONNX opset version
        use_timesteps: If True, include timesteps as input. If False, use fixed timestep
        fixed_timestep: Fixed timestep value to use when use_timesteps=False
    """
    print(f"Exporting UNet to {output_path} with opset {opset_version}")
    if not use_timesteps:
        print(f"  Using fixed timestep: {fixed_timestep} (no timesteps input)")
    
    # Convert path to string to handle both string and Path objects
    output_path = str(output_path)
    
    # Ensure device is CPU for export to avoid CUDA issues
    device = torch.device('cpu')
    
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet_model, use_timesteps=True, fixed_timestep=0):
            super().__init__()
            # UNet class is a wrapper around UNet2DConditionModel
            # Access the actual model
            self.unet = unet_model.model
            self.use_timesteps = use_timesteps
            self.fixed_timestep = fixed_timestep
            
        def forward(self, input_latents, *args):
            if self.use_timesteps:
                # Original behavior: timesteps and audio_prompts as separate inputs
                timesteps, audio_prompts = args
            else:
                # Simplified behavior: only audio_prompts, use fixed timestep
                audio_prompts = args[0]
                # Create fixed timestep tensor with same batch size as input
                batch_size = input_latents.shape[0]
                timesteps = torch.tensor([self.fixed_timestep] * batch_size, 
                                       device=input_latents.device, 
                                       dtype=torch.long)
            
            # Call the UNet model directly
            result = self.unet(
                sample=input_latents,
                timestep=timesteps,
                encoder_hidden_states=audio_prompts,
                return_dict=False
            )
            return result[0]  # Return just the noise prediction
    
    # Ensure model and inputs are on CPU to avoid CUDA issues
    wrapper = UNetWrapper(unet, use_timesteps=use_timesteps, fixed_timestep=fixed_timestep).to(device)
    wrapper.eval()
    
    # Create dummy inputs matching the expected format (always on CPU for export)
    input_latents = torch.randn(1, 8, 32, 32, device=device)  # Concatenated latents
    audio_prompts = torch.randn(1, 50, 384, device=device)
    
    if use_timesteps:
        # Original behavior with timesteps input
        timesteps = torch.tensor([0], device=device, dtype=torch.long)
        dummy_inputs = (input_latents, timesteps, audio_prompts)
        input_names = ['input_latents', 'timesteps', 'audio_prompts']
        dynamic_axes = {
            'input_latents': {0: 'batch_size'},
            'timesteps': {0: 'batch_size'},
            'audio_prompts': {0: 'batch_size', 1: 'sequence_length'},
            'noise_prediction': {0: 'batch_size'}
        }
    else:
        # Simplified behavior without timesteps input
        dummy_inputs = (input_latents, audio_prompts)
        input_names = ['input_latents', 'audio_prompts']
        dynamic_axes = {
            'input_latents': {0: 'batch_size'},
            'audio_prompts': {0: 'batch_size', 1: 'sequence_length'},
            'noise_prediction': {0: 'batch_size'}
        }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # For very large models, we'll use a different approach
    if output_path.endswith('.onnx'):
        temp_path = output_path[:-5] + '_temp.onnx'
    else:
        temp_path = output_path + '_temp.onnx'
    
    print("Attempting to export large UNet model with external data support...")
        
    try:
        # Direct export to file
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            temp_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=['noise_prediction'],
            # Uncomment for dynamic batch sizes if needed
            # dynamic_axes=dynamic_axes,
            verbose=False,
            # dynamo=True,
            training=torch.onnx.TrainingMode.EVAL
        )

        data_location = f"{output_path}.data"
        if os.path.exists(data_location):
            os.remove(data_location)

        import onnx
        # Load and convert to external data format
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

        tune_model(output_path, "unet", fp16=False)
        
        # [DON'T UNCOMMENT] Apply post-processing optimizations [doesn't work so commented out]
        # model = onnx.load(output_path)
        # model = patch_pow_constants(model)
        # model_simp, check = simplify(model)
        # onnx.save(model_simp, output_path)
            
        print(f"UNet exported successfully with external data")
        
    except Exception as e2:
        print(f"UNet model is too large for current ONNX export capabilities {e2}")
        return None
    
    print(f"UNet exported successfully to {output_path}")
    return wrapper

def is_one_element_tensor(tensor):
    arr = numpy_helper.to_array(tensor)
    return arr.shape == (1,)

def make_scalar_initializer(tensor, name):
    arr = numpy_helper.to_array(tensor)
    scalar = np.asscalar(arr)
    scalar_tensor = numpy_helper.from_array(np.array(scalar, dtype=arr.dtype), name=name)
    return scalar_tensor

def patch_pow_constants(model):
    # Build dict of initializer tensors
    init_dict = {init.name: init for init in model.graph.initializer}
    
    # Track new initializers
    new_initializers = []

    # Process Pow nodes
    for node in model.graph.node:
        if node.op_type == "Pow":
            for i, input_name in enumerate(node.input):
                # If the input is an initializer with shape [1]
                if input_name in init_dict:
                    tensor = init_dict[input_name]
                    if is_one_element_tensor(tensor):
                        scalar_tensor = make_scalar_initializer(tensor, tensor.name)
                        new_initializers.append(scalar_tensor)
                        init_dict[input_name] = scalar_tensor  # Replace in map

    # Replace all initializers with updated ones
    final_initializers = []
    used_names = set()
    for init in model.graph.initializer:
        if init.name in init_dict and init.name not in used_names:
            final_initializers.append(init_dict[init.name])
            used_names.add(init.name)

    model.graph.ClearField("initializer")
    model.graph.initializer.extend(final_initializers)

    # Also patch Constant nodes used in Pow
    for node in model.graph.node:
        if node.op_type == "Pow":
            for i, input_name in enumerate(node.input):
                # Find Constant node that outputs this
                for const_node in model.graph.node:
                    if const_node.op_type == "Constant" and const_node.output[0] == input_name:
                        for attr in const_node.attribute:
                            if attr.type == onnx.AttributeProto.TENSOR:
                                arr = numpy_helper.to_array(attr.t)
                                if arr.shape == (1,):
                                    scalar = np.asscalar(arr)
                                    scalar_tensor = numpy_helper.from_array(np.array(scalar, dtype=arr.dtype), name=attr.t.name)
                                    attr.t.CopyFrom(scalar_tensor)

    return model

def make_coreml_compatible(model):
    """
    Make ONNX model compatible with CoreML by addressing input dimension > 16384 issues.
    CoreML has a limitation where input dimensions cannot exceed 16384.
    This function replaces problematic normalization patterns with CoreML-friendly alternatives.
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import numpy as np
    
    print("Applying CoreML compatibility transformations...")
    
    # Find problematic patterns: Reshape[4D→3D] -> Normalization -> [Mul/Add operations] -> Reshape[3D→4D]
    problematic_patterns = []
    
    for i in range(len(model.graph.node) - 2):
        node1 = model.graph.node[i]
        
        # Look for initial Reshape that creates 3D tensor
        if node1.op_type == "Reshape":
            # Check if this reshape creates 3D tensor with potential large dimensions
            shape_input = node1.input[1] if len(node1.input) > 1 else None
            if shape_input:
                for init in model.graph.initializer:
                    if init.name == shape_input:
                        target_shape = numpy_helper.to_array(init)
                        # Look for [0, channels, -1] pattern that creates large last dimension
                        if len(target_shape) == 3 and target_shape[2] == -1:
                            # Found problematic initial reshape, now look for the complete pattern
                            pattern_nodes = [i]  # Start with the reshape
                            current_output = node1.output[0]
                            pattern_found = False
                            final_reshape_idx = None
                            norm_type = None
                            
                            # Trace through the graph to find the core Reshape->Norm->Reshape pattern
                            for j in range(i + 1, min(i + 4, len(model.graph.node))):  # Look ahead up to 3 nodes
                                candidate_node = model.graph.node[j]
                                
                                # Check if this node uses the current output tensor
                                if current_output in candidate_node.input:
                                    pattern_nodes.append(j)
                                    
                                    # Check if this is a normalization node
                                    if candidate_node.op_type in ["InstanceNormalization", "GroupNormalization"]:
                                        norm_type = candidate_node.op_type
                                        current_output = candidate_node.output[0]
                                    # Check if this is the second reshape (after normalization)
                                    elif candidate_node.op_type == "Reshape" and norm_type:
                                        final_reshape_idx = j
                                        pattern_found = True
                                        break
                                    else:
                                        # Not part of our core pattern
                                        break
                            
                            if pattern_found and norm_type and final_reshape_idx:
                                print(f"Found complex normalization pattern starting at {node1.name}")
                                print(f"  Pattern nodes: {[model.graph.node[idx].name for idx in pattern_nodes]}")
                                print(f"  Normalization type: {norm_type}")
                                print(f"  Reshape shape: {target_shape}")
                                problematic_patterns.append((pattern_nodes, norm_type, node1.input[0], model.graph.node[final_reshape_idx].output[0]))
                        break
    
    print(f"Found {len(problematic_patterns)} problematic normalization patterns")
    
    # Replace each pattern with a single 4D normalization
    nodes_to_remove = set()
    new_nodes = []
    
    for pattern_nodes, norm_type, input_tensor, output_tensor in problematic_patterns:
        # Get the original normalization node to copy attributes from
        norm_node = None
        for node_idx in pattern_nodes:
            node = model.graph.node[node_idx]
            if node.op_type in ["InstanceNormalization", "GroupNormalization"]:
                norm_node = node
                break
        
        if not norm_node:
            print(f"Warning: Could not find normalization node in pattern {pattern_nodes}")
            continue
            
        print(f"Replacing complex pattern with {len(pattern_nodes)} nodes: {[model.graph.node[idx].name for idx in pattern_nodes]}")
        print(f"  Input: {input_tensor} -> Output: {output_tensor}")
        
        # Create new normalization that preserves the original channel grouping
        # Get the channel dimension from the first reshape in the pattern
        first_reshape_idx = pattern_nodes[0]
        first_reshape_node = model.graph.node[first_reshape_idx]
        first_reshape_shape_input = first_reshape_node.input[1]
        
        # Get the target channels from the first reshape
        target_channels = None
        for init in model.graph.initializer:
            if init.name == first_reshape_shape_input:
                shape_array = numpy_helper.to_array(init)
                if len(shape_array) == 3:  # [0, channels, -1]
                    target_channels = shape_array[1]
                break
        
        if target_channels:
            # Create intermediate reshape to match the channel dimension of the normalization
            # Input: [1, 512, 32, 32] -> [1, 32, 32, -1] (preserving spatial as 4D)
            # Let ONNX calculate the last dimension automatically to preserve total elements
            
            intermediate_shape = np.array([0, int(target_channels), 32, -1], dtype=np.int64)
            intermediate_shape_name = f"{norm_node.name}_intermediate_shape"
            intermediate_shape_init = numpy_helper.from_array(intermediate_shape, intermediate_shape_name)
            
            # Add the intermediate shape to initializers
            model.graph.initializer.append(intermediate_shape_init)
            
            # First reshape: input -> intermediate 4D with correct channels
            intermediate_reshape = helper.make_node(
                "Reshape",
                inputs=[input_tensor, intermediate_shape_name],
                outputs=[f"{norm_node.name}_intermediate_4d"],
                name=f"{norm_node.name}_intermediate_reshape"
            )
            
            # Create the appropriate 4D normalization node
            if norm_type == "InstanceNormalization":
                new_norm_node = helper.make_node(
                    "InstanceNormalization",
                    inputs=[f"{norm_node.name}_intermediate_4d"] + norm_node.input[1:],
                    outputs=[f"{norm_node.name}_coreml_4d_output"],
                    name=f"{norm_node.name}_coreml_4d"
                )
            elif norm_type == "GroupNormalization":
                new_norm_node = helper.make_node(
                    "GroupNormalization",
                    inputs=[f"{norm_node.name}_intermediate_4d"] + norm_node.input[1:],
                    outputs=[f"{norm_node.name}_coreml_4d_output"],
                    name=f"{norm_node.name}_coreml_4d"
                )
            
            # Copy attributes from original normalization node
            for attr in norm_node.attribute:
                new_norm_node.attribute.append(attr)
            
            new_nodes.extend([intermediate_reshape, new_norm_node])
        else:
            # Fallback: direct 4D normalization (shouldn't happen)
            if norm_type == "InstanceNormalization":
                new_norm_node = helper.make_node(
                    "InstanceNormalization",
                    inputs=[input_tensor] + norm_node.input[1:],
                    outputs=[f"{norm_node.name}_coreml_4d_output"],
                    name=f"{norm_node.name}_coreml_4d"
                )
            elif norm_type == "GroupNormalization":
                new_norm_node = helper.make_node(
                    "GroupNormalization",
                    inputs=[input_tensor] + norm_node.input[1:],
                    outputs=[f"{norm_node.name}_coreml_4d_output"],
                    name=f"{norm_node.name}_coreml_4d"
                )
            
            for attr in norm_node.attribute:
                new_norm_node.attribute.append(attr)
            new_nodes.append(new_norm_node)
        
        # Add final reshape to match the original pattern's output shape
        final_shape_input = None
        if len(pattern_nodes) >= 3:  # Should have: [first_reshape, norm, second_reshape]
            second_reshape_idx = pattern_nodes[2]
            second_reshape_node = model.graph.node[second_reshape_idx]
            if second_reshape_node.op_type == "Reshape":
                final_shape_input = second_reshape_node.input[1]  # Get the target shape
        
        if final_shape_input:
            # Add a reshape to match the original pattern's final output shape
            final_reshape = helper.make_node(
                "Reshape",
                inputs=[f"{norm_node.name}_coreml_4d_output", final_shape_input],
                outputs=[output_tensor],
                name=f"{norm_node.name}_final_reshape"
            )
            new_nodes.append(final_reshape)
        else:
            # No final reshape needed - update the last normalization node to output directly
            if new_nodes:
                new_nodes[-1].output[0] = output_tensor
        
        # Mark all nodes in the pattern for removal
        for node_idx in pattern_nodes:
            nodes_to_remove.add(node_idx)
        
        print(f"  Replaced with direct 4D {norm_type}")
    
    # Apply the replacements
    if new_nodes:
        # First, collect all intermediate tensor names that will be replaced
        tensor_replacements = {}  # old_tensor_name -> new_tensor_name
        
        # Collect tensor names before we modify anything
        original_nodes = list(model.graph.node)
        for pattern_nodes, norm_type, input_tensor, output_tensor in problematic_patterns:
            # Map all intermediate tensors in the pattern to the final output tensor
            for node_idx in pattern_nodes:
                if node_idx < len(original_nodes):
                    node = original_nodes[node_idx]
                    for output_name in node.output:
                        if output_name != output_tensor:  # Don't replace the final output with itself
                            tensor_replacements[output_name] = output_tensor
        
        print(f"Tensor replacements: {len(tensor_replacements)} mappings")
        
        # Update ALL nodes in the graph to use the new tensor names
        updated_inputs = 0
        updated_outputs = 0
        
        for node in model.graph.node:
            # Update input references
            new_inputs = []
            for input_name in node.input:
                if input_name in tensor_replacements:
                    new_inputs.append(tensor_replacements[input_name])
                    updated_inputs += 1
                else:
                    new_inputs.append(input_name)
            node.input[:] = new_inputs
            
            # Update output references (though this should be rare)
            new_outputs = []
            for output_name in node.output:
                if output_name in tensor_replacements:
                    new_outputs.append(tensor_replacements[output_name])
                    updated_outputs += 1
                else:
                    new_outputs.append(output_name)
            node.output[:] = new_outputs
        
        print(f"Updated {updated_inputs} input references and {updated_outputs} output references")
        
        # Now create new node list without removed nodes
        new_node_list = []
        for i, node in enumerate(model.graph.node):
            if i not in nodes_to_remove:
                new_node_list.append(node)
        
        # Add the new nodes
        new_node_list.extend(new_nodes)
        
        # Replace the graph nodes
        del model.graph.node[:]
        model.graph.node.extend(new_node_list)
        
        print(f"Successfully replaced {len(problematic_patterns)} normalization patterns")
        
        # Clean up value_info for removed intermediate tensors
        removed_tensors = set(tensor_replacements.keys())
        
        new_value_info = []
        for value_info in model.graph.value_info:
            if value_info.name not in removed_tensors:
                new_value_info.append(value_info)
        
        del model.graph.value_info[:]
        model.graph.value_info.extend(new_value_info)
        
        print(f"Cleaned up {len(removed_tensors)} intermediate tensor references")
    else:
        print("No patterns replaced - normalization operations may still have large intermediate dimensions")
    
    print("CoreML compatibility transformations completed.")
    return model

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
        training=torch.onnx.TrainingMode.EVAL,
        # dynamo=True
    )

    tune_model(output_path, "vae", fp16=False)

    # Apply post-processing optimizations
    model = onnx.load(output_path)
    model_simp, check = simplify(model)
    
    # Apply CoreML compatibility transformations
    model_simp = make_coreml_compatible(model_simp)
    
    onnx.save(model_simp, output_path)

    
    
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

    tune_model(output_path, "vae", fp16=False)

    # Apply post-processing optimizations
    model = onnx.load(output_path)
    model_simp, check = simplify(model)
    
    # Apply CoreML compatibility transformations
    model_simp = make_coreml_compatible(model_simp)
    
    onnx.save(model_simp, output_path)
    
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

def convert_model_to_fp16(fp32_model_path, fp16_model_path, model_type="general"):
    """Convert FP32 ONNX model to FP16"""
    try:
        print(f"Converting {fp32_model_path} to FP16...")
        
        # Copy the FP32 model first
        shutil.copy(fp32_model_path, fp16_model_path)
        
        # Copy external data file if exists
        fp16_data_location = f"{fp16_model_path}.data"
        if os.path.exists(fp16_data_location):
            os.remove(fp16_data_location)
        
        # Apply FP16 conversion using tune_model
        tune_model(fp16_model_path, model_type, fp16=True)
        
        # Apply post-processing optimizations
        model = onnx.load(fp16_model_path)
        model = patch_pow_constants(model)
        model_simp, check = simplify(model)
        
        # Save original model as backup
        shutil.copy(fp16_model_path, fp16_model_path + ".original")
        onnx.save(model_simp, fp16_model_path)
        
        print(f"✓ FP16 model saved to {fp16_model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to FP16: {e}")
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

def cleanup_export_directory(output_dir):
    """Clean up export directory, keeping only .onnx, .onnx.data, and config.json files"""
    print(f"\n🧹 Cleaning up export directory: {output_dir}")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    kept_files = []
    removed_files = []
    
    for file_path in output_path.iterdir():
        if file_path.is_file():
            filename = file_path.name
            
            # Keep these files
            if (filename.endswith('.onnx') or 
                filename.endswith('.onnx.data') or 
                filename == 'onnx_config.json'):
                kept_files.append(filename)
            else:
                # Remove everything else
                try:
                    file_path.unlink()
                    removed_files.append(filename)
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")
    
    print(f"✓ Kept {len(kept_files)} essential files: {', '.join(kept_files)}")
    if removed_files:
        print(f"🗑️ Removed {len(removed_files)} temporary files")
    else:
        print("📝 No temporary files to remove")

def copy_to_streaming_assets(source_dir):
    """Copy exported models to Unity StreamingAssets folder"""
    # Define StreamingAssets path
    unity_streaming_assets = Path("../MysteryAI/Assets/StreamingAssets/LiveTalk")
    
    if not unity_streaming_assets.exists():
        print(f"Creating StreamingAssets directory: {unity_streaming_assets}")
        unity_streaming_assets.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    copied_files = []
    
    # Models to copy (all precision variants)
    models_to_copy = [
        "unet_fp32.onnx",
        "unet_fp16.onnx", 
        "unet_int8.onnx",
        "vae_encoder_fp32.onnx",
        "vae_encoder_fp16.onnx",
        "vae_encoder_int8.onnx",
        "vae_decoder_fp32.onnx",
        "vae_decoder_fp16.onnx",
        "vae_decoder_int8.onnx",
        "positional_encoding_fp32.onnx",
        "positional_encoding_fp16.onnx",
        "positional_encoding_int8.onnx",
        "whisper_encoder_fp32.onnx",
        "whisper_encoder_fp16.onnx",
        "whisper_encoder_int8.onnx",
        "face_parsing_fp32.onnx",
        "face_parsing_fp16.onnx",
        "face_parsing_int8.onnx",
        "onnx_config.json"
    ]
    
    # Copy external data files for large models
    external_data_files = [
        "unet_fp32.onnx.data",
        "unet_fp16.onnx.data",
        "unet_int8.onnx.data"
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

def export_model_with_precisions(export_func, model, base_path, model_name, export_fp32=True, export_fp16=False, export_int8=False, device="cpu", opset_version=18, **kwargs):
    """Export model in multiple precision formats"""
    success_count = 0
    base_path_str = str(base_path)
    model_type = model_name.lower().replace(" ", "_")
    
    # Create precision-specific paths
    fp32_path = base_path_str
    fp16_path = base_path_str.replace('.onnx', '_fp16.onnx')
    int8_path = base_path_str.replace('.onnx', '_int8.onnx')
    
    # Export FP32 model first (base model)
    if export_fp32 or export_fp16 or export_int8:  # Need FP32 as base for conversions
        try:
            print(f"\n=== Exporting {model_name} (FP32) ===")
            if export_func(model, fp32_path, device, opset_version, **kwargs):
                if verify_onnx_model(fp32_path):
                    if export_fp32:
                        success_count += 1
                        print(f"✓ {model_name} FP32 export successful")
                    
                    # Convert to FP16 if requested
                    if export_fp16:
                        if convert_model_to_fp16(fp32_path, fp16_path, model_type):
                            if verify_onnx_model(fp16_path):
                                success_count += 1
                                print(f"✓ {model_name} FP16 conversion successful")
                            else:
                                print(f"✗ {model_name} FP16 model verification failed")
                        else:
                            print(f"✗ {model_name} FP16 conversion failed")
                    
                    # Convert to INT8 if requested
                    if export_int8:
                        if convert_model_to_int8(fp32_path, int8_path, model_type):
                            if verify_onnx_model(int8_path):
                                success_count += 1
                                print(f"✓ {model_name} INT8 quantization successful")
                            else:
                                print(f"✗ {model_name} INT8 model verification failed")
                        else:
                            print(f"✗ {model_name} INT8 quantization failed")
                    
                    # Remove FP32 if not requested (was only needed for conversion)
                    if not export_fp32 and os.path.exists(fp32_path):
                        os.remove(fp32_path)
                        # Also remove external data file if exists
                        fp32_data = fp32_path + ".data"
                        if os.path.exists(fp32_data):
                            os.remove(fp32_data)
                else:
                    print(f"✗ {model_name} FP32 model verification failed")
            else:
                print(f"✗ {model_name} FP32 export failed")
        except Exception as e:
            print(f"✗ Failed to export {model_name}: {e}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Export MuseTalk models to ONNX")
    parser.add_argument("--output_dir", default="./models/onnx", 
                       help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", 
                       help="Device to use for export (cpu/cuda)")
    parser.add_argument("--models", nargs="+", 
                       choices=["unet", "vae_encoder", "vae_decoder", "pe", "whisper", "face_parsing", "all"],
                       default=["all"], help="Models to export")
    parser.add_argument("--opset_version", type=int, default=18,
                       help="ONNX opset version to use (11-18)")
    parser.add_argument("--precision", choices=["all", "fp32", "fp16", "floating", "int8"], 
                       default="floating", help="Precision to export: all (fp32+fp16+int8), fp32, fp16, floating (fp32+fp16), int8")
    parser.add_argument("--copy-to-unity", action="store_true", default=True,
                       help="Copy exported models to Unity StreamingAssets (default: True)")
    parser.add_argument("--no-copy-unity", action="store_true", 
                       help="Disable copying to Unity StreamingAssets")
    
    # UNet-specific options
    parser.add_argument("--unet-use-timesteps", action="store_true", default=False,
                       help="Include timesteps as input to UNet (default: False for simplified interface)")
    parser.add_argument("--unet-fixed-timestep", type=int, default=0,
                       help="Fixed timestep value to use when --unet-use-timesteps is False (default: 0)")
    
    args = parser.parse_args()
    
    # Handle precision and Unity copy flags
    copy_to_unity = args.copy_to_unity and not args.no_copy_unity
    
    # Determine which precisions to export
    export_fp32 = args.precision in ["all", "fp32", "floating"]
    export_fp16 = args.precision in ["all", "fp16", "floating"]
    export_int8 = args.precision in ["all", "int8"] and INT8_AVAILABLE
    
    if args.precision in ["all", "int8"] and not INT8_AVAILABLE:
        print("⚠️ INT8 quantization requested but onnxruntime quantization not available")
        export_int8 = False
    
    # Print precision configuration
    precisions = []
    if export_fp32: precisions.append("FP32")
    if export_fp16: precisions.append("FP16")
    if export_int8: precisions.append("INT8")
    print(f"📝 Exporting models in precision(s): {', '.join(precisions)}")
    
    # Print UNet configuration
    if not args.unet_use_timesteps:
        print(f"🎯 UNet simplified mode: Using fixed timestep {args.unet_fixed_timestep} (no timesteps input)")
    else:
        print(f"🎯 UNet standard mode: Including timesteps as input")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Using ONNX opset version: {args.opset_version}")
    
    # Load models (only v1.5 supported for ONNX)
    unet_model_path = "./models/musetalkV15/unet.pth"
    unet_config_path = "./models/musetalkV15/musetalk.json"
    
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

        # clean directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
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
            unet_path = output_dir / "unet.onnx"
            try:
                # Use custom export function for UNet with timestep options
                print(f"\n=== Exporting UNet (timesteps: {args.unet_use_timesteps}) ===")
                success_count += export_model_with_precisions(
                    export_unet_to_onnx, unet, unet_path, "UNet", 
                    export_fp32, export_fp16, export_int8, device, args.opset_version,
                    use_timesteps=args.unet_use_timesteps, fixed_timestep=args.unet_fixed_timestep
                )
            except Exception as e:
                print(f"Failed to export UNet: {e}")
        
        if "vae_encoder" in models_to_export:
            vae_encoder_path = output_dir / "vae_encoder.onnx"
            try:
                success_count += export_model_with_precisions(
                    export_vae_encoder_to_onnx, vae, vae_encoder_path, "VAE Encoder",
                    export_fp32, export_fp16, export_int8, device, args.opset_version
                )
            except Exception as e:
                print(f"Failed to export VAE Encoder: {e}")
        
        if "vae_decoder" in models_to_export:
            vae_decoder_path = output_dir / "vae_decoder.onnx"
            try:
                success_count += export_model_with_precisions(
                    export_vae_decoder_to_onnx, vae, vae_decoder_path, "VAE Decoder",
                    export_fp32, export_fp16, export_int8, device, args.opset_version
                )
            except Exception as e:
                print(f"Failed to export VAE Decoder: {e}")
        
        if "pe" in models_to_export:
            pe_path = output_dir / "positional_encoding.onnx"
            try:
                success_count += export_model_with_precisions(
                    export_positional_encoding_to_onnx, pe, pe_path, "Positional Encoding",
                    export_fp32, export_fp16, export_int8, device, args.opset_version
                )
            except Exception as e:
                print(f"Failed to export Positional Encoding: {e}")
        
        if "whisper" in models_to_export and 'whisper' in locals():
            whisper_path = output_dir / "whisper_encoder.onnx"
            try:
                success_count += export_model_with_precisions(
                    export_whisper_to_onnx, whisper, whisper_path, "Whisper",
                    export_fp32, export_fp16, export_int8, device, args.opset_version
                )
            except Exception as e:
                print(f"Failed to export Whisper: {e}")
        
        if "face_parsing" in models_to_export:
            face_parsing_path = output_dir / "face_parsing.onnx"
            try:
                success_count += export_model_with_precisions(
                    export_face_parsing_to_onnx, fp, face_parsing_path, "Face Parsing",
                    export_fp32, export_fp16, export_int8, device, args.opset_version
                )
            except Exception as e:
                print(f"Failed to export Face Parsing: {e}")
        
        print(f"\n✓ Successfully exported {success_count} models to ONNX format")
        print(f"Output directory: {output_dir}")
        
        # Save model configuration
        config = {
            "exported_models": models_to_export,
            "device": str(device),
            "opset_version": args.opset_version,
            "precision": args.precision,
            "precisions_exported": {
                "fp32": export_fp32,
                "fp16": export_fp16,  
                "int8": export_int8
            },
            "int8_available": INT8_AVAILABLE,
            "copied_to_unity": copy_to_unity,
            "success_count": success_count,
            "unet_config": {
                "use_timesteps": args.unet_use_timesteps,
                "fixed_timestep": args.unet_fixed_timestep
            }
        }
        
        config_path = output_dir / "onnx_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
        
        # Copy models to StreamingAssets
        if copy_to_unity:
            copied_files = copy_to_streaming_assets(output_dir)
        
        # Clean up temporary files
        cleanup_export_directory(output_dir)
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 