# INT8 Quantization Guide for MuseTalk

This guide explains how to use INT8 quantization with MuseTalk for optimal CPU performance, especially on Mac systems.

## Overview

INT8 quantization reduces model size by ~75% and provides 2-4x faster inference on CPU compared to FP32, making it ideal for:
- Mac systems (CPU-only)
- CPU-only inference setups
- Memory-constrained environments
- Real-time applications

## Benefits of INT8 Quantization

- **Memory Reduction**: ~75% smaller model files
- **Speed Improvement**: 2-4x faster CPU inference
- **Better Cache Utilization**: Smaller models fit better in CPU cache
- **No GPU Required**: Optimized specifically for CPU execution

## Export Models with INT8 Quantization

### Basic Export (Recommended)
```bash
# Export all models with INT8 quantization (default)
python scripts/export_to_onnx.py --version v1.5

# Export specific models
python scripts/export_to_onnx.py --models unet vae_encoder vae_decoder --int8
```

### Advanced Options
```bash
# Export without INT8 (FP32 only)
python scripts/export_to_onnx.py --no-int8

# Export without copying to Unity
python scripts/export_to_onnx.py --no-copy-unity

# Export specific version
python scripts/export_to_onnx.py --version v1.0 --int8
```

## Run Inference with INT8

### Python Inference
```bash
# Use INT8 models (default)
python scripts/onnx_inference.py \
    --avatar_path ./assets/demo/yongen \
    --audio_path ./assets/demo/yongen/yongen_song.wav \
    --output_path ./results/yongen_int8

# Force FP32 models
python scripts/onnx_inference.py \
    --avatar_path ./assets/demo/yongen \
    --audio_path ./assets/demo/yongen/yongen_song.wav \
    --output_path ./results/yongen_fp32 \
    --no-int8
```

### Unity Integration
The Unity integration automatically uses INT8 models when available:

```csharp
var config = new MuseTalkConfig
{
    ModelPath = "MuseTalk",
    Version = "v15",
    UseINT8 = true,  // Default: true
    PreferINT8Models = true  // Default: true
};

var inference = new MuseTalkInference(config);
```

## Model File Structure

After export, you'll have both FP32 and INT8 versions:

```
models/onnx/
├── unet_v15.onnx              # FP32 version
├── unet_v15_int8.onnx         # INT8 version (75% smaller)
├── vae_encoder_v15.onnx       # FP32 version
├── vae_encoder_v15_int8.onnx  # INT8 version
├── vae_decoder_v15.onnx       # FP32 version
├── vae_decoder_v15_int8.onnx  # INT8 version
└── ...
```

## Performance Comparison

| Model Type | Size | CPU Inference Speed | Memory Usage |
|------------|------|-------------------|--------------|
| FP32       | 100% | 1x (baseline)     | 100%         |
| INT8       | ~25% | 2-4x faster       | ~25%         |

## Troubleshooting

### INT8 Models Not Found
If you see warnings about INT8 models not found:
```
⚠️ INT8 model not found: models/onnx/unet_v15_int8.onnx, falling back to FP32
```

**Solution**: Re-export models with INT8 quantization:
```bash
python scripts/export_to_onnx.py --int8
```

### Quantization Errors
If quantization fails during export:
```
✗ Failed to convert model to INT8: quantize_dynamic() got an unexpected keyword argument 'optimize_model'
```

**Solution**: This is a version compatibility issue. The script will automatically fall back to basic quantization options.

### Performance Issues
If INT8 inference is slower than expected:
- Ensure you're using CPU device (`--device cpu`)
- Check that INT8 models are actually being loaded (look for "Using INT8 model" in logs)
- Verify ONNX Runtime is optimized for your CPU architecture

## Best Practices

1. **Always use INT8 for CPU inference** - It's the default for good reason
2. **Test both FP32 and INT8** - Verify quality is acceptable for your use case
3. **Monitor memory usage** - INT8 should use significantly less RAM
4. **Use appropriate batch sizes** - Smaller batches often work better with INT8

## Technical Details

### Quantization Method
- **Dynamic Quantization**: Weights are quantized to INT8, activations computed in FP32
- **Per-channel quantization**: Better accuracy than per-tensor
- **Symmetric quantization**: Optimized for neural network weights

### Supported Models
All MuseTalk models support INT8 quantization:
- UNet (diffusion model)
- VAE Encoder/Decoder
- Positional Encoding
- Whisper Encoder
- Face Parsing (BiSeNet)

### ONNX Runtime Optimizations
INT8 models automatically enable:
- Graph optimization level: ALL
- Execution mode: PARALLEL
- Memory arena optimizations
- CPU-specific kernel selection