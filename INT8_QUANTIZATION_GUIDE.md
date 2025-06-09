# MuseTalk INT8 Quantization Guide

## Overview
This guide explains the INT8 quantization optimizations implemented for MuseTalk models, providing significant performance improvements for CPU inference while maintaining high image quality.

## Optimal Configuration (Recommended)

**🎯 Automatic Quality/Performance Balance:**
- **VAE Models**: Always FP32 (preserves image quality, prevents color distortion)
- **Other Models**: INT8 QDQ (performance optimization, Mac-compatible)

### Model-Specific Strategy

| Model | Format | Reasoning |
|-------|--------|-----------|
| UNet | INT8 QDQ | Large model (3.4GB→851MB), quality preserved |
| **VAE Encoder** | **FP32** | Quality-sensitive, prevents color artifacts |
| **VAE Decoder** | **FP32** | Quality-sensitive, prevents blurriness |
| Positional Encoding | INT8 QDQ | Small model, performance gain |
| Whisper | INT8 QDQ | Audio features, performance gain |
| Face Parsing | INT8 QDQ | Segmentation, performance gain |

## Benefits

### 🎨 **Image Quality**
- **Perfect Colors**: FP32 VAE prevents color distortion and oversaturation
- **Sharp Details**: FP32 VAE preserves facial detail and texture quality
- **No Artifacts**: Eliminates quantization-induced blurriness and noise

### ⚡ **Performance**
- **75% Memory Reduction**: For UNet and auxiliary models
- **1.5-2x Faster**: CPU inference for quantized models
- **Mac Optimized**: QDQ format avoids ConvInteger compatibility issues

### 🧠 **Intelligent Defaults**
- **Automatic Detection**: VAE models automatically use FP32 regardless of settings
- **Zero Configuration**: Works out-of-the-box with optimal settings
- **Backward Compatible**: Fallback to FP32 when INT8 models unavailable

## Implementation

### Python Export (Automatic)
```bash
# Exports with optimal configuration automatically
python scripts/export_to_onnx.py --models all --int8 --copy-to-unity
```

**Behavior:**
- ✅ UNet: FP32 + INT8 QDQ exported
- ✅ VAE Encoder: **FP32 only** (quality preservation)
- ✅ VAE Decoder: **FP32 only** (quality preservation)  
- ✅ Others: FP32 + INT8 QDQ exported

### Unity C# (Automatic)
```csharp
// Default optimal configuration
var config = new MuseTalkConfig 
{
    UseINT8 = true,           // Enable INT8 where appropriate
    PreferINT8Models = true   // Try INT8 first
};

// VAE models automatically use FP32 regardless of UseINT8 setting
var inference = new MuseTalkInference(config);
```

### Python Inference (Automatic Fallback)
```bash
# Smart fallback: uses FP32 for VAE, INT8 for others
python scripts/onnx_inference.py --avatar_path <avatars> --audio_path <audio> --output_path <output>
```

## Technical Details

### QDQ Format (Mac Compatible)
- Uses Quantize-Dequantize operations instead of ConvInteger
- Compatible with macOS ONNX Runtime limitations
- Static quantization with MinMax calibration
- External data format for large models (>100MB)

### Quality Preservation Logic
```python
# Python export automatically skips INT8 for VAE
if model_type in ["vae_encoder", "vae_decoder"]:
    print("⚠️ Skipping INT8 quantization for VAE to preserve image quality")
    export_int8 = False
```

```csharp
// Unity automatically uses FP32 for VAE
bool isVAEModel = baseName.Contains("vae_encoder") || baseName.Contains("vae_decoder");
if (isVAEModel)
{
    return baseModelPath; // Always FP32 for VAE
}
```

## File Structure

### StreamingAssets (Unity)
```
MuseTalk/
├── unet_v15.onnx              # FP32 fallback
├── unet_v15_int8.onnx         # INT8 optimized ✓
├── vae_encoder_v15.onnx       # FP32 only (quality) ✓
├── vae_decoder_v15.onnx       # FP32 only (quality) ✓
├── whisper_encoder_int8.onnx  # INT8 optimized ✓
├── face_parsing_int8.onnx     # INT8 optimized ✓
└── positional_encoding_v15_int8.onnx # INT8 optimized ✓
```

### Model Sizes
```
UNet:               3.4GB → 851MB  (75% reduction)
VAE Encoder:        130MB (FP32)   (quality preserved)
VAE Decoder:        189MB (FP32)   (quality preserved)
Whisper:            33MB → 10MB    (70% reduction)
Face Parsing:       53MB → 13MB    (75% reduction)
Positional Encoding: 68KB → 18KB  (75% reduction)
```

## Testing Results

### Quality Comparison
- ❌ **INT8 VAE**: Color distortion, blurriness, artifacts
- ✅ **FP32 VAE**: Natural colors, sharp details, high quality
- ✅ **Mixed Precision**: Best of both worlds

### Performance
- **Inference Speed**: ~30-35 seconds for 25 frames (1 second video)
- **Effective FPS**: ~0.8-1.0 FPS on Mac CPU
- **Memory Usage**: ~50-60% reduction overall
- **Platform**: Optimized for Mac M1/M2 CPU inference

## Troubleshooting

### Common Issues
1. **Poor Image Quality**: Ensure VAE models are using FP32 (automatic)
2. **ConvInteger Errors**: Use QDQ format instead (automatic)
3. **Model Not Found**: Check StreamingAssets directory
4. **Slow Performance**: Verify INT8 models are loading for non-VAE models

### Verification
Check model loading logs:
```
✓ Using FP32 for VAE model (quality preservation): vae_encoder_v15.onnx
✓ Using INT8 model (performance optimization): unet_v15_int8.onnx
```

## Conclusion

The optimal configuration provides:
- 🎨 **Excellent image quality** (FP32 VAE)
- ⚡ **Strong performance** (INT8 for appropriate models)
- 🍎 **Mac compatibility** (QDQ format)
- 🔧 **Zero configuration** (automatic behavior)

This represents the best balance of quality and performance for CPU-based MuseTalk inference.