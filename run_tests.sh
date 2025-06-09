#!/bin/bash

# MuseTalk INT8 Quantization Testing Commands
# Make sure to activate conda environment first: conda activate MuseTalk

echo "🍎 MuseTalk INT8 Quantization Testing Commands"
echo "=============================================="

# Set conda environment
CONDA_ENV="MuseTalk"

# Function to run command with conda activation
run_cmd() {
    echo "Running: $1"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    eval "$1"
}

echo "📋 Available commands:"
echo ""

echo "1️⃣ Export models with INT8 quantization (optimal for Mac/CPU):"
echo "conda activate MuseTalk"
echo "python scripts/export_to_onnx.py --version v1.5 --output_dir ./models/onnx --device cpu --models all --opset_version 18 --int8 --copy-to-unity"
echo ""

echo "2️⃣ Test INT8 inference (optimal for Mac/CPU):"
echo "conda activate MuseTalk"
echo "python scripts/onnx_inference.py --avatar_path results/v15/avatars/avator_1/full_imgs --audio_path data/audio/yongen_1s.wav --output_path results/onnx_test_int8_1s --max_images 1 --batch_size 8 --device cpu --int8"
echo ""

echo "3️⃣ Test FP32 inference (baseline comparison):"
echo "conda activate MuseTalk"
echo "python scripts/onnx_inference.py --avatar_path results/v15/avatars/avator_1/full_imgs --audio_path data/audio/yongen_1s.wav --output_path results/onnx_test_fp32_1s --max_images 1 --batch_size 8 --device cpu --no-int8 --no-fp16"
echo ""

echo "4️⃣ Check CUDA availability:"
echo "conda activate MuseTalk"
echo "python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'"
echo ""

echo "5️⃣ Install FP16 conversion dependencies:"
echo "conda activate MuseTalk"
echo "pip install onnxconverter-common"
echo ""

echo "💡 Usage examples:"
echo ""

# Check if user wants to run a specific command
read -p "Run full automated test? (y/N): " choice
if [[ $choice == "y" || $choice == "Y" ]]; then
    echo "🚀 Running full automated test..."
    run_cmd "python test_fp16_workflow.py"
elif [[ $choice == "1" ]]; then
    echo "🔄 Exporting models..."
    run_cmd "python scripts/export_to_onnx.py --version v1.5 --output_dir ./models/onnx --device cpu --models all --opset_version 18 --fp16 --copy-to-unity"
elif [[ $choice == "2" ]]; then
    echo "🧪 Testing FP32 inference..."
    run_cmd "python scripts/onnx_inference.py --avatar_path results/v15/avatars/avator_1/full_imgs --audio_path data/audio/yongen_1s.wav --output_path results/onnx_test_1s --max_images 1 --batch_size 8"
elif [[ $choice == "3" ]]; then
    echo "🚀 Testing FP16 inference..."
    run_cmd "python scripts/onnx_inference.py --avatar_path results/v15/avatars/avator_1/full_imgs --audio_path data/audio/yongen_1s.wav --output_path results/onnx_test_fp16_1s --max_images 1 --batch_size 8 --device cuda --fp16"
elif [[ $choice == "4" ]]; then
    echo "🔍 Checking CUDA availability..."
    run_cmd "python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'"
else
    echo "ℹ️ No command selected. Use the commands above manually."
fi

echo ""
echo "✅ Done! Check the output directories for results." 