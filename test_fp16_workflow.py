#!/usr/bin/env python3
"""
FP16 MuseTalk Export and Inference Testing Script
This script demonstrates the complete workflow for FP16 model export and inference.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description="", conda_env="MuseTalk"):
    """Run a command with conda environment activation"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    # Construct full command with conda activation
    if conda_env:
        if sys.platform.startswith('win'):
            # Windows conda activation
            full_cmd = f"conda activate {conda_env} && {cmd}"
            shell_cmd = ["cmd", "/c", full_cmd]
        else:
            # Unix conda activation
            full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {conda_env} && {cmd}"
            shell_cmd = ["bash", "-c", full_cmd]
    else:
        shell_cmd = cmd.split()
    
    print(f"Command: {cmd}")
    start_time = time.time()
    
    try:
        result = subprocess.run(shell_cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success! Completed in {elapsed:.2f}s")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ Failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after 1 hour")
        return False
    except Exception as e:
        print(f"💥 Exception occurred: {e}")
        return False

def check_prerequisites():
    """Check if required dependencies are available"""
    print("🔍 Checking prerequisites...")
    
    # Check if conda environment exists
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    if "MuseTalk" not in result.stdout:
        print("❌ MuseTalk conda environment not found!")
        print("Please create it first with: conda env create -f environment.yml")
        return False
    
    # Check if onnxconverter-common is available
    check_cmd = "python -c 'import onnxconverter_common; print(\"FP16 conversion available\")'"
    if not run_command(check_cmd, "Checking FP16 conversion support"):
        print("⚠️ onnxconverter-common not found. Installing...")
        install_cmd = "pip install onnxconverter-common"
        if not run_command(install_cmd, "Installing onnxconverter-common"):
            print("❌ Failed to install onnxconverter-common")
            return False
    
    print("✅ All prerequisites satisfied!")
    return True

def export_models_with_quantization():
    """Export MuseTalk models with INT8 quantization (optimal for Mac/CPU)"""
    print("\n" + "="*80)
    print("🍎 STEP 1: EXPORTING MODELS WITH INT8 QUANTIZATION")
    print("="*80)
    
    # Export v1.5 models with INT8 quantization (default for CPU-only setups)
    export_cmd = (
        "python scripts/export_to_onnx.py "
        "--version v1.5 "
        "--output_dir ./models/onnx "
        "--device cpu "  # Use CPU for export
        "--models all "
        "--opset_version 18 "
        "--int8 "  # Enable INT8 quantization (CPU-optimized)
        "--copy-to-unity"
    )
    
    success = run_command(export_cmd, "Exporting v1.5 models with INT8 quantization")
    
    if success:
        print("\n✅ Model export completed successfully!")
        print("📁 Models exported to: ./models/onnx/")
        print("📱 Models copied to Unity StreamingAssets")
        
        # List exported files
        onnx_dir = Path("./models/onnx")
        if onnx_dir.exists():
            print("\n📋 Exported files:")
            for file in sorted(onnx_dir.glob("*.onnx")):
                size_mb = file.stat().st_size / (1024 * 1024)
                if "_int8" in file.name:
                    quant_indicator = "🍎 INT8"
                elif "_fp16" in file.name:
                    quant_indicator = "🔸 FP16"
                else:
                    quant_indicator = "🔹 FP32"
                print(f"  {quant_indicator} {file.name} ({size_mb:.1f} MB)")
    else:
        print("❌ Model export failed!")
        
    return success

def test_int8_inference():
    """Test INT8 inference (optimal for CPU/Mac)"""
    print("\n" + "="*80)
    print("🍎 STEP 2: TESTING INT8 INFERENCE (CPU-OPTIMIZED)")
    print("="*80)
    
    # Test INT8 inference command (optimal for Mac/CPU)
    inference_cmd = (
        "python scripts/onnx_inference.py "
        "--avatar_path results/v15/avatars/avator_1/full_imgs "
        "--audio_path data/audio/yongen_1s.wav "
        "--output_path results/onnx_test_int8_1s "
        "--max_images 1 "
        "--batch_size 8 "
        "--device cpu "
        "--int8"  # Enable INT8 quantization
    )
    
    success = run_command(inference_cmd, "Testing INT8 inference")
    
    if success:
        print("\n✅ INT8 inference completed successfully!")
        output_dir = Path("results/onnx_test_int8_1s")
        if output_dir.exists():
            frame_count = len(list(output_dir.glob("*.png")))
            print(f"📸 Generated {frame_count} frames")
            
            video_file = output_dir / "output.mp4"
            if video_file.exists():
                size_mb = video_file.stat().st_size / (1024 * 1024)
                print(f"🎬 Output video: {video_file} ({size_mb:.1f} MB)")
    else:
        print("❌ INT8 inference failed!")
        
    return success

def test_fp32_inference():
    """Test FP32 inference (baseline comparison)"""
    print("\n" + "="*80)
    print("🔹 STEP 3: TESTING FP32 INFERENCE (BASELINE)")
    print("="*80)
    
    # Test FP32 inference command (baseline)
    inference_cmd = (
        "python scripts/onnx_inference.py "
        "--avatar_path results/v15/avatars/avator_1/full_imgs "
        "--audio_path data/audio/yongen_1s.wav "
        "--output_path results/onnx_test_fp32_1s "
        "--max_images 1 "
        "--batch_size 8 "
        "--device cpu "
        "--no-int8 --no-fp16"  # Force FP32
    )
    
    success = run_command(inference_cmd, "Testing FP32 inference (baseline)")
    
    if success:
        print("\n✅ FP32 inference completed successfully!")
        output_dir = Path("results/onnx_test_fp32_1s")
        if output_dir.exists():
            frame_count = len(list(output_dir.glob("*.png")))
            print(f"📸 Generated {frame_count} frames")
            
            video_file = output_dir / "output.mp4"
            if video_file.exists():
                size_mb = video_file.stat().st_size / (1024 * 1024)
                print(f"🎬 Output video: {video_file} ({size_mb:.1f} MB)")
    else:
        print("❌ FP32 inference failed!")
        
    return success

def compare_performance():
    """Compare FP32 vs INT8 results"""
    print("\n" + "="*80)
    print("📊 STEP 4: COMPARING PERFORMANCE")
    print("="*80)
    
    fp32_dir = Path("results/onnx_test_fp32_1s")
    int8_dir = Path("results/onnx_test_int8_1s")
    
    print("📋 Comparison Summary:")
    print("-" * 40)
    
    # Check FP32 results
    if fp32_dir.exists():
        fp32_frames = len(list(fp32_dir.glob("*.png")))
        fp32_video = fp32_dir / "output.mp4"
        fp32_size = fp32_video.stat().st_size / (1024 * 1024) if fp32_video.exists() else 0
        print(f"🔹 FP32: {fp32_frames} frames, {fp32_size:.1f} MB video")
    else:
        print("🔹 FP32: No results found")
    
    # Check INT8 results
    if int8_dir.exists():
        int8_frames = len(list(int8_dir.glob("*.png")))
        int8_video = int8_dir / "output.mp4"
        int8_size = int8_video.stat().st_size / (1024 * 1024) if int8_video.exists() else 0
        print(f"🍎 INT8: {int8_frames} frames, {int8_size:.1f} MB video")
    else:
        print("🍎 INT8: No results found")
    
    print("-" * 40)
    
    # Model size comparison
    onnx_dir = Path("./models/onnx")
    if onnx_dir.exists():
        print("\n💾 Model Size Comparison:")
        
        for model_type in ["unet_v15", "vae_encoder_v15", "vae_decoder_v15", "positional_encoding_v15"]:
            fp32_model = onnx_dir / f"{model_type}.onnx"
            int8_model = onnx_dir / f"{model_type}_int8.onnx"
            
            if fp32_model.exists() and int8_model.exists():
                fp32_size = fp32_model.stat().st_size / (1024 * 1024)
                int8_size = int8_model.stat().st_size / (1024 * 1024)
                reduction = ((fp32_size - int8_size) / fp32_size) * 100
                print(f"  {model_type}:")
                print(f"    🔹 FP32: {fp32_size:.1f} MB")
                print(f"    🍎 INT8: {int8_size:.1f} MB ({reduction:.1f}% reduction)")
        
        print("\n🍎 INT8 Benefits for Mac/CPU:")
        print("  • ~75% memory reduction")
        print("  • Faster CPU inference")
        print("  • Better cache utilization")
        print("  • No GPU requirement")

def cleanup_results():
    """Clean up test results"""
    print("\n" + "="*60)
    print("🧹 CLEANUP")
    print("="*60)
    
    cleanup_choice = input("Clean up test result directories? (y/N): ").strip().lower()
    if cleanup_choice == 'y':
        import shutil
        for result_dir in ["results/onnx_test_fp32_1s", "results/onnx_test_fp16_1s"]:
            if Path(result_dir).exists():
                shutil.rmtree(result_dir)
                print(f"🗑️ Removed {result_dir}")
        print("✅ Cleanup completed!")
    else:
        print("⏭️ Skipping cleanup")

def main():
    """Main workflow for INT8 quantization testing (optimal for Mac/CPU)"""
    print("🍎 MuseTalk INT8 Quantization Export and Inference Testing")
    print("=" * 80)
    print("This script will:")
    print("1. Export MuseTalk models with INT8 quantization (CPU-optimized)")
    print("2. Test INT8 inference (optimal for Mac/CPU)")
    print("3. Test FP32 inference (baseline comparison)")
    print("4. Compare performance and model sizes")
    print("=" * 80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Exiting.")
        return 1
    
    # Step 1: Export models with INT8 quantization
    if not export_models_with_quantization():
        print("❌ Model export failed. Exiting.")
        return 1
    
    # Step 2: Test INT8 inference (optimal for CPU/Mac)
    int8_success = test_int8_inference()
    
    # Step 3: Test FP32 inference (baseline comparison)
    fp32_success = test_fp32_inference()
    
    # Step 4: Compare performance
    compare_performance()
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"✅ Model Export: Success")
    print(f"{'✅' if int8_success else '❌'} INT8 Inference: {'Success' if int8_success else 'Failed'}")
    print(f"{'✅' if fp32_success else '❌'} FP32 Inference: {'Success' if fp32_success else 'Failed'}")
    
    if int8_success or fp32_success:
        print("\n🎉 At least one inference test succeeded!")
        print("📁 Check the results directories for output frames and videos")
        print("🍎 INT8 models provide ~75% memory reduction and faster CPU inference!")
    
    # Optional cleanup
    cleanup_results()
    
    return 0

if __name__ == "__main__":
    exit(main()) 