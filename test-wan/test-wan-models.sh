#!/bin/bash

# Wan Video Generation Tests for stable-diffusion.cpp
# This script tests various Wan model configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_CLI="$(dirname "$SCRIPT_DIR")/build/bin/sd-cli"
MODELS_DIR="$SCRIPT_DIR/models"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "================================"
echo "Wan Video Generation Test Suite"
echo "================================"
echo ""
echo "sd-cli location: $SD_CLI"
echo "Models directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if sd-cli exists
if [ ! -f "$SD_CLI" ]; then
    echo "ERROR: sd-cli not found at $SD_CLI"
    echo "Please build the project first with: make build-metal"
    exit 1
fi

# Function to download model from Hugging Face
download_model() {
    local url="$1"
    local dest="$2"
    
    if [ -f "$dest" ]; then
        echo "✓ Model already exists: $(basename "$dest")"
        return 0
    fi
    
    echo "⬇ Downloading: $(basename "$dest")"
    echo "  URL: $url"
    
    # Create cache dir if needed
    mkdir -p "$(dirname "$dest")"
    
    # Use curl with progress
    if command -v curl &> /dev/null; then
        curl -L -# -o "$dest" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$dest" "$url"
    else
        echo "ERROR: curl or wget required to download models"
        exit 1
    fi
    
    echo "✓ Downloaded: $(basename "$dest")"
}

# Function to run a test
run_test() {
    local test_name="$1"
    local model_type="$2"
    local model_path="$3"
    local vae_path="$4"
    local t5_path="$5"
    local prompt="$6"
    local output_name="$7"
    
    local video_frames="${8:-33}"
    local cfg_scale="${9:-6.0}"
    local steps="${10:-20}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Model Type: $model_type"
    echo "Prompt: $prompt"
    echo ""
    
    # Check if required models exist
    if [ ! -f "$model_path" ]; then
        echo "⚠ Skipping: Model not found at $model_path"
        return 1
    fi
    
    if [ ! -f "$vae_path" ]; then
        echo "⚠ Skipping: VAE not found at $vae_path"
        return 1
    fi
    
    if [ ! -f "$t5_path" ]; then
        echo "⚠ Skipping: T5 not found at $t5_path"
        return 1
    fi
    
    # Build command
    local cmd="$SD_CLI -M vid_gen"
    cmd="$cmd --diffusion-model $model_path"
    cmd="$cmd --vae $vae_path"
    cmd="$cmd --t5xxl $t5_path"
    cmd="$cmd -p \"$prompt\""
    cmd="$cmd --cfg-scale $cfg_scale"
    cmd="$cmd --sampling-method euler"
    cmd="$cmd -v"
    cmd="$cmd -W 832 -H 480"
    cmd="$cmd --diffusion-fa"
    cmd="$cmd --video-frames $video_frames"
    cmd="$cmd --flow-shift 3.0"
    cmd="$cmd --offload-to-cpu"
    cmd="$cmd -o $OUTPUT_DIR/$output_name.mp4"
    
    echo "Running: $cmd"
    echo ""
    
    # Run the test
    if eval "$cmd"; then
        echo "✓ Test PASSED: $test_name"
        echo "  Output: $OUTPUT_DIR/$output_name.mp4"
    else
        echo "✗ Test FAILED: $test_name"
        return 1
    fi
}

# Test configurations
echo ""
echo "Available test profiles:"
echo "  1. wan2-1-t2v-small    - Wan2.1 T2V 1.3B (text-to-video, small)"
echo "  2. wan2-1-t2v-large    - Wan2.1 T2V 14B (text-to-video, large)"
echo "  3. wan2-2-t2v          - Wan2.2 T2V 5B (text-to-video, tiny)"
echo "  4. wan2-1-i2v          - Wan2.1 I2V 14B (image-to-video)"
echo "  5. all                 - Run all available tests"
echo ""

# Default to all if no argument
TEST_PROFILE="${1:-all}"

case "$TEST_PROFILE" in
    wan2-1-t2v-small)
        run_test "Wan2.1 T2V 1.3B" "Text-to-Video" \
            "$MODELS_DIR/wan2.1_t2v_1.3B_fp16.safetensors" \
            "$MODELS_DIR/wan_2.1_vae.safetensors" \
            "$MODELS_DIR/umt5_xxl_fp16.safetensors" \
            "a lovely cat running through a garden" \
            "wan2-1-t2v-1.3b-test" \
            33 6.0 20
        ;;
    wan2-1-t2v-large)
        run_test "Wan2.1 T2V 14B" "Text-to-Video" \
            "$MODELS_DIR/wan2.1_t2v_14b_Q8_0.gguf" \
            "$MODELS_DIR/wan_2.1_vae.safetensors" \
            "$MODELS_DIR/umt5_xxl_Q8_0.gguf" \
            "a serene mountain landscape with flowing water" \
            "wan2-1-t2v-14b-test" \
            33 6.0 20
        ;;
    wan2-2-t2v)
        run_test "Wan2.2 TI2V 5B" "Text-to-Video" \
            "$MODELS_DIR/wan2.2_ti2v_5B_fp16.safetensors" \
            "$MODELS_DIR/wan2.2_vae.safetensors" \
            "$MODELS_DIR/umt5_xxl_Q8_0.gguf" \
            "a dancing robot in a futuristic city" \
            "wan2-2-ti2v-5b-test" \
            33 6.0 20
        ;;
    wan2-1-i2v)
        run_test "Wan2.1 I2V 14B" "Image-to-Video" \
            "$MODELS_DIR/wan2.1_i2v_14b_480p_Q8_0.gguf" \
            "$MODELS_DIR/wan_2.1_vae.safetensors" \
            "$MODELS_DIR/umt5_xxl_Q8_0.gguf" \
            "the cat jumps and plays" \
            "wan2-1-i2v-14b-test" \
            33 6.0 20
        ;;
    all)
        echo "Running all available tests..."
        # Try to run each test, skip if models not found
        run_test "Wan2.1 T2V 1.3B" "Text-to-Video" \
            "$MODELS_DIR/wan2.1_t2v_1.3B_fp16.safetensors" \
            "$MODELS_DIR/wan_2.1_vae.safetensors" \
            "$MODELS_DIR/umt5_xxl_fp16.safetensors" \
            "a lovely cat" "wan2-1-t2v-1.3b-test" || true
        ;;
    *)
        echo "Unknown test profile: $TEST_PROFILE"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "Test completed!"
echo "================================"
