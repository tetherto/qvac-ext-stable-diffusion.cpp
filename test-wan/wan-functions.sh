#!/bin/bash

# Quick reference for common Wan video generation commands
# Place this file in your shell profile or use manually

# Wan2.1 T2V 1.3B - Text-to-Video (Fast)
wan_t2v_small() {
    local prompt="${1:-a lovely cat}"
    local output="${2:-./output_small.mp4}"
    
    ~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli \
        -M vid_gen \
        --diffusion-model ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan2.1_t2v_1.3B_fp16.safetensors \
        --vae ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan_2.1_vae.safetensors \
        --t5xxl ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/umt5_xxl_fp16.safetensors \
        -p "$prompt" \
        --cfg-scale 6.0 \
        --sampling-method euler \
        -v \
        -W 832 -H 480 \
        --diffusion-fa \
        --video-frames 33 \
        --flow-shift 3.0 \
        --offload-to-cpu \
        -o "$output"
}

# Wan2.1 T2V 14B - Text-to-Video (High Quality)
wan_t2v_large() {
    local prompt="${1:-a serene landscape}"
    local output="${2:-./output_large.mp4}"
    
    ~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli \
        -M vid_gen \
        --diffusion-model ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan2.1_t2v_14b_Q8_0.gguf \
        --vae ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan_2.1_vae.safetensors \
        --t5xxl ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/umt5_xxl_Q8_0.gguf \
        -p "$prompt" \
        --cfg-scale 6.0 \
        --sampling-method euler \
        -v \
        -W 832 -H 480 \
        --diffusion-fa \
        --video-frames 33 \
        --flow-shift 3.0 \
        --offload-to-cpu \
        -o "$output"
}

# Wan2.1 I2V 14B - Image-to-Video
wan_i2v() {
    local image="${1:-./input.png}"
    local prompt="${2:-dynamic motion}"
    local output="${3:-./output_i2v.mp4}"
    
    if [ ! -f "$image" ]; then
        echo "Error: Image file not found: $image"
        return 1
    fi
    
    ~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli \
        -M vid_gen \
        --diffusion-model ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan2.1_i2v_14b_480p_Q8_0.gguf \
        --vae ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan_2.1_vae.safetensors \
        --t5xxl ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/umt5_xxl_Q8_0.gguf \
        --clip_vision ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/clip_vision_h.safetensors \
        -i "$image" \
        -p "$prompt" \
        --cfg-scale 6.0 \
        --sampling-method euler \
        -v \
        -W 480 -H 832 \
        --diffusion-fa \
        --video-frames 33 \
        --flow-shift 3.0 \
        --offload-to-cpu \
        -o "$output"
}

# Wan2.2 TI2V 5B - Tiny Text-to-Video (Mobile-friendly)
wan_ti2v() {
    local prompt="${1:-a dancing robot}"
    local output="${2:-./output_ti2v.mp4}"
    
    ~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli \
        -M vid_gen \
        --diffusion-model ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan2.2_ti2v_5B_fp16.safetensors \
        --vae ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan2.2_vae.safetensors \
        --t5xxl ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/umt5_xxl_Q8_0.gguf \
        -p "$prompt" \
        --cfg-scale 6.0 \
        --sampling-method euler \
        -v \
        -W 480 -H 832 \
        --diffusion-fa \
        --video-frames 33 \
        --flow-shift 3.0 \
        --offload-to-cpu \
        -o "$output"
}

# Benchmark - Quick performance test
wan_benchmark() {
    local model_size="${1:-small}"
    
    case "$model_size" in
        small)
            echo "Running Wan2.1 T2V 1.3B benchmark..."
            time wan_t2v_small "a red ball rolling" /tmp/wan_bench_small.mp4
            ;;
        large)
            echo "Running Wan2.1 T2V 14B benchmark..."
            time wan_t2v_large "a red ball rolling" /tmp/wan_bench_large.mp4
            ;;
        *)
            echo "Usage: wan_benchmark [small|large]"
            ;;
    esac
}

# Help
wan_help() {
    cat << 'EOF'
Wan Video Generation Functions
===============================

Available commands:

  wan_t2v_small <prompt> <output_file>
    - Fast text-to-video using 1.3B model
    - Example: wan_t2v_small "a cat jumping" ~/videos/cat.mp4

  wan_t2v_large <prompt> <output_file>
    - High-quality text-to-video using 14B model
    - Example: wan_t2v_large "ocean waves" ~/videos/ocean.mp4

  wan_i2v <image_file> <prompt> <output_file>
    - Image-to-video animation
    - Example: wan_i2v ~/image.png "the cat moves" ~/videos/animated.mp4

  wan_ti2v <prompt> <output_file>
    - Tiny text-to-video using 5B model (Wan2.2)
    - Example: wan_ti2v "a dancing robot" ~/videos/robot.mp4

  wan_benchmark [small|large]
    - Performance benchmark
    - Example: wan_benchmark large

  wan_help
    - Show this help message

Quick Start:
============

  # 1. Install functions to your shell
  source ~/.wan-functions.sh

  # 2. Generate a video
  wan_t2v_small "a cute dog playing in the park" ~/videos/output.mp4

  # 3. Check output
  open ~/videos/output.mp4

EOF
}

# Default action - show help
if [ "$1" == "help" ] || [ -z "$1" ]; then
    wan_help
fi
