#!/bin/bash
# LTX-2.3 Video Generation Example
# Generates a test video using the LTX-2.3 GGUF model

set -euo pipefail

# Configuration
MODEL_DIR="/Users/user030/Documents/qvac/packages/diffusion-cpp/models"
DIFFUSION_MODEL="${MODEL_DIR}/LTX-2.3-22B-distilled-1.1-Q8_0.gguf"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-./ltx_generated_video.mp4}"
PROMPT="${PROMPT:-A beautiful sunset over the ocean with waves gently crashing on the shore, cinematic lighting, 4k quality}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, distorted}"
STEPS="${STEPS:-20}"
VIDEO_FRAMES="${VIDEO_FRAMES:-120}"
FPS="${FPS:-24}"
HEIGHT="${HEIGHT:-576}"
WIDTH="${WIDTH:-960}"
THREADS="${THREADS:-8}"

# Check if model exists
if [ ! -f "$DIFFUSION_MODEL" ]; then
    echo "Error: Model not found at $DIFFUSION_MODEL"
    echo "Please download LTX models using:"
    echo "  cd /Users/user030/Documents/qvac/packages/diffusion-cpp"
    echo "  ./scripts/download-model-ltx.sh"
    exit 1
fi

echo "================================"
echo "LTX-2.3 Video Generation"
echo "================================"
echo "Diffusion Model: $DIFFUSION_MODEL"
echo "Output Video: $OUTPUT_VIDEO"
echo "Prompt: $PROMPT"
echo "Negative Prompt: $NEGATIVE_PROMPT"
echo "Steps: $STEPS"
echo "Video Frames: $VIDEO_FRAMES"
echo "FPS: $FPS"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Threads: $THREADS"
echo "================================"
echo ""

# Run video generation
echo "Starting LTX video generation..."
./build/bin/sd-cli \
    -M vid_gen \
    --diffusion-model "$DIFFUSION_MODEL" \
    -p "$PROMPT" \
    -n "$NEGATIVE_PROMPT" \
    --steps "$STEPS" \
    --video-frames "$VIDEO_FRAMES" \
    --fps "$FPS" \
    -H "$HEIGHT" \
    -W "$WIDTH" \
    -t "$THREADS" \
    -o "$OUTPUT_VIDEO" \
    --preview none

echo ""
echo "✓ Video generation completed!"
echo "Output saved to: $OUTPUT_VIDEO"
echo ""
echo "To play the video:"
echo "  open $OUTPUT_VIDEO"
