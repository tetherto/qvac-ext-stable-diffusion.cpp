#!/usr/bin/env bash
set -euo pipefail

# Download MiniMax-H3 GGUF components published by realrebelai.
#
# The denoiser and Qwen3-VL encoder are from realrebelai/MiniMax-H3_GGUFs.
# MiniMax publish the shared video/audio VAEs separately in Comfy-Org/MiniMax-H3.
#
# Default: FL2VA Q3 + Q2 encoder. This is the prompt-to-video pairing that
# fits a 32 GB GPU when used with text-encoder CPU placement/offloading.
#
# Usage:
#   ./scripts/download-model-minimax-h3-realrebelai.sh
#   ./scripts/download-model-minimax-h3-realrebelai.sh --variant ref2va --quant q4
#   MODELS_DIR=/models/h3 ./scripts/download-model-minimax-h3-realrebelai.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models/minimax-h3-realrebelai}"
VARIANT="fl2va"
QUANT="q3"
HF_REVISION="${HF_REVISION:-main}"
GGUF_REPO="realrebelai/MiniMax-H3_GGUFs"
VAE_REPO="Comfy-Org/MiniMax-H3"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--variant fl2va|ref2va] [--quant q3|q4] [--models-dir DIR]

Downloads a MiniMax-H3 GGUF denoiser, matching Qwen3-VL encoder, video VAE,
and audio VAE. Default is FL2VA Q3 with the Q2 encoder.

Environment:
  MODELS_DIR    destination directory
  HF_REVISION   Hugging Face revision (default: main)
  HF_TOKEN      optional Hugging Face token
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --quant)
            QUANT="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$VARIANT" in
    fl2va) DENOISER_PREFIX="MiniMax-H3-FL2VA" ;;
    ref2va) DENOISER_PREFIX="MiniMax-H3-REF2VA" ;;
    *)
        echo "--variant must be fl2va or ref2va (got: $VARIANT)" >&2
        exit 2
        ;;
esac

case "$QUANT" in
    q3)
        DENOISER_FILE="$DENOISER_PREFIX-Q3_K_M.gguf"
        DENOISER_SIZE=15577923360
        LLM_FILE="qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
        LLM_SIZE=8487968160
        ;;
    q4)
        DENOISER_FILE="$DENOISER_PREFIX-Q4_K_M.gguf"
        DENOISER_SIZE=19864208160
        LLM_FILE="qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
        LLM_SIZE=14576977888
        ;;
    *)
        echo "--quant must be q3 or q4 (got: $QUANT)" >&2
        exit 2
        ;;
esac

file_size() {
    if stat -f '%z' "$1" >/dev/null 2>&1; then
        stat -f '%z' "$1"
    else
        stat -c '%s' "$1"
    fi
}

download_verified() {
    local repo="$1"
    local relative_path="$2"
    local expected_size="$3"
    local destination="$MODELS_DIR/$relative_path"
    local partial="$destination.partial"
    local url="https://huggingface.co/$repo/resolve/$HF_REVISION/$relative_path"

    mkdir -p "$(dirname "$destination")"
    if [[ -f "$destination" ]] && [[ "$(file_size "$destination")" == "$expected_size" ]]; then
        echo "verified: $relative_path"
        return
    fi

    rm -f "$destination"
    echo "downloading: $relative_path"
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl -fL --progress-bar --retry 5 --retry-delay 3 --retry-connrefused -C - \
            -H "Authorization: Bearer $HF_TOKEN" -o "$partial" "$url"
    else
        curl -fL --progress-bar --retry 5 --retry-delay 3 --retry-connrefused -C - \
            -o "$partial" "$url"
    fi

    if [[ "$(file_size "$partial")" != "$expected_size" ]]; then
        echo "size verification failed for $relative_path" >&2
        rm -f "$partial"
        exit 1
    fi
    mv "$partial" "$destination"
    echo "verified: $relative_path"
}

download_verified "$GGUF_REPO" "$DENOISER_FILE" "$DENOISER_SIZE"
download_verified "$GGUF_REPO" "$LLM_FILE" "$LLM_SIZE"
download_verified "$VAE_REPO" "vae/minimax_h3_video_vae_fp16.safetensors" 5207808496
download_verified "$VAE_REPO" "vae/minimax_h3_audio_vae_fp32.safetensors" 605254808

cat <<EOF

MiniMax-H3 $VARIANT $QUANT model set is ready in: $MODELS_DIR

Example:
  sd-cli --mode vid_gen \\
    --diffusion-model "$MODELS_DIR/$DENOISER_FILE" \\
    --llm "$MODELS_DIR/$LLM_FILE" \\
    --vae "$MODELS_DIR/vae/minimax_h3_video_vae_fp16.safetensors" \\
    --audio-vae "$MODELS_DIR/vae/minimax_h3_audio_vae_fp32.safetensors" \\
    --prompt "A realistic cinematic coffee advertisement" \\
    --width 960 --height 544 --video-frames 124 --steps 8 \\
    --cfg-scale 1.0 --guidance 7.0 --backend te=cpu --diffusion-fa \\
    --offload-to-cpu --output output.webm
EOF
