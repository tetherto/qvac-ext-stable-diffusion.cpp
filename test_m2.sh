#!/usr/bin/env bash
set -euo pipefail

BIN=/home/user/ltx2-bounty/build/bin/sd-cli
MODELS=/home/user/ltx2-models

DIFFUSION="$MODELS/distilled/ltx-2.3-22b-distilled-Q4_0.gguf"
VAE="$MODELS/vae/ltx-2.3-22b-distilled_video_vae.safetensors"
LLM="$MODELS/gemma-3-12b-it-Q4_K_S.gguf"
CONN="$MODELS/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"

echo "=== M2 T2V test ==="
$BIN -M vid_gen \
  --diffusion-model "$DIFFUSION" \
  --vae "$VAE" \
  --llm "$LLM" \
  --embeddings-connectors "$CONN" \
  -p "a lovely cat sitting on a sunny windowsill, high quality" \
  --cfg-scale 3.5 --sampling-method euler \
  -W 512 -H 288 --video-frames 25 --fps 8 \
  --diffusion-fa --offload-to-cpu \
  -s 42 -o /tmp/test_t2v.mp4

echo ""
echo "=== M2 I2V test ==="
$BIN -M vid_gen \
  --diffusion-model "$DIFFUSION" \
  --vae "$VAE" \
  --llm "$LLM" \
  --embeddings-connectors "$CONN" \
  -p "a lovely cat sitting on a sunny windowsill, high quality" \
  --cfg-scale 3.5 --sampling-method euler \
  -W 512 -H 288 --video-frames 25 --fps 8 \
  --diffusion-fa --offload-to-cpu \
  -i /tmp/test_init.png \
  -s 42 -o /tmp/test_i2v.mp4

echo ""
echo "=== Results ==="
# sd-cli appends .avi to the output path regardless of the extension given
ls -lh /tmp/test_t2v.mp4.avi /tmp/test_i2v.mp4.avi
