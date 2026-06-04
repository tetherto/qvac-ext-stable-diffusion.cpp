#!/usr/bin/env bash
# M2 smoke test: confirm LTX-2 T2V and I2V produce a video on the current build.
# Portable — override paths via env vars (defaults preserve the original setup):
#   LTX2_BIN     path to the sd-cli binary      (default: $LTX2_BUILD/bin/sd-cli)
#   LTX2_BUILD   build dir holding bin/sd-cli   (default: build)
#   LTX2_MODELS  dir with the LTX-2 model files (default: /home/user/ltx2-models)
#   LTX2_OUT     output dir for generated video (default: /tmp)
#   LTX2_INIT_IMAGE  init frame for the I2V run (default: $LTX2_OUT/test_init.png)
#
# For frame-level parity across CPU/Vulkan/Metal, use script/ltx2_parity.py.
set -euo pipefail

BIN="${LTX2_BIN:-${LTX2_BUILD:-build}/bin/sd-cli}"
MODELS="${LTX2_MODELS:-/home/user/ltx2-models}"
OUT_DIR="${LTX2_OUT:-/tmp}"
INIT_IMAGE="${LTX2_INIT_IMAGE:-$OUT_DIR/test_init.png}"

DIFFUSION="$MODELS/distilled/ltx-2.3-22b-distilled-Q4_0.gguf"
VAE="$MODELS/vae/ltx-2.3-22b-distilled_video_vae.safetensors"
LLM="$MODELS/gemma-3-12b-it-Q4_K_S.gguf"
CONN="$MODELS/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"

[ -x "$BIN" ] || command -v "$BIN" >/dev/null 2>&1 \
  || { echo "ERROR: sd-cli not found at '$BIN' (set LTX2_BIN or LTX2_BUILD)"; exit 1; }
for f in "$DIFFUSION" "$VAE" "$LLM" "$CONN"; do
  [ -f "$f" ] || { echo "ERROR: missing model '$f' (set LTX2_MODELS)"; exit 1; }
done

COMMON=(-M vid_gen
  --diffusion-model "$DIFFUSION" --vae "$VAE" --llm "$LLM"
  --embeddings-connectors "$CONN"
  -p "a lovely cat sitting on a sunny windowsill, high quality"
  --cfg-scale 3.5 --sampling-method euler
  -W 512 -H 288 --video-frames 25 --fps 8
  --diffusion-fa --offload-to-cpu
  -s 42)

mkdir -p "$OUT_DIR"

echo "=== M2 T2V test ==="
"$BIN" "${COMMON[@]}" -o "$OUT_DIR/test_t2v.mp4"

echo ""
echo "=== M2 I2V test ==="
[ -f "$INIT_IMAGE" ] || { echo "ERROR: init image '$INIT_IMAGE' not found (set LTX2_INIT_IMAGE)"; exit 1; }
"$BIN" "${COMMON[@]}" -i "$INIT_IMAGE" -o "$OUT_DIR/test_i2v.mp4"

echo ""
echo "=== Results ==="
# sd-cli appends .avi to the output path regardless of the extension given
ls -lh "$OUT_DIR/test_t2v.mp4.avi" "$OUT_DIR/test_i2v.mp4.avi"
