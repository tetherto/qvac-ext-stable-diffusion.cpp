#!/usr/bin/env bash
# M2 smoke test: build a tiny synthetic LTX-2 stack (video DiT + Gemma-3 text
# encoder + text projection) and run a full end-to-end CPU generate, verifying
# the Gemma encoder, the DiT denoising loop, the LinearQuadratic scheduler and
# the Video-VAE decode all execute and produce a video file. This exercises the
# exact pipeline used for the real ~46 GB checkpoint without downloading any
# weights.
#
# Usage: script/ci_ltx2_load_smoke.sh [path-to-sd-cli]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD_CLI="${1:-$ROOT/build/bin/sd-cli}"
GGUF="$(mktemp -t ltx2_tiny.XXXXXX.gguf)"
GEMMA="$GGUF.gemma.gguf"
OUT="$(mktemp -t ltx2_smoke_out.XXXXXX)"
LOG="$(mktemp -t ltx2_smoke.XXXXXX.log)"
PY="${PYTHON:-python3}"

cleanup() { rm -f "$GGUF" "$GEMMA" "$LOG" "$OUT" "$OUT.avi" "$OUT.png"; }
trap cleanup EXIT

if [ ! -x "$SD_CLI" ]; then
    echo "FAIL: sd-cli not found at $SD_CLI" >&2
    exit 1
fi

echo "==> generating synthetic LTX-2 DiT + Gemma-3 GGUFs"
"$PY" "$ROOT/script/make_synthetic_ltx2_gguf.py" --out "$GGUF" --gemma-out "$GEMMA"

echo "==> running end-to-end T2V generate on CPU"
"$SD_CLI" -M vid_gen --diffusion-model "$GGUF" --llm "$GEMMA" -p "smoke test" \
    --steps 4 --video-frames 9 -W 32 -H 32 --scheduler linear_quadratic \
    -o "$OUT" >"$LOG" 2>&1

cat "$LOG"

echo "==> checking generate markers"
fail=0
for marker in \
    "Version: LTX-2" \
    "Gemma-3 encoder: num_layers=" \
    "LTX-2 DiT: num_layers=" \
    "get_sigmas with LinearQuadratic scheduler" \
    "get_learned_condition completed" \
    "sampling completed" \
    "decode_first_stage completed" \
    "save result"; do
    if ! grep -qF "$marker" "$LOG"; then
        echo "FAIL: missing expected log marker: '$marker'" >&2
        fail=1
    fi
done

if grep -qiE "load tensors from model loader failed|get_sd_version failed|generate failed" "$LOG"; then
    echo "FAIL: generation reported failure" >&2
    fail=1
fi

if [ ! -s "$OUT.avi" ]; then
    echo "FAIL: expected video output '$OUT.avi' was not written" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "LTX-2 generate smoke test FAILED" >&2
    exit 1
fi

echo "LTX-2 generate smoke test PASSED"
