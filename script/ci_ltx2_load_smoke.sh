#!/usr/bin/env bash
# M1 smoke test: build a tiny synthetic LTX-2 video-DiT GGUF and verify it is
# detected and fully bound on CPU by sd-cli. This exercises the exact load path
# used for the real 46 GB checkpoint without downloading any weights.
#
# Usage: script/ci_ltx2_load_smoke.sh [path-to-sd-cli]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD_CLI="${1:-$ROOT/build/bin/sd-cli}"
GGUF="$(mktemp -t ltx2_tiny.XXXXXX.gguf)"
LOG="$(mktemp -t ltx2_smoke.XXXXXX.log)"
PY="${PYTHON:-python3}"

cleanup() { rm -f "$GGUF" "$LOG"; }
trap cleanup EXIT

if [ ! -x "$SD_CLI" ]; then
    echo "FAIL: sd-cli not found at $SD_CLI" >&2
    exit 1
fi

echo "==> generating synthetic LTX-2 DiT GGUF"
"$PY" "$ROOT/script/make_synthetic_ltx2_gguf.py" --out "$GGUF"

echo "==> loading via sd-cli (generation is expected to stop: M1 is load-only)"
# sd-cli returns non-zero because M1 has no text encoder yet; we assert on the
# load markers in the log instead of the exit code.
"$SD_CLI" -M vid_gen --diffusion-model "$GGUF" -p "smoke" \
    --steps 1 --video-frames 1 -W 32 -H 32 -o /tmp/ltx2_smoke_out >"$LOG" 2>&1 || true

cat "$LOG"

echo "==> checking load markers"
fail=0
for marker in \
    "Version: LTX-2" \
    "LTX-2 DiT: num_layers=" \
    "loading tensors completed" \
    "total params memory size"; do
    if ! grep -qF "$marker" "$LOG"; then
        echo "FAIL: missing expected log marker: '$marker'" >&2
        fail=1
    fi
done

if grep -qiE "load tensors from model loader failed|get_sd_version failed" "$LOG"; then
    echo "FAIL: tensor load reported failure" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "LTX-2 load smoke test FAILED" >&2
    exit 1
fi

echo "LTX-2 load smoke test PASSED"
