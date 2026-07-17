#!/usr/bin/env bash
# Validation / regression smoke test for ABot-World support.
#
# ABot-World-0-5B-LF is a causal, interactive derivative of Wan2.2-TI2V-5B. This
# script verifies:
#   1. an ABot GGUF is detected as "ABot-World" and all tensors load (incl. the
#      act_control_adapter action-conditioning block);
#   2. the batch generate_video() path rejects it with the explanatory error
#      (the causal interactive session is a separate, not-yet-landed feature);
#   3. (optional) a stock Wan model still detects + generates unchanged, proving
#      no regression from the sd_version_is_wan_ti2v_family() refactor.
#
# Usage:
#   ABOT_DIT=/path/abot-dit-q8_0.gguf ABOT_VAE=/path/wan2.2_vae.safetensors \
#   ABOT_T5=/path/umt5-xxl.(pth|gguf|safetensors) \
#   [WAN_DIT=... WAN_VAE=... WAN_T5=...] \
#   SD_CLI=./build/bin/sd-cli script/validate_abot_world.sh
set -euo pipefail

SD_CLI="${SD_CLI:-./build/bin/sd-cli}"
: "${ABOT_DIT:?set ABOT_DIT}"; : "${ABOT_VAE:?set ABOT_VAE}"; : "${ABOT_T5:?set ABOT_T5}"

log=$(mktemp)
fail() { echo "FAIL: $1"; echo "--- log tail ---"; tail -30 "$log"; exit 1; }

echo "== ABot-World detection + load + guarded rejection =="
set +e
"$SD_CLI" -M vid_gen --diffusion-model "$ABOT_DIT" --vae "$ABOT_VAE" \
  --t5xxl "$ABOT_T5" -p "a coastal street at dusk" -W 480 -H 832 \
  --video-frames 9 -v >"$log" 2>&1
set -e

grep -q "ABot-World-5B"                                  "$log" || fail "not detected as ABot-World-5B"
grep -q "loading tensors completed"                      "$log" || fail "tensor load did not complete"
grep -qi "wrong shape\|not in model file"                "$log" && fail "tensor shape/name mismatch on load"
grep -q "not supported by batch generate_video"          "$log" || fail "missing the guarded-rejection error"
echo "PASS: ABot-World detected, all tensors loaded, batch path correctly rejected."

if [[ -n "${WAN_DIT:-}" ]]; then
  echo "== Regression: stock Wan still generates =="
  out=$(mktemp -u).png
  "$SD_CLI" -M vid_gen --diffusion-model "$WAN_DIT" --vae "${WAN_VAE:?}" \
    --t5xxl "${WAN_T5:?}" -p "a bird flying" -W 480 -H 832 --video-frames 9 \
    -o "$out" -v >"$log" 2>&1 || fail "stock Wan generation failed (regression!)"
  grep -qi "not supported by batch generate_video" "$log" && fail "stock Wan wrongly rejected (regression!)"
  echo "PASS: stock Wan generation unaffected."
fi

echo "ALL CHECKS PASSED"
