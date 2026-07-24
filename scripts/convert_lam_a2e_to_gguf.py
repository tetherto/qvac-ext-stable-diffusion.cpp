#!/usr/bin/env python3
"""LAM Audio2Expression PyTorch checkpoint → GGUF converter.

Reads the upstream ``lam_audio2exp_streaming.tar`` checkpoint (Apache-2.0,
https://github.com/aigc3d/LAM_Audio2Expression) and emits a single GGUF
consumed by the isolated LAM-A2E target in qvac-ext-stable-diffusion.cpp.

Layout notes:

- ``general.architecture = "lam-audio2exp"`` — selected by the addon's
  model factory.
- PyTorch conv1d weights are (C_out, C_in, K) row-major, which lands as
  ggml ne = [K, C_in, C_out] on a straight copy — exactly what
  ``ggml_conv_1d`` expects. No transposes are performed anywhere; the
  converter is a 1:1 map of the state dict.
- The positional-conv weight-norm (weight_g/weight_v, dim=2) is folded
  into a plain conv weight at conversion time.
- Inference-unused tensors (``lm_head``, ``identity_encoder.grus``,
  ``masked_spec_embed``) are dropped.
- ``--dtype f32`` (default) keeps everything float32; ``--dtype f16``
  stores matmul/conv weights as f16 and keeps norms + biases f32.

Usage:
    python3 scripts/convert_lam_a2e_to_gguf.py \
        --checkpoint pretrained_models/lam_audio2exp_streaming.tar \
        --out lam-audio2exp-f32.gguf --dtype f32
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from gguf import GGUFWriter

ARCH = "lam-audio2exp"

ARKIT_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
]

# wav2vec2-base feature extractor geometry (configs/wav2vec2_config.json)
FE_KERNELS = [10, 3, 3, 3, 3, 2, 2]
FE_STRIDES = [5, 2, 2, 2, 2, 2, 2]

SKIP_PREFIXES = (
    "audio_encoder.lm_head.",
    "identity_encoder.grus.",
)
SKIP_KEYS = ("audio_encoder.masked_spec_embed",)


def load_state(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {}
    for key, value in ckpt["state_dict"].items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("backbone."):
            key = key[9:]
        if key in SKIP_KEYS or key.startswith(SKIP_PREFIXES):
            continue
        state[key] = value
    return state


def fold_pos_conv_weight_norm(state):
    """weight = g * v / ||v||, norm over dims (0,1) per kernel position (dim=2)."""
    g = state.pop("audio_encoder.encoder.pos_conv_embed.conv.weight_g")  # (1,1,128)
    v = state.pop("audio_encoder.encoder.pos_conv_embed.conv.weight_v")  # (768,48,128)
    norm = v.norm(p=2, dim=(0, 1), keepdim=True)
    state["audio_encoder.encoder.pos_conv_embed.conv.weight"] = g * v / norm
    return state


def build_name_map():
    """checkpoint key -> (gguf name, is_matmul_weight)."""
    m = {}
    for i in range(7):
        m[f"audio_encoder.feature_extractor.conv_layers.{i}.conv.weight"] = (f"fe.conv{i}.weight", True)
    m["audio_encoder.feature_extractor.conv_layers.0.layer_norm.weight"] = ("fe.gn.weight", False)
    m["audio_encoder.feature_extractor.conv_layers.0.layer_norm.bias"] = ("fe.gn.bias", False)

    m["audio_encoder.feature_projection.layer_norm.weight"] = ("fp.ln.weight", False)
    m["audio_encoder.feature_projection.layer_norm.bias"] = ("fp.ln.bias", False)
    m["audio_encoder.feature_projection.projection.weight"] = ("fp.proj.weight", True)
    m["audio_encoder.feature_projection.projection.bias"] = ("fp.proj.bias", False)

    m["audio_encoder.encoder.pos_conv_embed.conv.weight"] = ("enc.pos_conv.weight", True)
    m["audio_encoder.encoder.pos_conv_embed.conv.bias"] = ("enc.pos_conv.bias", False)
    m["audio_encoder.encoder.layer_norm.weight"] = ("enc.ln.weight", False)
    m["audio_encoder.encoder.layer_norm.bias"] = ("enc.ln.bias", False)

    for i in range(12):
        src = f"audio_encoder.encoder.layers.{i}"
        dst = f"enc.blk{i}"
        for proj in ("q", "k", "v"):
            m[f"{src}.attention.{proj}_proj.weight"] = (f"{dst}.attn_{proj}.weight", True)
            m[f"{src}.attention.{proj}_proj.bias"] = (f"{dst}.attn_{proj}.bias", False)
        m[f"{src}.attention.out_proj.weight"] = (f"{dst}.attn_o.weight", True)
        m[f"{src}.attention.out_proj.bias"] = (f"{dst}.attn_o.bias", False)
        m[f"{src}.layer_norm.weight"] = (f"{dst}.ln1.weight", False)
        m[f"{src}.layer_norm.bias"] = (f"{dst}.ln1.bias", False)
        m[f"{src}.feed_forward.intermediate_dense.weight"] = (f"{dst}.ffn_up.weight", True)
        m[f"{src}.feed_forward.intermediate_dense.bias"] = (f"{dst}.ffn_up.bias", False)
        m[f"{src}.feed_forward.output_dense.weight"] = (f"{dst}.ffn_down.weight", True)
        m[f"{src}.feed_forward.output_dense.bias"] = (f"{dst}.ffn_down.bias", False)
        m[f"{src}.final_layer_norm.weight"] = (f"{dst}.ln2.weight", False)
        m[f"{src}.final_layer_norm.bias"] = (f"{dst}.ln2.bias", False)

    m["feature_projection.weight"] = ("head.proj.weight", True)
    m["feature_projection.bias"] = ("head.proj.bias", False)
    m["identity_encoder.id_mlp.weight"] = ("head.id_mlp.weight", True)
    m["identity_encoder.id_mlp.bias"] = ("head.id_mlp.bias", False)
    for i in range(3):
        src = f"identity_encoder.first_net.conv_layers.{i}"
        m[f"{src}.conv.weight"] = (f"head.first{i}.conv.weight", True)
        m[f"{src}.conv.bias"] = (f"head.first{i}.conv.bias", False)
        m[f"{src}.norm.weight"] = (f"head.first{i}.ln.weight", False)
        m[f"{src}.norm.bias"] = (f"head.first{i}.ln.bias", False)
    m["identity_encoder.first_net.conv_layers.0.residual_layer.0.weight"] = ("head.first0.res.weight", True)
    m["identity_encoder.first_net.conv_layers.0.residual_layer.0.bias"] = ("head.first0.res.bias", False)
    for i in range(3):
        m[f"decoder.0.{i}.conv.weight"] = (f"head.dec{i}.conv.weight", True)
        m[f"decoder.0.{i}.conv.bias"] = (f"head.dec{i}.conv.bias", False)
        m[f"decoder.0.{i}.norm.weight"] = (f"head.dec{i}.ln.weight", False)
        m[f"decoder.0.{i}.norm.bias"] = (f"head.dec{i}.ln.bias", False)
    m["output_proj.weight"] = ("head.out.weight", True)
    m["output_proj.bias"] = ("head.out.bias", False)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    args = parser.parse_args()

    state = fold_pos_conv_weight_norm(load_state(args.checkpoint))
    name_map = build_name_map()

    unmapped = sorted(set(state) - set(name_map))
    if unmapped:
        raise RuntimeError(f"unmapped checkpoint tensors: {unmapped}")
    missing = sorted(set(name_map) - set(state))
    if missing:
        raise RuntimeError(f"expected checkpoint tensors not found: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = GGUFWriter(args.out, ARCH)
    writer.add_name("LAM Audio2Expression (streaming)")
    writer.add_string(f"{ARCH}.dtype", args.dtype)
    writer.add_uint32(f"{ARCH}.sample_rate", 16000)
    writer.add_uint32(f"{ARCH}.fps", 30)
    writer.add_uint32(f"{ARCH}.n_coeffs", 52)
    writer.add_uint32(f"{ARCH}.n_identity", 12)
    writer.add_uint32(f"{ARCH}.identity_feat_dim", 64)
    writer.add_uint32(f"{ARCH}.hidden_dim", 512)
    writer.add_uint32(f"{ARCH}.window_frames", 64)
    writer.add_float32(f"{ARCH}.layer_norm_eps", 1e-5)
    writer.add_uint32(f"{ARCH}.enc.n_layers", 12)
    writer.add_uint32(f"{ARCH}.enc.n_heads", 12)
    writer.add_uint32(f"{ARCH}.enc.hidden", 768)
    writer.add_uint32(f"{ARCH}.enc.ffn", 3072)
    writer.add_uint32(f"{ARCH}.enc.pos_conv_kernel", 128)
    writer.add_uint32(f"{ARCH}.enc.pos_conv_groups", 16)
    writer.add_array(f"{ARCH}.fe.kernels", FE_KERNELS)
    writer.add_array(f"{ARCH}.fe.strides", FE_STRIDES)
    writer.add_array(f"{ARCH}.coeff_names", ARKIT_BLENDSHAPES)

    total_bytes = 0
    for key in sorted(state, key=lambda k: name_map[k][0]):
        gguf_name, is_matmul = name_map[key]
        arr = state[key].detach().cpu().float().numpy()
        arr = np.ascontiguousarray(arr)
        if args.dtype == "f16" and is_matmul:
            arr = arr.astype(np.float16)
        writer.add_tensor(gguf_name, arr)
        total_bytes += arr.nbytes
        print(f"  {gguf_name}: {list(arr.shape)} {arr.dtype}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({total_bytes / 1e6:.1f} MB tensor data, dtype={args.dtype})")


if __name__ == "__main__":
    main()
