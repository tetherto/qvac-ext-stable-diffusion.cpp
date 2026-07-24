#!/usr/bin/env python3
"""Remap a raw LAM-A2E GGUF (pytorch/backbone names) into lam-audio2exp dialect."""

from __future__ import annotations

import argparse
import os

import numpy as np
from gguf import GGUFReader, GGUFWriter

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

FE_KERNELS = [10, 3, 3, 3, 3, 2, 2]
FE_STRIDES = [5, 2, 2, 2, 2, 2, 2]
SKIP_PREFIXES = (
    "audio_encoder.lm_head.",
    "identity_encoder.grus.",
)
SKIP_KEYS = ("audio_encoder.masked_spec_embed",)


def build_name_map():
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


def strip_prefix(name: str) -> str:
    if name.startswith("module."):
        name = name[7:]
    if name.startswith("backbone."):
        name = name[9:]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    args = parser.parse_args()

    reader = GGUFReader(args.inp)
    state = {}
    for tensor in reader.tensors:
        key = strip_prefix(tensor.name)
        if key in SKIP_KEYS or key.startswith(SKIP_PREFIXES):
            continue
        state[key] = np.array(tensor.data, dtype=np.float32, copy=True)

    g = state.pop("audio_encoder.encoder.pos_conv_embed.conv.weight_g")
    v = state.pop("audio_encoder.encoder.pos_conv_embed.conv.weight_v")
    if g.shape != (1, 1, 128) or v.shape != (768, 48, 128):
        raise RuntimeError(f"unexpected pos-conv shapes g={g.shape} v={v.shape}")
    norm = np.linalg.norm(v, axis=(0, 1), keepdims=True)
    folded = (g * v / np.maximum(norm, 1e-12)).astype(np.float32, copy=True)
    state["audio_encoder.encoder.pos_conv_embed.conv.weight"] = folded

    name_map = build_name_map()
    unmapped = sorted(set(state) - set(name_map))
    if unmapped:
        raise RuntimeError(f"unmapped tensors ({len(unmapped)}): {unmapped[:30]}")
    missing = sorted(set(name_map) - set(state))
    if missing:
        raise RuntimeError(f"missing tensors: {missing}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    total = 0
    for key in sorted(state, key=lambda k: name_map[k][0]):
        gguf_name, is_matmul = name_map[key]
        arr = np.ascontiguousarray(state[key])
        if args.dtype == "f16" and is_matmul:
            arr = arr.astype(np.float16)
        writer.add_tensor(gguf_name, arr)
        total += arr.nbytes
        print(f"  {gguf_name}: {list(arr.shape)} {arr.dtype}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
