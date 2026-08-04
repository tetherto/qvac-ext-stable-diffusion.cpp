#!/usr/bin/env python3
"""Convert an LTX-2.3 safetensors checkpoint to GGUF (video stream only).

LTX-2.3 (`Lightricks/LTX-2.3`, e.g. `ltx-2.3-22b-dev.safetensors`) is an
audio-video model. Per the qvac LTX-2 bounty, only the *video* stream is in
scope: the audio DiT, audio-video cross-attention, audio VAE and vocoder are
dropped. The Gemma-3-12B text encoder is NOT in this checkpoint and is
converted separately with llama.cpp tooling (see `docs/ltx2.md`).

What is kept (video-only):
  - vae.*                                  (CausalVideoAutoencoder enc/dec + stats)
  - text_embedding_projection.video_*      (multi-layer Gemma feature aggregation)
  - model.diffusion_model.patchify_proj / proj_out / scale_shift_table
  - model.diffusion_model.adaln_single.*   (global timestep AdaLN)
  - model.diffusion_model.prompt_adaln_single.*
  - model.diffusion_model.video_embeddings_connector.*
  - model.diffusion_model.transformer_blocks.N.{attn1,attn2,ff,
        scale_shift_table,prompt_scale_shift_table}

What is dropped (out of scope / audio):
  - vocoder.*, audio_vae.*
  - *.audio_* , audio_adaln_single.*, audio_embeddings_connector.*
  - audio-video cross attention: *_to_video_attn, video_to_audio_attn,
        av_ca_* , *_a2v_* , *_v2a_* , scale_shift_table_a2v_*

The filtering/naming logic is pure-stdlib so `--dry-run` works without numpy/
safetensors/gguf installed (it reads only the safetensors JSON header). The
actual conversion path imports those libs lazily; install them with:
    pip install -r script/requirements-ltx2.txt

Examples:
    # Inspect what would be converted (no heavy deps, no full read):
    python script/convert_ltx2_to_gguf.py --src ltx-2.3-22b-dev.safetensors --dry-run

    # Produce an F16 GGUF (the M1 deliverable):
    python script/convert_ltx2_to_gguf.py --src ltx-2.3-22b-dev.safetensors \
        --dst ltx-2.3-22b-video-f16.gguf --type f16

    # Quantised checkpoints:
    python script/convert_ltx2_to_gguf.py --src ltx-2.3-22b-dev.safetensors \
        --dst ltx-2.3-22b-video-q8_0.gguf --type q8_0
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from typing import Iterator

# --------------------------------------------------------------------------
# Pure-stdlib filtering / naming (no third-party imports here on purpose)
# --------------------------------------------------------------------------

ARCH = "ltx2"

# Quantisation types this tool can emit. F16 is the M1 deliverable.
QUANT_TYPES = ("f16", "q8_0", "q5_1", "q4_0")

# Per-transformer-block leaf prefixes that belong to the audio coupling and are
# dropped for video-only inference.
_AUDIO_BLOCK_LEAVES = (
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "audio_prompt_scale_shift_table",
    "audio_scale_shift_table",
    "audio_to_video_attn",
    "video_to_audio_attn",
    "scale_shift_table_a2v_ca_audio",
    "scale_shift_table_a2v_ca_video",
)

_DM = "model.diffusion_model."
_BLOCK_RE = None  # compiled lazily


def _block_leaf(name: str) -> str | None:
    """Return the leaf path of a `transformer_blocks.N.<leaf>` tensor, else None."""
    global _BLOCK_RE
    if _BLOCK_RE is None:
        import re

        _BLOCK_RE = re.compile(re.escape(_DM) + r"transformer_blocks\.\d+\.(.*)")
    m = _BLOCK_RE.match(name)
    return m.group(1) if m else None


def is_video_tensor(name: str) -> bool:
    """True if `name` is part of the in-scope video-only model."""
    # Hard drops: audio modalities + vocoder.
    if name.startswith(("vocoder.", "audio_vae.")):
        return False
    # Any top-level audio component under the DiT namespace
    # (audio_adaln_single, audio_embeddings_connector, audio_patchify_proj,
    #  audio_proj_out, audio_prompt_adaln_single, audio_scale_shift_table, ...).
    if name.startswith(_DM + "audio_"):
        return False
    if name.startswith("text_embedding_projection.audio_"):
        return False
    # Audio-video cross-attention scaffolding (top-level adaln tables).
    if name.startswith(_DM + "av_ca_"):
        return False
    if "_a2v_" in name or "_v2a_" in name:
        return False
    # Per-block audio / cross-modal leaves.
    leaf = _block_leaf(name)
    if leaf is not None and leaf.startswith(_AUDIO_BLOCK_LEAVES):
        return False
    return True


def map_name(name: str) -> str:
    """Map an HF tensor name to the GGUF name expected by the C++ loader.

    Kept intentionally light: the DiT keeps its native `model.diffusion_model.*`
    names (same convention as Wan), and the VAE keeps `vae.*` (the C++
    name_conversion maps `vae.` -> `first_stage_model.`). LTX-specific blocks
    (connector, text projection) keep their names and are matched by the
    `ltx2.hpp` block tree.
    """
    return name


def should_quantize(name: str, shape: list[int], qtype: str) -> bool:
    """Only quantise large 2D DiT linear weights; keep everything else as F16.

    Norms/biases (1D), small modulation tables (F32), conv weights (5D) and the
    whole VAE stay in higher precision for quality and because block-quant
    formats require 2D row lengths divisible by the block size (32).
    """
    if qtype == "f16":
        return False
    if not name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    if name.startswith("vae."):  # keep VAE accurate; it is small (~0.7B)
        return False
    if ".norm" in name or "_norm" in name:
        return False
    # q4_0/q5_1/q8_0 all use a block size of 32 along the last dim.
    if shape[-1] % 32 != 0:
        return False
    return True


# --------------------------------------------------------------------------
# safetensors header (stdlib only)
# --------------------------------------------------------------------------


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, data_offset) reading only the JSON header."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def plan(header: dict) -> tuple[list[str], list[str]]:
    keep, drop = [], []
    for name in header:
        if name == "__metadata__":
            continue
        (keep if is_video_tensor(name) else drop).append(name)
    return sorted(keep), sorted(drop)


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _dtype_bytes(dt: str) -> int:
    return {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
            "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}.get(dt, 2)


def print_plan(header: dict, qtype: str) -> None:
    keep, drop = plan(header)
    keep_bytes = sum(_numel(header[k]["shape"]) * _dtype_bytes(header[k]["dtype"]) for k in keep)
    keep_params = sum(_numel(header[k]["shape"]) for k in keep)
    nq = sum(1 for k in keep if should_quantize(k, header[k]["shape"], qtype))
    print(f"arch: {ARCH}   target type: {qtype}")
    print(f"KEEP (video): {len(keep)} tensors, {keep_params/1e9:.2f}B params, "
          f"{keep_bytes/1e9:.2f} GB (source dtype)")
    print(f"  of which quantised to {qtype}: {nq}; remaining kept as F16/F32")
    print(f"DROP (audio/vocoder/av-cross): {len(drop)} tensors")
    meta = header.get("__metadata__", {})
    if "model_version" in meta:
        print(f"model_version: {meta['model_version']}")
    print("\nsample KEEP:")
    for k in keep[:8]:
        print(f"  + {k}  {header[k]['shape']} {header[k]['dtype']}")
    print("sample DROP:")
    for k in drop[:8]:
        print(f"  - {k}  {header[k]['shape']} {header[k]['dtype']}")


# --------------------------------------------------------------------------
# Conversion (lazy heavy imports)
# --------------------------------------------------------------------------


def parse_config(header: dict) -> dict:
    meta = header.get("__metadata__", {})
    cfg = {}
    if isinstance(meta, dict) and "config" in meta:
        try:
            cfg = json.loads(meta["config"])
        except (json.JSONDecodeError, TypeError):
            cfg = {}
    return cfg


def convert(src: str, dst: str, qtype: str, include_vae: bool) -> None:
    import numpy as np  # noqa: F401
    import gguf
    from safetensors import safe_open

    header, _ = read_safetensors_header(src)
    keep, drop = plan(header)
    if not include_vae:
        keep = [k for k in keep if not k.startswith("vae.")]
    cfg = parse_config(header)
    tcfg = cfg.get("transformer", {})

    writer = gguf.GGUFWriter(dst, ARCH)
    writer.add_name("LTX-2.3 video")
    writer.add_description("LTX-2.3 video stream (DiT + VideoVAE + connector), audio dropped")
    if tcfg:
        writer.add_uint32("ltx2.dit.num_layers", int(tcfg.get("num_layers", 48)))
        writer.add_uint32("ltx2.dit.num_heads", int(tcfg.get("num_attention_heads", 32)))
        writer.add_uint32("ltx2.dit.head_dim", int(tcfg.get("attention_head_dim", 128)))
        writer.add_uint32("ltx2.dit.in_channels", int(tcfg.get("in_channels", 128)))
        writer.add_uint32("ltx2.dit.caption_channels", int(tcfg.get("caption_channels", 3840)))
        writer.add_uint32("ltx2.dit.cross_attention_dim", int(tcfg.get("cross_attention_dim", 4096)))
        writer.add_uint32("ltx2.connector.num_layers", int(tcfg.get("connector_num_layers", 8)))
        writer.add_uint32("ltx2.connector.num_registers", int(tcfg.get("connector_num_learnable_registers", 128)))
        writer.add_float32("ltx2.rope.theta", float(tcfg.get("positional_embedding_theta", 10000.0)))

    qmap = {
        "f16": gguf.GGMLQuantizationType.F16,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q5_1": gguf.GGMLQuantizationType.Q5_1,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
    }

    n = 0
    with safe_open(src, framework="numpy") as st:
        for name in keep:
            arr = st.get_tensor(name)
            # bf16 arrives as uint16-backed; promote to float32 for processing.
            if arr.dtype == np.uint16 or str(arr.dtype) == "bfloat16":
                arr = _bf16_to_f32(arr)
            else:
                arr = arr.astype(np.float32, copy=False)

            out_name = map_name(name)
            if should_quantize(name, list(arr.shape), qtype):
                data = gguf.quants.quantize(arr, qmap[qtype])
                writer.add_tensor(out_name, data, raw_dtype=qmap[qtype])
            else:
                writer.add_tensor(out_name, arr.astype(np.float16))
            n += 1
            if n % 100 == 0:
                print(f"  ... {n}/{len(keep)} tensors", file=sys.stderr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n} tensors to {dst}")


def _bf16_to_f32(arr):
    import numpy as np

    u16 = arr.view(np.uint16) if arr.dtype != np.uint16 else arr
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


# --------------------------------------------------------------------------


def self_test() -> int:
    """Validate the keep/drop filter on representative names (no checkpoint)."""
    keep = [
        _DM + "patchify_proj.weight",
        _DM + "proj_out.weight",
        _DM + "scale_shift_table",
        _DM + "adaln_single.linear.weight",
        _DM + "video_embeddings_connector.learnable_registers",
        _DM + "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight",
        _DM + "transformer_blocks.0.attn1.to_q.weight",
        _DM + "transformer_blocks.0.attn2.to_k.weight",
        _DM + "transformer_blocks.0.ff.net.0.proj.weight",
        _DM + "transformer_blocks.0.scale_shift_table",
        _DM + "transformer_blocks.0.prompt_scale_shift_table",
        "vae.decoder.conv_in.weight",
    ]
    drop = [
        "vocoder.conv_pre.weight",
        "audio_vae.encoder.conv_in.weight",
        _DM + "audio_patchify_proj.weight",
        _DM + "audio_scale_shift_table",
        _DM + "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight",
        "text_embedding_projection.audio_aggregate_embed.weight",
        _DM + "av_ca_scale_shift_table",
        _DM + "transformer_blocks.0.audio_attn1.to_q.weight",
        _DM + "transformer_blocks.0.audio_to_video_attn.to_q.weight",
        _DM + "transformer_blocks.0.scale_shift_table_a2v_ca_audio",
    ]
    ok = True
    for n in keep:
        if not is_video_tensor(n):
            print(f"SELF-TEST FAIL: should KEEP but dropped: {n}")
            ok = False
    for n in drop:
        if is_video_tensor(n):
            print(f"SELF-TEST FAIL: should DROP but kept: {n}")
            ok = False
    print("self-test PASSED" if ok else "self-test FAILED")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Convert LTX-2.3 safetensors -> GGUF (video only)")
    ap.add_argument("--src", help="input .safetensors checkpoint")
    ap.add_argument("--dst", help="output .gguf path (required unless --dry-run)")
    ap.add_argument("--type", default="f16", choices=QUANT_TYPES, help="output tensor type")
    ap.add_argument("--no-vae", action="store_true", help="exclude VAE tensors from the GGUF")
    ap.add_argument("--dry-run", action="store_true",
                    help="only read the header and print the keep/drop plan")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the keep/drop filter on built-in names (no checkpoint needed)")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.src:
        ap.error("--src is required unless --self-test is given")

    header, _ = read_safetensors_header(args.src)

    if args.dry_run:
        print_plan(header, args.type)
        return 0

    if not args.dst:
        ap.error("--dst is required unless --dry-run is given")
    convert(args.src, args.dst, args.type, include_vae=not args.no_vae)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
