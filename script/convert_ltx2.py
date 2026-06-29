#!/usr/bin/env python3
"""
Convert LTX-2.3 safetensors weights to GGUF format.

Usage:
    pip install numpy gguf
    python script/convert_ltx2.py --model /path/to/ltx-2.3-22b-dev.safetensors \
        --output ltx-2.3-22b-dev.gguf --type q4_0

Supported types: f16, q4_0, q5_1, q8_0
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import numpy as np
    import gguf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy gguf")
    sys.exit(1)


QUANT_TYPES = {
    "f16":  gguf.GGMLQuantizationType.F16,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q5_1": gguf.GGMLQuantizationType.Q5_1,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}

# Tensors kept at F16 regardless of requested quantisation (norms, biases, embeddings).
KEEP_F16_PATTERNS = [
    "norm",
    "bias",
    "embed",
    "connector",
    "patch_embed",
    "time_embed",
    "pos_embed",
    ".proj_out.",
    "final_layer",
]

# Safetensors dtype string → numpy dtype.
# BF16 has no native numpy type; we load it as uint16 and convert via bf16_to_fp32.
_ST_DTYPE: dict = {
    "F32":  np.float32,
    "F16":  np.float16,
    "BF16": np.uint16,
    "I8":   np.int8,
    "I16":  np.int16,
    "I32":  np.int32,
    "I64":  np.int64,
    "U8":   np.uint8,
    "U16":  np.uint16,
    "U32":  np.uint32,
    "U64":  np.uint64,
    "F64":  np.float64,
}


def should_keep_f16(name: str) -> bool:
    return any(p in name for p in KEEP_F16_PATTERNS)


def bf16_to_fp32(data: np.ndarray) -> np.ndarray:
    # BF16 is float32 with the lower 16 bits zeroed — copy bytes into the high
    # 16 bits of uint32 words, then reinterpret as float32.
    # Flatten to 1D first so .view(uint8) gives a flat byte array.
    flat = np.ascontiguousarray(data).ravel()
    buf = np.zeros(flat.size * 4, dtype=np.uint8)
    src = flat.view(np.uint8)   # shape (flat.size * 2,)
    buf[2::4] = src[0::2]       # BF16 low byte  → float32 byte 2
    buf[3::4] = src[1::2]       # BF16 high byte → float32 byte 3
    return np.frombuffer(buf, dtype=np.float32).reshape(data.shape)


def _iter_safetensors(path: Path):
    """Yield (name, array, dtype_str) for every tensor in a safetensors file."""
    with open(path, "rb") as fh:
        hdr_size = struct.unpack("<Q", fh.read(8))[0]
        metadata = json.loads(fh.read(hdr_size).decode())
        data_base = 8 + hdr_size
        for name, info in metadata.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape     = info["shape"]
            lo, hi    = info["data_offsets"]
            expected  = int(np.prod(shape)) * np.dtype(_ST_DTYPE[dtype_str]).itemsize
            actual    = hi - lo
            if actual != expected:
                raise ValueError(
                    f"Tensor '{name}': expected {expected} bytes for shape {shape} "
                    f"dtype {dtype_str}, got {actual} in file"
                )
            fh.seek(data_base + lo)
            raw = np.frombuffer(fh.read(actual), dtype=_ST_DTYPE[dtype_str])
            yield name, raw.reshape(shape), dtype_str


def convert_tensor(data: np.ndarray, qtype: gguf.GGMLQuantizationType,
                   name: str, original_dtype: str = ""):
    if original_dtype == "BF16":
        fp32 = bf16_to_fp32(data)
    else:
        fp32 = data.astype(np.float32)

    if qtype == gguf.GGMLQuantizationType.F16 or should_keep_f16(name):
        return fp32.astype(np.float16), gguf.GGMLQuantizationType.F16

    return gguf.quants.quantize(fp32, qtype), qtype


def convert(model_path: Path, output_path: Path, qtype_str: str, vae_path=None):
    qtype = QUANT_TYPES.get(qtype_str)
    if qtype is None:
        print(f"Unknown type '{qtype_str}'. Choose from: {', '.join(QUANT_TYPES)}")
        sys.exit(1)

    writer = gguf.GGUFWriter(str(output_path), arch="ltx2")
    writer.add_string("general.name", model_path.stem)
    writer.add_string("general.quantization_version", qtype_str)

    sources = [(model_path, "")]
    if vae_path:
        sources.append((vae_path, "vae."))

    for src_path, prefix in sources:
        print(f"Loading {src_path} ...")
        tensors = list(_iter_safetensors(src_path))
        for i, (key, tensor, dtype_str) in enumerate(tensors):
            scoped_name = prefix + key
            data, used_qtype = convert_tensor(tensor, qtype, scoped_name, dtype_str)
            writer.add_tensor(scoped_name, data, raw_dtype=used_qtype)
            if (i + 1) % 50 == 0 or (i + 1) == len(tensors):
                print(f"  [{i+1}/{len(tensors)}] {scoped_name} -> {used_qtype.name}")

    print(f"Writing {output_path} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LTX-2.3 safetensors to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to diffusion model safetensors file")
    parser.add_argument("--vae", type=Path, default=None,
                        help="Optional: path to VAE safetensors to bundle into output")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .gguf file path")
    parser.add_argument("--type", dest="qtype", default="q8_0",
                        choices=list(QUANT_TYPES.keys()),
                        help="Quantisation type (default: q8_0)")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)
    if args.vae and not args.vae.exists():
        print(f"VAE not found: {args.vae}")
        sys.exit(1)

    convert(args.model, args.output, args.qtype, args.vae)


if __name__ == "__main__":
    main()
