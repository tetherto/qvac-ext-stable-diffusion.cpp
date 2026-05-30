#!/usr/bin/env python3
"""
Convert LTX-2.3 safetensors weights to GGUF format.

Usage:
    pip install safetensors numpy gguf
    python script/convert_ltx2.py --model /path/to/ltx-2.3-22b-dev.safetensors \
        --output ltx-2.3-22b-dev.gguf --type q4_0

Supported types: f16, q4_0, q5_1, q8_0
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    from safetensors import safe_open
    import gguf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install safetensors numpy gguf")
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


def should_keep_f16(name: str) -> bool:
    return any(p in name for p in KEEP_F16_PATTERNS)


def bf16_to_fp32(data: np.ndarray) -> np.ndarray:
    # BF16 is float32 with the lower 16 bits zeroed — copy bytes into the high
    # 16 bits of uint32 words, then reinterpret as float32.
    buf = np.zeros(data.size * 4, dtype=np.uint8)
    src = data.view(np.uint8)
    buf[2::4] = src[0::2]
    buf[3::4] = src[1::2]
    return np.frombuffer(buf, dtype=np.float32).reshape(data.shape)


def convert_tensor(data: np.ndarray, qtype: gguf.GGMLQuantizationType, name: str):
    if str(data.dtype) == "bfloat16":
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
    writer.add_string("general.architecture", "ltx2")
    writer.add_string("general.name", model_path.stem)
    writer.add_string("general.quantization_version", qtype_str)

    sources = [(model_path, "")]
    if vae_path:
        sources.append((vae_path, "vae."))

    for src_path, prefix in sources:
        print(f"Loading {src_path} ...")
        with safe_open(str(src_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for i, key in enumerate(keys):
                tensor = f.get_tensor(key).numpy()
                scoped_name = prefix + key
                data, used_qtype = convert_tensor(tensor, qtype, scoped_name)
                writer.add_tensor(scoped_name, data, raw_dtype=used_qtype)
                if (i + 1) % 50 == 0 or (i + 1) == len(keys):
                    print(f"  [{i+1}/{len(keys)}] {scoped_name} -> {used_qtype.name}")

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
