#!/usr/bin/env python3
"""Generate a tiny synthetic LTX-2 video-DiT GGUF for the load-on-CPU smoke test.

This emits the exact tensor names of the LTX-2 video DiT block tree
(src/ltx2.hpp) at drastically reduced dimensions, so CI can verify that the
model loads and binds every tensor on CPU without the 46 GB real checkpoint.
The C++ side infers geometry from these shapes, so the same code path that
loads this file loads the real weights.

    python script/make_synthetic_ltx2_gguf.py --out /tmp/ltx2_tiny.gguf
"""

from __future__ import annotations

import argparse

import numpy as np
import gguf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--connector-layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--in-channels", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=128)
    ap.add_argument("--registers", type=int, default=16)
    ap.add_argument("--freq", type=int, default=16)
    args = ap.parse_args()

    P = "model.diffusion_model."
    dim, inner, heads = args.dim, args.ffn, args.heads
    inc, freq, regs = args.in_channels, args.freq, args.registers

    w = gguf.GGUFWriter(args.out, "ltx2")
    tensors: dict[str, tuple[int, ...]] = {}

    def lin(name, out_f, in_f):
        # HF layout [out, in]; ggml reads ne0=in, ne1=out.
        tensors[name + ".weight"] = (out_f, in_f)
        tensors[name + ".bias"] = (out_f,)

    def attn(prefix, q_dim, kv_dim):
        lin(prefix + ".to_q", q_dim, q_dim)
        lin(prefix + ".to_k", q_dim, kv_dim)
        lin(prefix + ".to_v", q_dim, kv_dim)
        lin(prefix + ".to_out.0", q_dim, q_dim)
        tensors[prefix + ".q_norm.weight"] = (q_dim,)
        tensors[prefix + ".k_norm.weight"] = (q_dim,)
        lin(prefix + ".to_gate_logits", heads, q_dim)

    def ff(prefix):
        lin(prefix + ".net.0.proj", inner, dim)
        lin(prefix + ".net.2", dim, inner)

    # top-level
    lin(P + "patchify_proj", dim, inc)
    lin(P + "proj_out", inc, dim)
    tensors[P + "scale_shift_table"] = (2, dim)
    lin(P + "adaln_single.emb.timestep_embedder.linear_1", dim, freq)
    lin(P + "adaln_single.emb.timestep_embedder.linear_2", dim, dim)
    lin(P + "adaln_single.linear", 9 * dim, dim)
    lin(P + "prompt_adaln_single.emb.timestep_embedder.linear_1", dim, freq)
    lin(P + "prompt_adaln_single.emb.timestep_embedder.linear_2", dim, dim)
    lin(P + "prompt_adaln_single.linear", 2 * dim, dim)

    # connector
    C = P + "video_embeddings_connector."
    tensors[C + "learnable_registers"] = (regs, dim)
    for i in range(args.connector_layers):
        b = C + f"transformer_1d_blocks.{i}"
        attn(b + ".attn1", dim, dim)
        ff(b + ".ff")

    # DiT blocks
    for i in range(args.layers):
        b = P + f"transformer_blocks.{i}"
        attn(b + ".attn1", dim, dim)
        attn(b + ".attn2", dim, dim)
        ff(b + ".ff")
        tensors[b + ".scale_shift_table"] = (9, dim)
        tensors[b + ".prompt_scale_shift_table"] = (2, dim)

    for name, shape in tensors.items():
        w.add_tensor(name, np.zeros(shape, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {len(tensors)} tensors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
