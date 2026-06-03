#!/usr/bin/env python3
"""Generate a tiny synthetic LTX-2 GGUF for the CPU smoke / end-to-end test.

This emits the exact tensor names of the full LTX-2 stack at drastically
reduced dimensions:
  * the video DiT block tree            (src/ltx2.hpp,    model.diffusion_model.*)
  * the Gemma-3 text encoder            (src/gemma3.hpp,  text_encoders.gemma3.*)
  * the LTX multi-layer text projection (conditioner,     text_embedding_projection.*)

so CI can verify that the model loads, binds every tensor, and runs a full
generate on CPU without the ~46 GB real checkpoint. The C++ side infers
geometry from these shapes, so the same code path that loads this file loads
the real weights. The Video-VAE is geometric (weightless) and needs no tensors.

    python script/make_synthetic_ltx2_gguf.py --out /tmp/ltx2_tiny.gguf
"""

from __future__ import annotations

import argparse

import numpy as np
import gguf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="LTX-2 checkpoint (DiT + text projection); load via --diffusion-model")
    ap.add_argument("--gemma-out", help="Gemma-3 encoder file; load via --llm. Defaults to <out>.gemma.gguf")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--connector-layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--in-channels", type=int, default=128)
    ap.add_argument("--ffn", type=int, default=128)
    ap.add_argument("--registers", type=int, default=16)
    ap.add_argument("--freq", type=int, default=16)
    # Gemma-3 encoder (tiny)
    ap.add_argument("--gemma-layers", type=int, default=2)
    ap.add_argument("--gemma-hidden", type=int, default=32)
    ap.add_argument("--gemma-heads", type=int, default=2)
    ap.add_argument("--gemma-kv-heads", type=int, default=1)
    ap.add_argument("--gemma-head-dim", type=int, default=16)
    ap.add_argument("--gemma-intermediate", type=int, default=64)
    ap.add_argument("--gemma-vocab", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    P = "model.diffusion_model."
    dim, inner, heads = args.dim, args.ffn, args.heads
    inc, freq, regs = args.in_channels, args.freq, args.registers

    w = gguf.GGUFWriter(args.out, "ltx2")
    tensors: dict[str, tuple[int, ...]] = {}

    def lin(name, out_f, in_f):
        # HF layout [out, in]; ggml reads ne0=in, ne1=out.
        tensors[name + ".weight"] = (out_f, in_f)
        tensors[name + ".bias"] = (out_f,)

    def lin_nb(name, out_f, in_f):
        tensors[name + ".weight"] = (out_f, in_f)

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

    # ---- video DiT ----
    lin(P + "patchify_proj", dim, inc)
    lin(P + "proj_out", inc, dim)
    tensors[P + "scale_shift_table"] = (2, dim)
    lin(P + "adaln_single.emb.timestep_embedder.linear_1", dim, freq)
    lin(P + "adaln_single.emb.timestep_embedder.linear_2", dim, dim)
    lin(P + "adaln_single.linear", 9 * dim, dim)
    lin(P + "prompt_adaln_single.emb.timestep_embedder.linear_1", dim, freq)
    lin(P + "prompt_adaln_single.emb.timestep_embedder.linear_2", dim, dim)
    lin(P + "prompt_adaln_single.linear", 2 * dim, dim)

    C = P + "video_embeddings_connector."
    tensors[C + "learnable_registers"] = (regs, dim)
    for i in range(args.connector_layers):
        b = C + f"transformer_1d_blocks.{i}"
        attn(b + ".attn1", dim, dim)
        ff(b + ".ff")

    for i in range(args.layers):
        b = P + f"transformer_blocks.{i}"
        attn(b + ".attn1", dim, dim)
        attn(b + ".attn2", dim, dim)
        ff(b + ".ff")
        tensors[b + ".scale_shift_table"] = (9, dim)
        tensors[b + ".prompt_scale_shift_table"] = (2, dim)

    # ---- LTX multi-layer text projection: (hidden * gemma_layers) -> dim ----
    # Lives in the LTX checkpoint, so it is namespaced under model.diffusion_model.
    lin(P + "text_embedding_projection.video_proj", dim, args.gemma_hidden * args.gemma_layers)

    # ---- Gemma-3 text encoder (separate file, relative names for --llm) ----
    gemma_tensors: dict[str, tuple[int, ...]] = {}

    def glin_nb(name, out_f, in_f):
        gemma_tensors[name + ".weight"] = (out_f, in_f)

    G = ""
    gh = args.gemma_hidden
    ghd = args.gemma_head_dim
    gq = args.gemma_heads * ghd
    gkv = args.gemma_kv_heads * ghd
    gi = args.gemma_intermediate
    # HF Embedding is [num_embeddings, embedding_dim] -> ggml ne0=hidden, ne1=vocab
    gemma_tensors[G + "embed_tokens.weight"] = (args.gemma_vocab, gh)
    for i in range(args.gemma_layers):
        b = G + f"layers.{i}"
        gemma_tensors[b + ".input_layernorm.weight"] = (gh,)
        glin_nb(b + ".self_attn.q_proj", gq, gh)
        glin_nb(b + ".self_attn.k_proj", gkv, gh)
        glin_nb(b + ".self_attn.v_proj", gkv, gh)
        glin_nb(b + ".self_attn.o_proj", gh, gq)
        gemma_tensors[b + ".self_attn.q_norm.weight"] = (ghd,)
        gemma_tensors[b + ".self_attn.k_norm.weight"] = (ghd,)
        gemma_tensors[b + ".post_attention_layernorm.weight"] = (gh,)
        gemma_tensors[b + ".pre_feedforward_layernorm.weight"] = (gh,)
        glin_nb(b + ".mlp.gate_proj", gi, gh)
        glin_nb(b + ".mlp.up_proj", gi, gh)
        glin_nb(b + ".mlp.down_proj", gh, gi)
        gemma_tensors[b + ".post_feedforward_layernorm.weight"] = (gh,)
    gemma_tensors[G + "norm.weight"] = (gh,)

    for name, shape in tensors.items():
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.02)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {len(tensors)} tensors to {args.out}")

    gemma_out = args.gemma_out or (args.out + ".gemma.gguf")
    gw = gguf.GGUFWriter(gemma_out, "gemma3")
    for name, shape in gemma_tensors.items():
        gw.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.02)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"wrote {len(gemma_tensors)} tensors to {gemma_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
