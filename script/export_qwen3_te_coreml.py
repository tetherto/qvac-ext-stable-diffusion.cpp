"""Export FLUX.2-klein's Qwen3-4B text encoder (layers 1-27, 512 tokens) to CoreML for the Neural Engine.

sd.cpp feeds the chat-templated prompt right-padded to 512 tokens (pad id 151643), with causal attention and
padded keys masked, and uses the hidden states after layers 9, 18 and 27, concatenated to 7680 values per token.
This exports those 27 layers as chunks run in order:
  chunk 1 inputs:  input_ids [1, 512] int32, key_mask [1, 512] fp16 (1 for prompt tokens, 0 for padding)
  later inputs:    stream_<k> [1, 512, 2560] fp16 (residual stream after layer k), key_mask
  outputs:         h_<9|18|27> [1, 512, 2560] fp16 for the taps inside the chunk, stream_<end> if not the last
Changes for fp16 on the Neural Engine (results are the same math):
  - RMSNorm as LayerNorm of [x, -x] (mean 0, variance mean(x^2)): token 0's residual stream reaches ~16,400,
    and squaring that overflows fp16.
  - attention per head (one op over all heads is wrong on an M2's Neural Engine), sharing the 8 KV heads.
  - RoPE cos/sin and the causal mask are constants: positions are always 0..511.
Usage: python export_qwen3_te_coreml.py --model DIR --chunk K [--cuts 5,12,19] [--bits 8] --output OUT
"""
import argparse
import gc
import shutil
import time
from pathlib import Path

import numpy as np

TAPS = (9, 18, 27)
LENGTH = 512


def bounds(cuts):
    b = [0, *sorted(cuts), TAPS[-1]]
    return list(zip(b[:-1], b[1:]))


def chunk_io(lo, hi):
    ins = ["input_ids", "key_mask"] if lo == 0 else [f"h_{lo}" if lo in TAPS else f"stream_{lo}", "key_mask"]
    outs = [f"h_{t}" for t in TAPS if lo < t <= hi]
    if hi < TAPS[-1] and f"h_{hi}" not in outs:
        outs.append(f"stream_{hi}")
    return ins, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="folder with text_encoder/")
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--cuts", default="5,12,19",
                    help="cut after these layers (1-27); chunk 1 also holds the 389M-value token embedding table")
    ap.add_argument("--bits", type=int, default=8, choices=(0, 4, 6, 8))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import coremltools as ct
    from transformers import AutoConfig, Qwen3Model

    torch.set_grad_enabled(False)
    chunks = bounds(int(c) for c in args.cuts.split(","))
    lo, hi = chunks[args.chunk - 1]
    config = AutoConfig.from_pretrained(args.model / "text_encoder")
    config.num_hidden_layers = hi  # layers after this chunk are not loaded
    t0 = time.time()
    model = Qwen3Model.from_pretrained(args.model / "text_encoder", config=config, torch_dtype=torch.float16).eval()
    D, H, KV, HD = config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    eps = config.rms_norm_eps

    inv = 1.0 / (config.rope_theta ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    ang = torch.arange(LENGTH, dtype=torch.float32)[:, None] * inv[None]
    cos = torch.cat((ang.cos(), ang.cos()), -1).half()[None, None]  # HF rotate_half layout
    sin = torch.cat((ang.sin(), ang.sin()), -1).half()[None, None]
    causal = torch.triu(torch.full((LENGTH, LENGTH), -1e4), 1).half()

    def rms(x, w):
        y = torch.nn.functional.layer_norm(torch.cat((x, -x), -1), (2 * x.shape[-1],), eps=eps)
        return y[..., : x.shape[-1]] * w

    def rope(t):
        a, b = t[..., : HD // 2], t[..., HD // 2:]
        return t * cos + torch.cat((-b, a), -1) * sin

    class Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(model.layers[lo:hi])
            self.embed = model.embed_tokens if lo == 0 else None
            self.register_buffer("causal", causal, persistent=False)

        def forward(self, x, key_mask):
            if self.embed is not None:
                x = self.embed(x)
            mask = self.causal + (1 - key_mask)[:, None, :] * -1e4  # [1, S, S]
            outs = []
            for i, layer in enumerate(self.layers):
                n = lo + i + 1
                at = layer.self_attn
                h = rms(x, layer.input_layernorm.weight)
                q = rope(rms(at.q_proj(h).reshape(1, LENGTH, H, HD), at.q_norm.weight).transpose(1, 2))
                k = rope(rms(at.k_proj(h).reshape(1, LENGTH, KV, HD), at.k_norm.weight).transpose(1, 2))
                v = at.v_proj(h).reshape(1, LENGTH, KV, HD).transpose(1, 2)
                heads = []
                for j in range(H):
                    kv = j // (H // KV)
                    s = (q[:, j] @ k[:, kv].transpose(-1, -2)) * (HD ** -0.5) + mask
                    heads.append(torch.softmax(s, -1) @ v[:, kv])
                x = x + at.o_proj(torch.cat(heads, -1))
                h = rms(x, layer.post_attention_layernorm.weight)
                mlp = layer.mlp
                x = x + mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(h)) * mlp.up_proj(h))
                if n in TAPS or n == hi:
                    outs.append(x)
            return tuple(outs) if len(outs) > 1 else outs[0]

    ins, outs = chunk_io(lo, hi)
    module = Chunk().eval()
    if lo == 0:
        example = (torch.full((1, LENGTH), 151643, dtype=torch.int32), torch.ones(1, LENGTH).half())
        ct_in = [ct.TensorType(name="input_ids", shape=(1, LENGTH), dtype=np.int32),
                 ct.TensorType(name="key_mask", shape=(1, LENGTH), dtype=np.float16)]
    else:
        example = (torch.zeros(1, LENGTH, D).half(), torch.ones(1, LENGTH).half())
        ct_in = [ct.TensorType(name=ins[0], shape=(1, LENGTH, D), dtype=np.float16),
                 ct.TensorType(name="key_mask", shape=(1, LENGTH), dtype=np.float16)]
    # Output order follows the forward: taps in layer order, then the chunk's end (which may itself be a tap).
    names = []
    for i in range(lo, hi):
        n = i + 1
        if n in TAPS:
            names.append(f"h_{n}")
        elif n == hi:
            names.append(f"stream_{n}")
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False, check_trace=False)
    m = ct.convert(traced, convert_to="mlprogram", inputs=ct_in,
                   outputs=[ct.TensorType(name=n, dtype=np.float16) for n in names],
                   compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14,
                   skip_model_load=True)
    del traced, module, model
    gc.collect()
    print(f"chunk {args.chunk}: layers {lo + 1}-{hi}, outputs {names}, converted in {time.time() - t0:.0f}s", flush=True)
    if args.bits:
        t0 = time.time()
        cfg = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=args.bits, weight_threshold=2048,
                                                    num_kmeans_workers=args.workers)
        m = ct.optimize.coreml.palettize_weights(m, ct.optimize.coreml.OptimizationConfig(cfg))
        print(f"palettized to {args.bits} bits in {time.time() - t0:.0f}s", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pkg = args.output / f".chunk{args.chunk}.mlpackage"
    shutil.rmtree(pkg, ignore_errors=True)
    m.save(str(pkg))
    del m
    final = args.output / f"chunk{args.chunk}.mlmodelc"
    shutil.rmtree(final, ignore_errors=True)
    ct.utils.compile_model(str(pkg), str(final))
    shutil.rmtree(pkg, ignore_errors=True)
    size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file()) / 1e9
    print(f"saved {final} ({size:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
