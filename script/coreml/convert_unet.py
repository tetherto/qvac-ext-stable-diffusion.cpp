"""Export a diffusers SD 1.x/2.x or SDXL UNet to CoreML for sd-cli --coreml-unet.

The model takes float32 inputs and returns float32, with the names the runtime expects:
  sample [N, 4, H/8, W/8], timestep [N], encoder_hidden_states [N, T, D] -> noise_pred [N, 4, H/8, W/8]
SDXL UNets also take y [N, 2816]: the pooled text embedding followed by the embedded size and crop ids, as
sd.cpp's SDXL conditioner builds it (the model applies add_embedding to it directly).

Every input shape is fixed, because the Neural Engine only runs fixed-shape graphs: with
EnumeratedShapes or RangeDim the traced UNet keeps runtime reshapes and CoreML moves the whole
graph to the CPU. To support several resolutions with one file, pass several --sizes: each one
becomes a fixed-shape function of a multifunction model (iOS 18+), and all functions share one
copy of the weights. The runtime picks the function whose input shapes match.

Only list sizes the model was trained at. SDXS-512 is a one-step model distilled at 512x512 only:
at 512x768, 768x512 and 768x768 it duplicates features (extra eyes and noses) on every backend,
so this script refuses other sizes for it unless --allow-untrained-sizes is given. For other
models it warns when a size differs from the one in the UNet config. See docs/coreml.md.

Weights: fp16 fits small UNets such as SDXS (603 MB). The SD 2.x UNet is 1.6 GB in fp16, too big to compile for
the ANE, so CoreML runs it on the CPU; use int8 weights. --precision w8b (one scale per block of 32 weights,
iOS 18+) keeps a 20-step SD 2.1 image close to ggml Metal (24.8 dB); w8 (one scale per output channel) drifts
enough over 20 steps to change the composition (19.9 dB). Both run at about the same speed.
On iOS, prefer p6 (6-bit palettized, Apple's ml-stable-diffusion recipe): the ANE reads palettized
weights directly, so the compiled model is cached, while w8b is expanded to fp16 on the CPU on every
load (about 105 s and 5.3 GB on an iPhone 16). p6 runs k-means over every large layer, which takes
a long time and much memory: 20-40 minutes for SD 2.1 on a 16 GB Mac (needs scikit-learn).
SDXL's UNet is 5.1 GB in fp16 (10.3 GB as traced in fp32); for iOS use mixed and --chunks 2 (two chunks of about
700 MB, the size of Apple's iOS SDXL chunks). Plain p4 loses fine texture over 20 steps; mixed gives 6 bits to the
quarter of the weights most sensitive to 4 bits and matches p6 at 75% of its size. The export needs about 30 GB of
RAM; the sensitivity pass behind mixed takes about 20 minutes on an M3 Ultra and is cached in --sensitivity.

Examples:
  python convert_unet.py --unet sdxs/unet --out sdxs_unet_fp16.mlpackage --compile
  python convert_unet.py --unet sdxs/unet --out sdxs_unet_multi.mlpackage --sizes 512x512,512x768,768x512 --compile
  python convert_unet.py --unet sd21/unet --out sd21_unet_w8b.mlpackage --precision w8b --compile
  python convert_unet.py --unet sd21-base/unet --out sd21base_unet_p6.mlpackage --precision p6 --batch 2 --compile
  python convert_unet.py --unet sdxl-base/unet --out sdxl_unet_mixed.mlpackage --precision mixed --batch 2 \
      --attention split_einsum_v2 --sizes 768x768 --chunks 2 --compile

pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4
"""

import argparse
import gc
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput


def silu_via_tanh(self, x):
    # Same function as x * sigmoid(x). The Neural Engine's fp16 silu op (which coremltools also
    # fuses x * sigmoid(x) into) is inaccurate enough to visibly change one-step SDXS images
    # (cosine 0.99939 vs fp32 on M2), while this form stays on the ANE at cosine 0.999995.
    return 0.5 * x * (1.0 + torch.tanh(0.5 * x))


class SplitEinsumAttention:
    """Attention written for the Apple Neural Engine (same math as standard attention).

    Our own implementation of the idea behind Apple's SPLIT_EINSUM_V2 (apple/ml-stable-diffusion): every head
    becomes its own small matmul + softmax instead of one batched multi-head attention, and long query sequences
    (4096 tokens for a 64x64 latent) are split into chunks. The ANE runs many small 2D attentions much faster
    than one large 4D one.
    """

    def __init__(self, query_chunk=512):
        self.query_chunk = query_chunk

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        if attention_mask is not None or attn.spatial_norm is not None or attn.group_norm is not None or hidden_states.ndim != 3:
            raise ValueError("SplitEinsumAttention supports the plain 3D self/cross attention of SD 1.x/2.x UNets only")
        residual = hidden_states
        context  = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross and encoder_hidden_states is not None:
            context = attn.norm_encoder_hidden_states(context)

        q = attn.to_q(hidden_states) * attn.scale  # scale before q @ k^T keeps the fp16 scores small
        k = attn.to_k(context)
        v = attn.to_v(context)
        head_dim = q.shape[-1] // attn.heads
        chunk    = min(self.query_chunk, q.shape[1])  # split, not chunk: coremltools cannot convert torch.chunk

        heads = []
        for qh, kh, vh in zip(q.split(head_dim, dim=-1), k.split(head_dim, dim=-1), v.split(head_dim, dim=-1)):
            kt    = kh.transpose(1, 2)
            parts = [torch.bmm(torch.softmax(torch.bmm(qc, kt), dim=-1), vh) for qc in qh.split(chunk, dim=1)]
            heads.append(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])
        out = attn.to_out[1](attn.to_out[0](torch.cat(heads, dim=-1)))

        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def _conv1x1(x, linear):
    """A Linear layer applied to channels-first [B, C, 1, S] data, as a 1x1 convolution."""
    return F.conv2d(x, linear.weight[:, :, None, None], linear.bias)


def _layer_norm_channels(x, norm):
    """LayerNorm over the channel axis of [B, C, 1, S] data."""
    mean = x.mean(dim=1, keepdim=True)
    var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
    x    = (x - mean) * torch.rsqrt(var + norm.eps)
    return x * norm.weight[None, :, None, None] + norm.bias[None, :, None, None]


def _attention_channels(attn, x, context, query_chunk):
    """Split-einsum attention on channels-first data: x [B, C, 1, S], context [B, D, 1, T]."""
    q        = _conv1x1(x, attn.to_q) * attn.scale
    k        = _conv1x1(context, attn.to_k)
    v        = _conv1x1(context, attn.to_v)
    head_dim = q.shape[1] // attn.heads
    chunk    = min(query_chunk, q.shape[3])
    heads    = []
    for qh, kh, vh in zip(q.split(head_dim, dim=1), k.split(head_dim, dim=1), v.split(head_dim, dim=1)):
        kt    = kh.transpose(1, 3)  # [B, T, 1, d]
        parts = []
        for qc in qh.split(chunk, dim=3):
            w = torch.softmax(torch.einsum("bchq,bkhc->bkhq", qc, kt), dim=1)  # [B, T, 1, q]
            parts.append(torch.einsum("bkhq,bchk->bchq", w, vh))  # [B, d, 1, q]
        heads.append(torch.cat(parts, dim=3) if len(parts) > 1 else parts[0])
    return _conv1x1(torch.cat(heads, dim=1), attn.to_out[0])


def ane_transformer_forward(query_chunk):
    """Forward for diffusers' Transformer2DModel (SD 1.x/2.x) that keeps data channels-first, [B, C, 1, S].

    Our own implementation of the layout Apple's ml-stable-diffusion uses for the ANE: every Linear in the block
    becomes a 1x1 convolution and LayerNorm runs over the channel axis, so there are no [B, S, C] <-> [B, C, H, W]
    transposes, and attention is split per head and per query chunk as in SplitEinsumAttention. Same math as the
    standard block.
    """

    def forward(self, hidden_states, encoder_hidden_states=None, *args, return_dict=True, **kwargs):
        b, c, h, w = hidden_states.shape
        residual   = hidden_states
        x          = self.norm(hidden_states).reshape(b, c, 1, h * w)
        x          = _conv1x1(x, self.proj_in) if self.use_linear_projection else self.proj_in(x)
        context    = encoder_hidden_states.transpose(1, 2).unsqueeze(2)  # [B, T, D] -> [B, D, 1, T]
        for block in self.transformer_blocks:
            if block.norm_type != "layer_norm" or block.only_cross_attention or block.pos_embed is not None:
                raise ValueError("the ANE transformer layout supports SD 1.x/2.x transformer blocks only")
            n = _layer_norm_channels(x, block.norm1)
            x = x + _attention_channels(block.attn1, n, n, query_chunk)
            n = _layer_norm_channels(x, block.norm2)
            x = x + _attention_channels(block.attn2, n, context, query_chunk)
            n = _layer_norm_channels(x, block.norm3)
            geglu        = block.ff.net[0]
            hidden, gate = _conv1x1(n, geglu.proj).split(geglu.proj.out_features // 2, dim=1)
            x = x + _conv1x1(hidden * F.gelu(gate), block.ff.net[2])
        x   = _conv1x1(x, self.proj_out) if self.use_linear_projection else self.proj_out(x)
        out = x.reshape(b, c, h, w) + residual
        return (out,) if not return_dict else Transformer2DModelOutput(sample=out)

    return forward


def apply_attention(unet, attention, query_chunk):
    """Switches the UNet's attention to one of the ANE rewrites (standard leaves it unchanged)."""
    if attention == "split_einsum_v2":
        unet.set_attn_processor(SplitEinsumAttention(query_chunk))
    elif attention == "ane":
        for module in unet.modules():
            if type(module).__name__ == "Transformer2DModel":
                module.forward = ane_transformer_forward(query_chunk).__get__(module)


def max_rel_err(out, reference):
    """Largest absolute difference relative to the reference's largest magnitude."""
    return float(np.abs(out - reference).max() / max(np.abs(reference).max(), 1e-12))


# The rewrites change only the order of float32 operations; anything above this is a math error.
REWRITE_TOLERANCE = 1e-4


def adm_dim(cfg):
    """Values of SDXL's vector conditioning y (2816 for SDXL base), or 0 for UNets without it."""
    if cfg.class_embed_type is not None or cfg.addition_embed_type not in (None, "text_time"):
        raise SystemExit(f"error: unsupported UNet conditioning (class_embed_type={cfg.class_embed_type}, "
                         f"addition_embed_type={cfg.addition_embed_type}); SD 1.x/2.x and SDXL are supported")
    return cfg.projection_class_embeddings_input_dim if cfg.addition_embed_type == "text_time" else 0


def use_embedded_time_ids(unet):
    """Makes the UNet take SDXL's size and crop ids already embedded, as sd.cpp's y carries them."""
    if adm_dim(unet.config):
        unet.add_time_proj = torch.nn.Identity()


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.adm  = adm_dim(unet.config)

    def forward(self, sample, timestep, encoder_hidden_states, y=None):
        added = None
        if y is not None:
            # With add_time_proj replaced by Identity (use_embedded_time_ids), diffusers concatenates
            # text_embeds and time_ids back into y unchanged, so where y is split does not matter.
            split = self.adm - self.unet.config.addition_time_embed_dim
            added = {"text_embeds": y[:, :split], "time_ids": y[:, split:]}
        return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added, return_dict=False)[0]


def sdxl_example(unet, batch, width, height):
    """Random pooled text embeddings with real size/crop ids: the y sd.cpp passes, and the unmodified UNet's
    added_cond_kwargs for the same values (the reference)."""
    cfg        = unet.config
    n_ids      = 6  # original size, crop, target size; any count works for the check, as y is not split by it
    text_dim   = cfg.projection_class_embeddings_input_dim - n_ids * cfg.addition_time_embed_dim
    text       = torch.randn(batch, text_dim)
    time_ids   = torch.tensor([[height, width, 0, 0, height, width]] * batch, dtype=torch.float32)
    with torch.no_grad():
        embedded = unet.add_time_proj(time_ids.flatten()).reshape(batch, -1)
    return torch.cat([text, embedded], dim=1), {"text_embeds": text, "time_ids": time_ids}


# Weights palettized by every p4/p6/mixed export: smaller ones stay fp16 (coremltools' weight_threshold).
PALETTE_THRESHOLD = 2048


def palettize_tensor(w, nbits, iters=10):
    """w with its values replaced by the nearest of 2**nbits k-means centroids: the effect of coremltools' kmeans
    palettization (1-D Lloyd iterations from quantile seeds), fast enough to try on every layer."""
    flat      = w.detach().flatten().float()
    k         = 2 ** nbits
    ordered   = flat.sort().values
    centroids = ordered[torch.arange(k) * (flat.numel() - 1) // (k - 1)]  # integer math: float32 indices overflow
    for _ in range(iters):
        idx       = torch.bucketize(flat, (centroids[1:] + centroids[:-1]) / 2)
        sums      = torch.zeros(k).scatter_add_(0, idx, flat)
        counts    = torch.bincount(idx, minlength=k)
        centroids = torch.where(counts > 0, sums / counts.clamp(min=1), centroids)
    idx = torch.bucketize(flat, (centroids[1:] + centroids[:-1]) / 2)
    return centroids[idx].reshape(w.shape).to(w.dtype)


def psnr(out, reference):
    mse = float(((out - reference) ** 2).mean())
    return 10 * math.log10(float(reference.abs().max()) ** 2 / max(mse, 1e-20))


def layer_sensitivity(unet, example, added, low_bits, cache_path):
    """Output PSNR of the UNet with one layer's weights palettized to low_bits, against fp32, for every weight that
    gets palettized. The UNet runs once per layer (about 800 for SDXL), so results go to cache_path as they come
    and a rerun only computes the missing layers."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("low_bits", low_bits) != low_bits:
        raise SystemExit(f"error: {cache_path} was measured at {cache['low_bits']} bits, not {low_bits}")
    results = cache.get("psnr", {})
    params  = [(n, p) for n, p in unet.named_parameters() if p.numel() >= PALETTE_THRESHOLD and n not in results]
    with torch.no_grad():
        reference = unet(*example, added_cond_kwargs=added, return_dict=False)[0]
        t0        = time.time()
        for i, (name, param) in enumerate(params):
            saved = param.data.clone()
            param.data.copy_(palettize_tensor(saved, low_bits))
            results[name] = psnr(unet(*example, added_cond_kwargs=added, return_dict=False)[0], reference)
            param.data.copy_(saved)
            if (i + 1) % 25 == 0 or i + 1 == len(params):
                cache_path.write_text(json.dumps({"low_bits": low_bits, "psnr": results}, indent=1))
                left = (time.time() - t0) / (i + 1) * (len(params) - i - 1)
                print(f"sensitivity: {i + 1}/{len(params)} layers, about {left / 60:.0f} min left", flush=True)
    return results


def mixed_bits(unet, sensitivity, target_bits, low_bits, high_bits):
    """Bits per weight averaging at most target_bits: low_bits everywhere, then high_bits for the layers whose
    low_bits palette hurts the output most (lowest PSNR first), as long as the average stays within the target."""
    # Weights only: coremltools folds 1-D parameters (norm biases and scales) into other ops, so they have no
    # constant of their own to palettize.
    sizes  = {n: p.numel() for n, p in unet.named_parameters() if n in sensitivity and p.dim() >= 2}
    bits   = {n: low_bits for n in sizes}
    budget = (target_bits - low_bits) * sum(sizes.values())
    for name in sorted(sizes, key=sensitivity.get):
        cost = (high_bits - low_bits) * sizes[name]
        if cost <= budget:
            bits[name] = high_bits
            budget    -= cost
    average = sum(bits[n] * sizes[n] for n in sizes) / sum(sizes.values())
    upgraded = sum(b == high_bits for b in bits.values())
    print(f"mixed: {upgraded} of {len(bits)} layers at {high_bits} bits, the rest at {low_bits}; "
          f"average {average:.2f} bits")
    return bits


def mil_weight_name(param_name):
    """The name coremltools gives a traced UNetWrapper parameter's fp16 weight constant."""
    return "unet_" + param_name.replace(".", "_") + "_to_fp16"


def parse_size(text):
    width, height = (int(v) for v in text.lower().split("x"))
    if width % 8 or height % 8:
        raise argparse.ArgumentTypeError(f"{text}: width and height must be multiples of 8")
    return width, height


def native_size(cfg):
    side = cfg.sample_size * 8
    return side, side


def is_sdxs(cfg):
    # BK-SDM-Tiny and Segmind Tiny-SD also have no mid block and three levels; SDXS additionally drops
    # attention at the top level (sd.cpp detects VERSION_SDXS by the missing output_blocks.7.1).
    return (cfg.mid_block_type is None and len(cfg.block_out_channels) == 3 and
            list(cfg.down_block_types)[0] == "DownBlock2D")


def prediction_type(unet_dir, override):
    """epsilon or v_prediction, from --prediction or the pipeline's scheduler config next to unet/."""
    if override:
        return override
    sched = Path(unet_dir).resolve().parent / "scheduler" / "scheduler_config.json"
    if sched.exists():
        return json.loads(sched.read_text()).get("prediction_type", "epsilon")
    raise SystemExit(f"error: {sched} not found; pass --prediction epsilon or --prediction v_prediction")


def check_sizes(cfg, sizes, allow_untrained):
    native = native_size(cfg)
    off = [s for s in sizes if s != native]
    if not off:
        return
    listed = ", ".join(f"{w}x{h}" for w, h in off)
    if is_sdxs(cfg) and not allow_untrained:
        raise SystemExit(f"error: this looks like SDXS, which only gives good images at {native[0]}x{native[1]}; "
                         f"refusing {listed} (pass --allow-untrained-sizes to export anyway)")
    print(f"warning: {listed} differs from the UNet's native {native[0]}x{native[1]}; check image quality at those sizes")


def function_name(width, height):
    return f"unet_{width}x{height}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unet", required=True, help="diffusers UNet folder (the pipeline's unet/ directory)")
    parser.add_argument("--out", required=True, help="output .mlpackage path")
    parser.add_argument("--precision", choices=["fp16", "w8", "w8b", "p6", "p4", "mixed"], default="fp16",
                        help="weights: fp16, int8 with one scale per output channel (w8), int8 with one scale per "
                             "block of 32 weights (w8b, iOS 18+), a 6-bit (p6) or 4-bit (p4) k-means palette per "
                             "layer, or mixed: 4-bit palettes with 6 bits for the layers most sensitive to 4 bits, "
                             "averaging --mixed-bits (large UNets on iOS: p6 for SD 2.x, mixed for SDXL); compute "
                             "is fp16 in all cases")
    parser.add_argument("--mixed-bits", type=float, default=4.5,
                        help="average bits per palettized weight for --precision mixed (between 4 and 6)")
    parser.add_argument("--sensitivity",
                        help="JSON cache of the per-layer sensitivity --precision mixed measures (default: OUT with "
                             "_sensitivity.json); reused by later exports of the same UNet")
    parser.add_argument("--attention", choices=["standard", "split_einsum_v2", "ane"], default="standard",
                        help="standard diffusers attention; split_einsum_v2: per-head attention with chunked queries; "
                             "ane: the same attention with the whole transformer block kept channels-first ([B, C, 1, S], "
                             "Linear as 1x1 conv), as Apple lays it out for the ANE (fastest for SD 2.x)")
    parser.add_argument("--query-chunk", type=int, default=512, help="query tokens per chunk for split_einsum_v2")
    parser.add_argument("--palette-mode", choices=["kmeans", "uniform"], default="kmeans",
                        help="how p6/p4 pick the palette values: kmeans (accurate, slow) or uniform (fast; for "
                             "speed experiments, same file format and run time)")
    parser.add_argument("--kmeans-workers", type=int, default=2,
                        help="parallel k-means workers for p6/p4; each one adds memory, keep it low on a 16 GB Mac")
    parser.add_argument("--batch", type=int, default=1, help="UNet batch: 1, or 2 for batched CFG")
    parser.add_argument("--sizes", default="512x512",
                        help="comma-separated image sizes as WIDTHxHEIGHT, only ones the model was trained at "
                             "(SDXS: 512x512); more than one makes a multifunction model")
    parser.add_argument("--prediction", choices=["epsilon", "v_prediction"],
                        help="what the UNet predicts; recorded in the model so sd.cpp can pick the denoiser "
                             "(default: the pipeline's scheduler/scheduler_config.json)")
    parser.add_argument("--allow-untrained-sizes", action="store_true",
                        help="export sizes other than the model's native one even for SDXS (expect bad images)")
    parser.add_argument("--context-len", type=int, default=77, help="text tokens; sd.cpp uses 77 per prompt chunk")
    parser.add_argument("--target", choices=["iOS17", "iOS18"],
                        help="op set (default: iOS18 for w8b, which needs it, else iOS17). A multifunction model built "
                             "with iOS18 fails to load on the ANE (macOS 27.2); iOS17 runs at the same speed")
    parser.add_argument("--chunks", type=int, choices=[1, 2], default=1,
                        help="2 splits the model into two chunks of about equal size (coremltools bisect_model), "
                             "written to the folder OUT without .mlpackage; pass that folder to --coreml-unet. "
                             "Large UNets (SDXL) need this to compile for the ANE on iOS")
    parser.add_argument("--compile", action="store_true", help="also write a compiled .mlmodelc next to the package")
    parser.add_argument("--keep-silu", action="store_true", help="keep the native silu op (inaccurate on the ANE)")
    args = parser.parse_args()

    sizes = [parse_size(s) for s in args.sizes.split(",")]
    if args.chunks > 1 and len(sizes) > 1:
        raise SystemExit("error: --chunks needs a single --sizes entry")
    if args.target is None:
        args.target = "iOS18" if args.precision == "w8b" else "iOS17"
    if args.precision == "w8b" and args.target != "iOS18":
        raise SystemExit("error: --precision w8b needs --target iOS18 (per-block quantization is an iOS 18 op)")
    if args.precision == "mixed" and not 4 <= args.mixed_bits <= 6:
        raise SystemExit("error: --mixed-bits must be between 4 and 6")
    if args.precision == "w8b" and len(sizes) > 1:
        print("warning: w8b needs the iOS 18 op set, and multifunction models built with it failed to load on the ANE")
    # Some pipelines ship only the fp16 variant (diffusion_pytorch_model.fp16.safetensors); trace in fp32 either way.
    has_default = any((Path(args.unet) / f).exists() for f in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"))
    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float32,
                                                variant=None if has_default else "fp16").eval()
    wrapper = UNetWrapper(unet).eval()
    cfg = unet.config
    adm = adm_dim(cfg)
    check_sizes(cfg, sizes, args.allow_untrained_sizes)
    prediction = prediction_type(args.unet, args.prediction)
    print(f"prediction type: {prediction}")
    multi = len(sizes) > 1
    parts_dir = Path(tempfile.mkdtemp(prefix="unet_parts_")) if multi else None
    checks = []  # (function name or None, inputs, fp32 reference output)

    # References come from the unmodified diffusers UNet (standard attention, x * sigmoid(x), SDXL's raw size and
    # crop ids), so the checks below catch mistakes in the rewrites as well as conversion error.
    examples = {}
    for width, height in sizes:
        shapes = {
            "sample": (args.batch, cfg.in_channels, height // 8, width // 8),
            "timestep": (args.batch,),
            "encoder_hidden_states": (args.batch, args.context_len, cfg.cross_attention_dim),
        }
        example = [
            torch.randn(shapes["sample"]),
            torch.full(shapes["timestep"], 999.0),
            torch.randn(shapes["encoder_hidden_states"]),
        ]
        added = None
        if adm:
            shapes["y"] = (args.batch, adm)
            y, added = sdxl_example(unet, args.batch, width, height)
            example.append(y)
        with torch.no_grad():
            reference = unet(*example[:3], added_cond_kwargs=added, return_dict=False)[0].numpy()
        examples[width, height] = (shapes, tuple(example), reference, added)

    bits = None
    if args.precision == "mixed":
        # Measured on one item of the unmodified UNet mid-trajectory (t = 500); the rewrites below do not change the
        # math, and the batch does not change which layers are sensitive.
        _, example, _, added = examples[sizes[0]]
        calibration = (example[0][:1], torch.full_like(example[1][:1], 500.0), example[2][:1])
        added = {k: v[:1] for k, v in added.items()} if added else None
        cache = Path(args.sensitivity) if args.sensitivity else Path(args.out).with_name(Path(args.out).stem + "_sensitivity.json")
        sensitivity = layer_sensitivity(unet, calibration, added, 4, cache)
        bits = mixed_bits(unet, sensitivity, args.mixed_bits, 4, 6)

    if not args.keep_silu:
        torch.nn.SiLU.forward = silu_via_tanh
    apply_attention(unet, args.attention, args.query_chunk)
    use_embedded_time_ids(unet)

    for width, height in sizes:
        shapes, example, reference, _ = examples[width, height]
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, strict=False)
            err = max_rel_err(wrapper(*example).numpy(), reference)
        print(f"{width}x{height}: rewritten UNet vs diffusers (fp32): max relative error {err:.2e}")
        if err > REWRITE_TOLERANCE:
            raise SystemExit(f"error: the {args.attention} attention / silu / time id rewrite differs from diffusers by {err:.2e} "
                             f"(tolerance {REWRITE_TOLERANCE:.0e})")

        t0 = time.time()
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name=name, shape=shape, dtype=np.float32) for name, shape in shapes.items()],
            outputs=[ct.TensorType(name="noise_pred", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=getattr(ct.target, args.target),
            skip_model_load=True,  # loading compiles another full-size temporary copy; the checks use the final model
        )
        if args.precision in ("w8", "w8b"):
            # Symmetric only: asymmetric (with a zero point) moves the whole UNet off the ANE onto the CPU.
            granularity = {"granularity": "per_block", "block_size": 32} if args.precision == "w8b" else {"granularity": "per_channel"}
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                                   weight_threshold=2048, **granularity)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel,
                                                                 ct.optimize.coreml.OptimizationConfig(op_config))
        elif args.precision in ("p6", "p4"):
            op_config = ct.optimize.coreml.OpPalettizerConfig(mode=args.palette_mode, nbits=int(args.precision[1:]),
                                                              weight_threshold=PALETTE_THRESHOLD,
                                                              num_kmeans_workers=args.kmeans_workers)
            mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, ct.optimize.coreml.OptimizationConfig(op_config))
        elif args.precision == "mixed":
            def palette(nbits):
                return ct.optimize.coreml.OpPalettizerConfig(mode=args.palette_mode, nbits=nbits,
                                                             weight_threshold=PALETTE_THRESHOLD,
                                                             num_kmeans_workers=args.kmeans_workers)
            consts  = {op.name for op in mlmodel._mil_program.functions["main"].operations if op.op_type == "const"}
            by_name = {mil_weight_name(n): palette(b) for n, b in bits.items() if b != 4}
            missing = sorted(set(by_name) - consts)
            if missing:
                raise SystemExit(f"error: {len(missing)} weights not found in the CoreML program, e.g. {missing[0]}")
            mlmodel = ct.optimize.coreml.palettize_weights(
                mlmodel, ct.optimize.coreml.OptimizationConfig(global_config=palette(4), op_name_configs=by_name))
        print(f"{width}x{height}: converted in {time.time() - t0:.1f} s")

        inputs = {name: tensor.numpy() for name, tensor in zip(shapes, example)}
        mlmodel.user_defined_metadata["prediction_type"] = prediction
        if multi:
            mlmodel.save(str(parts_dir / f"{function_name(width, height)}.mlpackage"))
            checks.append((function_name(width, height), inputs, reference))
        else:
            mlmodel.short_description = f"UNet {Path(args.unet).resolve()} ({args.precision}) for sd-cli --coreml-unet"
            mlmodel.save(args.out)
            checks.append((None, inputs, reference))
        # coremltools keeps a temporary package per model in memory (several GB for SDXL) until it is released.
        del mlmodel, traced
        gc.collect()

    if multi:
        desc = ct.utils.MultiFunctionDescriptor()
        for width, height in sizes:
            name = function_name(width, height)
            desc.add_function(str(parts_dir / f"{name}.mlpackage"), src_function_name="main", target_function_name=name)
        desc.default_function_name = function_name(*sizes[0])
        ct.utils.save_multifunction(desc, args.out)
        merged = ct.models.MLModel(args.out, skip_model_load=True)
        if merged.user_defined_metadata.get("prediction_type") != prediction:
            merged.user_defined_metadata["prediction_type"] = prediction
            merged.save(args.out)
        shutil.rmtree(parts_dir)
        print(f"saved {len(sizes)} functions to {args.out}")

    # The checks run the compiled model: loading an .mlpackage compiles a new temporary copy for every load
    # (and the coremltools loader rejects function_name on an .mlpackage).
    out_dir = Path(args.out).resolve().parent
    compiled = out_dir / (Path(args.out).stem + ".mlmodelc")
    if compiled.exists():
        shutil.rmtree(compiled)
    ct.utils.compile_model(args.out, str(compiled))  # the CoreML framework's compiler; no Xcode needed

    # Check against PyTorch fp32 on this Mac. The ANE runs fp16, so expect small differences.
    for name, inputs, reference in checks:
        for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU):
            function = {"function_name": name} if name else {}
            model = ct.models.CompiledMLModel(str(compiled), compute_units=units, **function)
            model.predict(inputs)  # warm-up
            t0 = time.time()
            out = model.predict(inputs)["noise_pred"]
            dt = time.time() - t0
            cos = cosine(out, reference)
            print(f"{name or 'model'} {units.name}: {dt * 1000:.1f} ms, "
                  f"max abs err {np.abs(out - reference).max():.4f}, cosine {cos:.6f}")

    if args.chunks > 1:
        write_chunks(args.out, compiled, args.compile, prediction, checks[0][1:])
    elif args.compile:
        print(f"compiled to {compiled}")
    else:
        shutil.rmtree(compiled)


def cosine(out, reference):
    return float(np.dot(out.ravel(), reference.ravel()) / (np.linalg.norm(out) * np.linalg.norm(reference)))


def write_chunks(package, compiled, compile_chunks, prediction, check):
    """Splits the checked model into two chunks in the folder named like the package, then removes the whole
    model: sd.cpp runs the chunks one after the other (see src/coreml_unet.h)."""
    out_dir = Path(package).with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # coremltools' own chunk check needs pytest; the check below runs the chunks the way sd.cpp does instead.
    ct.models.utils.bisect_model(str(package), str(out_dir), check_output_correctness=False)
    chunks = sorted(out_dir.glob("*.mlpackage"))
    for chunk in chunks:
        model = ct.models.MLModel(str(chunk), skip_model_load=True)  # bisect_model drops the metadata
        model.user_defined_metadata["prediction_type"] = prediction
        tmp = chunk.with_name(chunk.stem + ".tmp.mlpackage")  # MLModel.save cannot overwrite its own package
        model.save(str(tmp))
        shutil.rmtree(chunk)
        tmp.rename(chunk)
    shutil.rmtree(package)
    if compiled.exists():
        shutil.rmtree(compiled)

    # Compiled once per chunk for the check, which loads them as sd.cpp does; kept only with --compile.
    inputs, reference = check
    features = dict(inputs)
    for chunk in chunks:
        chunk_compiled = chunk.with_suffix(".mlmodelc")
        ct.utils.compile_model(str(chunk), str(chunk_compiled))
        names = [i.name for i in ct.models.MLModel(str(chunk), skip_model_load=True).get_spec().description.input]
        model = ct.models.CompiledMLModel(str(chunk_compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
        features.update(model.predict({name: features[name] for name in names}))
        del model
    out = features["noise_pred"]
    print(f"{len(chunks)} chunks CPU_AND_NE: max abs err {np.abs(out - reference).max():.4f}, "
          f"cosine {cosine(out, reference):.6f}")
    if not compile_chunks:
        for chunk in chunks:
            shutil.rmtree(chunk.with_suffix(".mlmodelc"))
    print(f"wrote {len(chunks)} chunks to {out_dir}; pass the folder to --coreml-unet")


if __name__ == "__main__":
    main()
