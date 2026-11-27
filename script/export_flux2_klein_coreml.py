#!/usr/bin/env python3
"""Validate and export the complete FLUX.2-klein 4B denoiser for Core ML.

The implementation is imported from Black Forest Labs' official flux2 source
checkout. This script only adapts the engine's fixed-shape tensor interface.
Run each mode in a separate process to limit peak memory.

`chunk` exports the Neural Engine model: one chunk per process, FP16 with the
residual-stream rescale, per-head attention and palettized weights, compiled to
<output>/chunk<k>.mlmodelc. sd-cli takes the folder with --coreml-diffusion.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from compare_flux_coreml_fixture import error_metrics, validate_capture


EXPECTED_CHANNELS = 128
EXPECTED_CONTEXT_WIDTH = 7680
DOUBLE_BLOCKS = 5
TOTAL_BLOCKS = 25  # 5 double-stream blocks, then 20 single-stream blocks
# Cut before these of the 25 blocks. An M2 Neural Engine runs a program of at
# most about 1 GB of stored weights (an M3 about 2 GB); a larger one falls back
# to the CPU entirely. With 8-bit palettes these cuts give 0.63-0.85 GB chunks.
DEFAULT_CUTS = (3, 6, 13, 20)


def fixture_shapes(manifest):
    tensors = manifest["tensors"]
    latent = tuple(reversed(tensors["latent"]["ne"]))
    context = tuple(reversed(tensors["context"]["ne"]))
    timesteps = tuple(reversed(tensors["timesteps"]["ne"]))
    if latent[0] != 1 or latent[1] != EXPECTED_CHANNELS:
        raise ValueError(f"expected one FLUX.2-klein latent with 128 channels, got {latent}")
    if context[0:2] != (1, 1) or context[3] != EXPECTED_CONTEXT_WIDTH:
        raise ValueError(f"expected context [1,1,T,7680], got {context}")
    if timesteps != (1, 1, 1, 1):
        raise ValueError(f"expected a single timestep, got {timesteps}")
    if "pooled" in tensors:
        raise ValueError("FLUX.2-klein export does not accept pooled conditioning")
    if latent[2] < 1 or latent[3] < 1 or context[2] < 1:
        raise ValueError("the fixture contains an empty sequence")
    return latent, timesteps, context


def position_ids(height, width, text_tokens, torch, device):
    """Match Rope::gen_flux_ids for text-to-image FLUX.2, with no references."""
    text = torch.zeros((1, text_tokens, 4), dtype=torch.float32, device=device)
    text[0, :, 3] = torch.arange(text_tokens, dtype=torch.float32, device=device)
    image = torch.zeros((1, height, width, 4), dtype=torch.float32, device=device)
    image[0, :, :, 1] = torch.arange(height, dtype=torch.float32, device=device)[:, None]
    image[0, :, :, 2] = torch.arange(width, dtype=torch.float32, device=device)[None, :]
    return image.reshape(1, height * width, 4), text


def rescale_for_fp16(model, stream, swiglu, torch):
    """Shrink the residual streams by 1/stream and SwiGLU products by 1/swiglu, without changing the output.

    Every read of a residual stream goes through a LayerNorm without affine parameters (block inputs and the
    final layer), which is invariant to scaling its input once eps is scaled with it. So dividing what writes
    the streams (img_in, txt_in, attention output projections, MLP down projections) by stream scales both
    streams exactly. Likewise silu(a) * b scales with b, so dividing b's rows of the up projection by swiglu
    and multiplying the down projection's matching columns by swiglu is exact. In FP16 the streams and
    SwiGLU products of FLUX.2-klein otherwise exceed 65504 on later denoising steps.
    """
    import math

    if any(f < 1 or not math.log2(f).is_integer() for f in (stream, swiglu)):
        raise ValueError("rescale factors must be powers of two, so the scaled weights stay exact")
    hidden = model.hidden_size
    mlp_hidden = model.double_blocks[0].img_mlp[2].in_features

    def scale_(tensor, factor):
        if tensor is not None:
            with torch.no_grad():
                tensor.mul_(factor)

    for linear in (model.img_in, model.txt_in):
        scale_(linear.weight, 1 / stream)
        scale_(linear.bias, 1 / stream)
    for block in model.double_blocks:
        for attn, mlp in ((block.img_attn, block.img_mlp), (block.txt_attn, block.txt_mlp)):
            scale_(attn.proj.weight, 1 / stream)
            scale_(attn.proj.bias, 1 / stream)
            scale_(mlp[0].weight[mlp_hidden:], 1 / swiglu)
            scale_(mlp[2].weight, swiglu / stream)
    for block in model.single_blocks:
        scale_(block.linear1.weight[3 * hidden + mlp_hidden:], 1 / swiglu)
        scale_(block.linear2.weight[:, :hidden], 1 / stream)
        scale_(block.linear2.weight[:, hidden:], swiglu / stream)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.eps /= stream * stream


def load_official_model(source, weights, dtype, device, torch, safe_fp16=False, rescale=None,
                        split_attention_heads=False, load_only=None):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("install safetensors before exporting") from exc
    source_root = source.resolve() / "src"
    if not (source_root / "flux2" / "model.py").is_file():
        raise ValueError(f"official flux2 source missing: {source_root / 'flux2' / 'model.py'}")
    sys.path.insert(0, str(source_root))
    from flux2 import model as flux2_model

    # With no reference tokens, the official attention function reduces to
    # ordinary full self-attention. Avoid tracing its empty reference branch.
    def no_reference_attention(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache=None):
        if num_ref_tokens != 0 or kv_cache is not None:
            raise ValueError("the Core ML exporter supports text-to-image without references")
        if split_attention_heads:
            # One attention per head. On an M2's Neural Engine, attention over all 24 heads of the
            # 1536-token sequence in one op returns wrong values (normalized error above 1), while
            # the per-head form matches FP32 and runs faster; M3 Neural Engines handle both.
            scale = q.shape[-1] ** -0.5
            out = torch.cat([torch.softmax((q[:, h:h + 1] @ k[:, h:h + 1].transpose(-1, -2)) * scale, -1)
                             @ v[:, h:h + 1] for h in range(q.shape[1])], 1)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)

    # The official RoPE implementation represents each rotation as a 2x2
    # matrix and temporarily creates rank-6 tensors. Core ML permits rank <=5.
    # Keep cosine and sine as a pair and rotate adjacent channels directly.
    def compact_rope(pos, dim, theta):
        scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
        angles = pos.unsqueeze(-1) * (1.0 / (theta ** scale))
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).float()

    def compact_embed_nd(self, ids):
        parts = [compact_rope(ids[..., i], self.axes_dim[i], self.theta)
                 for i in range(len(self.axes_dim))]
        return torch.cat(parts, dim=-2).unsqueeze(1)

    def compact_apply_rope(xq, xk, freqs):
        cos = freqs[..., 0]
        sin = freqs[..., 1]

        def rotate(x):
            pairs = x.float().reshape(*x.shape[:-1], -1, 2)
            real, imag = pairs[..., 0], pairs[..., 1]
            out = torch.stack((cos * real - sin * imag,
                               sin * real + cos * imag), dim=-1)
            return out.reshape_as(x).type_as(x)

        return rotate(xq), rotate(xk)

    flux2_model.causal_attn_fn = no_reference_attention
    flux2_model.EmbedND.forward = compact_embed_nd
    flux2_model.apply_rope = compact_apply_rope
    original_timestep_embedding = flux2_model.timestep_embedding

    def precise_timestep_embedding(t, *args, **kwargs):
        # Keep the engine's fractional timestep in float32 through the
        # high-frequency sin/cos calculation, then match the model weights.
        return original_timestep_embedding(t.float(), *args, **kwargs).to(dtype=dtype)

    flux2_model.timestep_embedding = precise_timestep_embedding
    if safe_fp16:
        if dtype != torch.float16:
            raise ValueError("--safe-fp16 requires FP16 weights and activations")

        def double_residuals(self, img, txt, img_attn, txt_attn, mods):
            (img_gate1, img_shift2, img_scale2, img_gate2,
             txt_gate1, txt_shift2, txt_scale2, txt_gate2) = mods
            # The BF16 checkpoint can produce gated residuals beyond FP16's
            # finite range. Keep the residual stream in FP32 while retaining
            # compact FP16 linear weights and attention projections.
            img = img.float() + img_gate1.float() * self.img_attn.proj(img_attn).float()
            img_mod = (1 + img_scale2.float()) * self.img_norm2(img) + img_shift2.float()
            img = img + img_gate2.float() * self.img_mlp(img_mod).float()
            txt = txt.float() + txt_gate1.float() * self.txt_attn.proj(txt_attn).float()
            txt_mod = (1 + txt_scale2.float()) * self.txt_norm2(txt) + txt_shift2.float()
            txt = txt + txt_gate2.float() * self.txt_mlp(txt_mod).float()
            return img, txt

        def single_out(self, x, attn, mlp, mod_gate):
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
            return x.float() + mod_gate.float() * output.float()

        flux2_model.DoubleStreamBlock._apply_residuals = double_residuals
        flux2_model.SingleStreamBlock._out = single_out
    with torch.device("meta"):
        model = flux2_model.Flux2(flux2_model.Klein4BParams())
    expected = dict(model.named_parameters())
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        actual_names = set(handle.keys())
        if actual_names != set(expected):
            missing = sorted(set(expected) - actual_names)
            unexpected = sorted(actual_names - set(expected))
            raise ValueError(f"checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        for name, parameter in expected.items():
            # load_only(name) selects the weights one chunk uses; the others stay on the meta
            # device, so a 16 GB Mac can export the 4B model chunk by chunk.
            if load_only is not None and not load_only(name):
                continue
            value = handle.get_tensor(name)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"weight shape mismatch for {name}")
            module_name, attribute = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            setattr(module, attribute,
                    torch.nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False))
            del value
    if rescale:
        rescale_for_fp16(model, *rescale, torch)
    if safe_fp16:
        from types import MethodType

        def cast_linear_input(self, x):
            return torch.nn.functional.linear(x.to(self.weight.dtype), self.weight, self.bias)

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = MethodType(cast_linear_input, module)
    return model.eval()


def make_wrapper(model, height, width, text_tokens, device, torch):
    class EngineInputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            image_ids, text_ids = position_ids(height, width, text_tokens, torch, device)
            self.register_buffer("image_ids", image_ids, persistent=False)
            self.register_buffer("text_ids", text_ids, persistent=False)

        def forward(self, latent, timesteps, context):
            # Core ML IO is f32 and reverse-ggml shape. The transformer uses
            # sequence-major tokens and the checkpoint's lower-precision dtype.
            dtype = self.model.img_in.weight.dtype
            x = latent.permute(0, 2, 3, 1).reshape(1, height * width, EXPECTED_CHANNELS).to(dtype)
            ctx = context.reshape(1, text_tokens, EXPECTED_CONTEXT_WIDTH).to(dtype)
            t = timesteps.reshape(1)
            out = self.model(x, self.image_ids, t, ctx, self.text_ids, None)
            return out.reshape(1, height, width, EXPECTED_CHANNELS).permute(0, 3, 1, 2).float()

    return EngineInputs().eval()


def chunk_bounds(cuts):
    bounds = [0, *sorted(cuts), TOTAL_BLOCKS]
    if len(set(bounds)) != len(bounds) or bounds[0] < 0 or bounds[-1] > TOTAL_BLOCKS:
        raise ValueError(f"cuts must be distinct block indices between 1 and {TOTAL_BLOCKS - 1}")
    return list(zip(bounds[:-1], bounds[1:]))


def chunk_weight_filter(lo, hi):
    """Names of the checkpoint weights blocks [lo, hi) and their shared parts use."""
    shared = (("time_in.", "img_in.", "txt_in.") if lo == 0 else ()) + \
             (("double_stream_modulation_img.", "double_stream_modulation_txt.") if lo < DOUBLE_BLOCKS else ()) + \
             (("single_stream_modulation.",) if hi > DOUBLE_BLOCKS else ()) + \
             (("final_layer.",) if hi == TOTAL_BLOCKS else ())
    blocks = tuple(f"double_blocks.{i}." for i in range(lo, min(hi, DOUBLE_BLOCKS))) + \
        tuple(f"single_blocks.{j}." for j in range(max(lo, DOUBLE_BLOCKS) - DOUBLE_BLOCKS, hi - DOUBLE_BLOCKS))
    return lambda name: name.startswith(shared + blocks)


def chunk_io(lo, hi):
    """Input and output names of the chunk for blocks [lo, hi).

    A boundary inside the double blocks passes img_<k> and txt_<k>; one after them passes
    hidden_<k>. The first chunk also returns vec, which every later chunk takes.
    """
    def edge(k):
        return [f"img_{k}", f"txt_{k}"] if k <= DOUBLE_BLOCKS else [f"hidden_{k}"]

    inputs = ["latent", "timesteps", "context"] if lo == 0 else edge(lo) + ["vec"]
    outputs = ["output"] if hi == TOTAL_BLOCKS else edge(hi) + (["vec"] if lo == 0 else [])
    return inputs, outputs


def make_chunk(model, lo, hi, height, width, text_tokens, torch):
    """The engine-facing denoiser restricted to blocks [lo, hi), with chunk_io's inputs and outputs."""
    from flux2 import model as flux2_model

    image_ids, text_ids = position_ids(height, width, text_tokens, torch, "cpu")

    class Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model
            self.register_buffer("image_ids", image_ids, persistent=False)
            self.register_buffer("text_ids", text_ids, persistent=False)

        def forward(self, *args):
            m = self.m
            if lo == 0:
                latent, timesteps, context = args
                dtype = m.img_in.weight.dtype
                x = latent.permute(0, 2, 3, 1).reshape(1, height * width, EXPECTED_CHANNELS).to(dtype)
                ctx = context.reshape(1, text_tokens, EXPECTED_CONTEXT_WIDTH).to(dtype)
                vec = m.time_in(flux2_model.timestep_embedding(timesteps.reshape(1), 256))
                state = (m.img_in(x), m.txt_in(ctx))
            else:
                *state, vec = args
            pe_x, pe_ctx = m.pe_embedder(self.image_ids), m.pe_embedder(self.text_ids)
            if lo < DOUBLE_BLOCKS:
                img, txt = state
                mod_img = m.double_stream_modulation_img(vec)
                mod_txt = m.double_stream_modulation_txt(vec)
                for block in m.double_blocks[lo:min(hi, DOUBLE_BLOCKS)]:
                    img, txt, _ = block.forward_kv_extract(img, txt, pe_x, pe_ctx, mod_img, mod_txt,
                                                           num_ref_tokens=0)
                if hi <= DOUBLE_BLOCKS:
                    return (img, txt, vec) if lo == 0 else (img, txt)
                state = (torch.cat((txt, img), dim=1),)
            x = state[0]
            pe = torch.cat((pe_ctx, pe_x), dim=2)
            mod, _ = m.single_stream_modulation(vec)
            for block in m.single_blocks[max(lo, DOUBLE_BLOCKS) - DOUBLE_BLOCKS:hi - DOUBLE_BLOCKS]:
                x, _ = block.forward_kv_extract(x, pe, mod, text_tokens, num_ref_tokens=0)
            if hi < TOTAL_BLOCKS:
                return (x, vec) if lo == 0 else x
            x = m.final_layer(x[:, text_tokens:, ...], vec)
            return x.reshape(1, height, width, EXPECTED_CHANNELS).permute(0, 3, 1, 2).float()

    return Chunk().eval()


def timestep_only_ops(program, input_name):
    """Names of the ops before any matmul whose value depends only on input_name and constants.

    These are the sinusoidal timestep embedding: in FP16 a timestep of about 1000 has a
    resolution of 0.5, which makes its sines wrong, so they stay FP32 (on the CPU).
    """
    depends = {input_name}
    keep = set()
    for op in program.functions["main"].operations:
        if op.op_type in ("linear", "matmul", "conv"):
            continue
        values = []
        for value in op.inputs.values():
            values.extend(value if isinstance(value, (list, tuple)) else [value])
        computed = [v for v in values if getattr(v, "op", None) is not None and v.op.op_type != "const"]
        if (computed and all(v.name in depends for v in computed)) or \
                (not computed and any(getattr(v, "name", None) == input_name for v in values)):
            keep.add(op.name)
            depends.update(o.name for o in op.outputs)
    return keep


def read_fixture_array(fixture, manifest, name, np):
    entry = manifest["tensors"][name]
    if entry["type"] != "f32":
        raise ValueError(f"{name} must be float32")
    return np.fromfile(fixture / entry["file"], dtype=np.float32).reshape(
        tuple(reversed(entry["ne"])))


class NonFiniteActivation(RuntimeError):
    def __init__(self, report):
        super().__init__(report["module"])
        self.report = report


def watch_first_nonfinite(model, torch):
    """Stop at the first invalid module or unwrapped double-block residual."""
    handles = []

    def tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)

    def summary(values):
        result = []
        for value in tensors(values):
            if not value.is_floating_point():
                continue
            finite = torch.isfinite(value)
            item = {"shape": list(value.shape), "dtype": str(value.dtype),
                    "finite_values": int(finite.sum().item()),
                    "total_values": value.numel()}
            if finite.any().item():
                good = value[finite].float()
                item["finite_abs_max"] = float(good.abs().max().item())
            result.append(item)
        return result

    def make_hook(name):
        def hook(module, inputs, output):
            if all(torch.isfinite(value).all().item()
                   for value in tensors(output) if value.is_floating_point()):
                return
            raise NonFiniteActivation({"module": name, "type": type(module).__name__,
                                       "inputs": summary(inputs), "outputs": summary(output)})
        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(make_hook(name or "<root>")))

    # Flux2.forward calls DoubleStreamBlock.forward_kv_extract directly, so
    # ordinary module hooks cannot observe the residual arithmetic inside it.
    # Inspect those operations separately without retaining their activations.
    blocks = model.model.double_blocks
    block_type = type(blocks[0])
    original_apply_residuals = block_type._apply_residuals
    block_names = {id(block): f"model.double_blocks.{index}"
                   for index, block in enumerate(blocks)}

    def checked_apply_residuals(self, img, txt, img_attn, txt_attn, mods):
        (img_gate1, img_shift2, img_scale2, img_gate2,
         txt_gate1, txt_shift2, txt_scale2, txt_gate2) = mods
        name = block_names[id(self)]

        def checked(operation, inputs, output):
            if not torch.isfinite(output).all().item():
                raise NonFiniteActivation({"module": name, "type": "DoubleStreamBlock",
                                           "operation": operation,
                                           "inputs": summary(inputs),
                                           "outputs": summary(output)})
            return output

        img_proj = self.img_attn.proj(img_attn)
        img_weighted = checked("image attention gate multiply",
                               (img_gate1, img_proj), img_gate1 * img_proj)
        img = checked("image attention residual add", (img, img_weighted), img + img_weighted)
        img_norm = self.img_norm2(img)
        img_mod = checked("image MLP modulation", (img_norm, img_scale2, img_shift2),
                          (1 + img_scale2) * img_norm + img_shift2)
        img_mlp = self.img_mlp(img_mod)
        img_weighted = checked("image MLP gate multiply", (img_gate2, img_mlp),
                               img_gate2 * img_mlp)
        img = checked("image MLP residual add", (img, img_weighted), img + img_weighted)

        txt_proj = self.txt_attn.proj(txt_attn)
        txt_weighted = checked("text attention gate multiply",
                               (txt_gate1, txt_proj), txt_gate1 * txt_proj)
        txt = checked("text attention residual add", (txt, txt_weighted), txt + txt_weighted)
        txt_norm = self.txt_norm2(txt)
        txt_mod = checked("text MLP modulation", (txt_norm, txt_scale2, txt_shift2),
                          (1 + txt_scale2) * txt_norm + txt_shift2)
        txt_mlp = self.txt_mlp(txt_mod)
        txt_weighted = checked("text MLP gate multiply", (txt_gate2, txt_mlp),
                               txt_gate2 * txt_mlp)
        txt = checked("text MLP residual add", (txt, txt_weighted), txt + txt_weighted)
        return img, txt

    block_type._apply_residuals = checked_apply_residuals

    def cleanup():
        block_type._apply_residuals = original_apply_residuals
        for handle in handles:
            handle.remove()

    return cleanup


def export_chunk(args, parser, rescale, latent_shape, timestep_shape, context_shape, np, torch):
    import shutil

    try:
        chunks = chunk_bounds(int(c) for c in args.cuts.split(","))
    except ValueError as exc:
        parser.error(f"--cuts: {exc}")
    if not args.chunk or not 1 <= args.chunk <= len(chunks):
        parser.error(f"--chunk must be between 1 and {len(chunks)}")
    if not args.source or not args.weights or not args.weights.is_file() or not args.output:
        parser.error("chunk requires an existing --source and --weights, and --output")
    final = args.output / f"chunk{args.chunk}.mlmodelc"
    if final.exists():
        parser.error(f"{final} already exists")
    import coremltools as ct

    lo, hi = chunks[args.chunk - 1]
    height, width, text_tokens = latent_shape[2], latent_shape[3], context_shape[2]
    print(f"Chunk {args.chunk} of {len(chunks)}: blocks {lo}-{hi - 1}; loading its weights", flush=True)
    started = time.perf_counter()
    model = load_official_model(args.source, args.weights.resolve(), torch.float16, "cpu", torch,
                                rescale=rescale, split_attention_heads=not args.fused_attention,
                                load_only=chunk_weight_filter(lo, hi))
    module = make_chunk(model, lo, hi, height, width, text_tokens, torch)
    in_names, out_names = chunk_io(lo, hi)
    hidden = model.hidden_size
    shapes = {"latent": latent_shape, "timesteps": timestep_shape, "context": context_shape,
              "img": (1, height * width, hidden), "txt": (1, text_tokens, hidden),
              "hidden": (1, text_tokens + height * width, hidden), "vec": (1, hidden)}

    def shape(name):
        return shapes[name] if name in shapes else shapes[name.split("_")[0]]

    if lo == 0:
        example = tuple(torch.from_numpy(read_fixture_array(args.fixture.resolve(), validate_capture(
            args.fixture.resolve()), name, np).copy()) for name in in_names)
    else:
        example = tuple(torch.zeros(*shape(name), dtype=torch.float16) for name in in_names)
    with torch.inference_mode():
        traced = torch.jit.trace(module, example, strict=False, check_trace=False)
    # Engine inputs and the final output are FP32; tensors between chunks stay FP16.
    inputs = [ct.TensorType(name=name, shape=tuple(shape(name)),
                            dtype=np.float32 if lo == 0 else np.float16) for name in in_names]
    outputs = [ct.TensorType(name=name, dtype=np.float32 if name == "output" else np.float16) for name in out_names]
    target = ct.target.macOS14  # iOS 17 op set: iOS 18 op set models have failed to load on the Neural Engine
    fp32_ops = set()
    if lo == 0:
        program = ct.convert(traced, convert_to="milinternal", inputs=inputs, minimum_deployment_target=target)
        fp32_ops = timestep_only_ops(program, "timesteps")
        del program
    converted = ct.convert(
        traced, convert_to="mlprogram", inputs=inputs, outputs=outputs,
        compute_precision=ct.transform.FP16ComputePrecision(op_selector=lambda op: op.name not in fp32_ops),
        minimum_deployment_target=target, skip_model_load=True)
    del traced, module, model
    gc.collect()
    print(f"Converted in {time.perf_counter() - started:.0f}s", flush=True)
    if args.palette_bits:
        started = time.perf_counter()
        config = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=args.palette_bits, weight_threshold=2048,
                                                       num_kmeans_workers=args.kmeans_workers)
        converted = ct.optimize.coreml.palettize_weights(converted, ct.optimize.coreml.OptimizationConfig(config))
        print(f"Palettized to {args.palette_bits} bits in {time.perf_counter() - started:.0f}s", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / f".chunk{args.chunk}.mlpackage"
    shutil.rmtree(package, ignore_errors=True)
    converted.save(str(package))
    del converted
    ct.utils.compile_model(str(package), str(final))
    shutil.rmtree(package, ignore_errors=True)
    size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file()) / 1e9
    print(f"Saved {final} ({size:.2f} GB)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "trace", "convert", "compile", "chunk"))
    parser.add_argument("--source", type=Path, help="checkout of black-forest-labs/flux2")
    parser.add_argument("--weights", type=Path, help="FLUX.2-klein 4B safetensors")
    parser.add_argument("--fixture", type=Path, help="captured GGML denoiser invocation")
    parser.add_argument("--output", type=Path, help="new .pt, .mlpackage, or .mlmodelc path")
    parser.add_argument("--trace", type=Path, help="input .pt for conversion")
    parser.add_argument("--package", type=Path, help="input .mlpackage for compile")
    parser.add_argument("--max-nrmse", type=float, default=0.03)
    parser.add_argument("--finite-only", action="store_true",
                        help="check finite PyTorch output without comparing to the fixture output (check mode)")
    parser.add_argument("--locate-nonfinite", action="store_true",
                        help="stop at the first non-finite module output (check mode)")
    parser.add_argument("--check-dtype", choices=("bf16", "fp16", "fp32"), default="bf16",
                        help="weight and activation precision for check; trace always uses fp16")
    parser.add_argument("--safe-fp16", action="store_true",
                        help="keep residual streams in FP32 with FP16 linear weights (check/trace)")
    parser.add_argument("--fp16-rescale", metavar="STREAM[,SWIGLU]",
                        help="divide the residual streams (and SwiGLU products) by these powers of two, "
                             "an exact rewrite that keeps FP16 activations finite (check/trace)")
    parser.add_argument("--chunk", type=int, help="chunk: which chunk to export (1-based)")
    parser.add_argument("--cuts", default=",".join(map(str, DEFAULT_CUTS)),
                        help="chunk: cut before these of the 25 blocks (double 0-4, single 5-24)")
    parser.add_argument("--palette-bits", type=int, choices=(0, 4, 6, 8), default=8,
                        help="chunk: k-means palette bits per weight; 0 keeps FP16 weights")
    parser.add_argument("--kmeans-workers", type=int, default=2,
                        help="chunk: k-means worker processes for palettization")
    parser.add_argument("--fused-attention", action="store_true",
                        help="chunk: one attention op over all heads (wrong on M2 Neural Engines)")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="PyTorch device for check/trace; use mps if CPU inference is too slow")
    args = parser.parse_args(argv)
    if args.finite_only and args.mode != "check":
        parser.error("--finite-only is supported only in check mode")
    if args.locate_nonfinite and (args.mode != "check" or not args.finite_only):
        parser.error("--locate-nonfinite requires check --finite-only")
    if args.safe_fp16 and (args.mode not in ("check", "trace") or
                          (args.mode == "check" and args.check_dtype != "fp16")):
        parser.error("--safe-fp16 requires check --check-dtype fp16 or trace")
    if args.safe_fp16 and args.locate_nonfinite:
        parser.error("--safe-fp16 and --locate-nonfinite cannot be combined")
    rescale = None
    if args.mode == "chunk" and not args.fp16_rescale:
        args.fp16_rescale = "8"
    if args.fp16_rescale:
        if args.mode not in ("check", "trace", "chunk"):
            parser.error("--fp16-rescale applies to check, trace and chunk")
        try:
            rescale = tuple(float(v) for v in args.fp16_rescale.split(","))
            rescale = rescale + (1.0,) * (2 - len(rescale))
            if len(rescale) != 2:
                raise ValueError
        except ValueError:
            parser.error("--fp16-rescale takes STREAM or STREAM,SWIGLU")

    if args.mode == "compile":
        if not args.package or not args.package.is_dir() or not args.output:
            parser.error("compile requires --package and --output")
        if args.output.exists() or args.output.suffix != ".mlmodelc":
            parser.error("compile output must be a new .mlmodelc directory")
        from coremltools.models.utils import compile_model
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(compile_model(str(args.package.resolve()), str(args.output.resolve())), flush=True)
        return

    if not args.fixture:
        parser.error("check/trace/convert/chunk require --fixture")
    try:
        manifest = validate_capture(args.fixture.resolve())
        latent_shape, timestep_shape, context_shape = fixture_shapes(manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.max_nrmse <= 0:
        parser.error("--max-nrmse must be positive")

    import numpy as np
    import torch

    torch.set_grad_enabled(False)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable in this environment")
    fixture = args.fixture.resolve()
    if args.mode == "chunk":
        export_chunk(args, parser, rescale, latent_shape, timestep_shape, context_shape, np, torch)
        return
    if args.mode in ("check", "trace"):
        if not args.source or not args.weights or not args.weights.is_file():
            parser.error("check/trace require an existing --source and --weights")
        arrays = tuple(read_fixture_array(fixture, manifest, name, np)
                       for name in ("latent", "timesteps", "context"))
        inputs = tuple(torch.from_numpy(array.copy()).to(args.device) for array in arrays)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.check_dtype] \
            if args.mode == "check" else torch.float16
        print(f"Loading complete FLUX.2-klein 4B checkpoint as {dtype} on {args.device}", flush=True)
        started = time.perf_counter()
        model = load_official_model(args.source, args.weights.resolve(), dtype, args.device, torch,
                                    safe_fp16=args.safe_fp16, rescale=rescale)
        print(f"Checkpoint loaded in {time.perf_counter() - started:.1f}s", flush=True)
        wrapper = make_wrapper(model, latent_shape[2], latent_shape[3], context_shape[2], args.device, torch)

        if args.mode == "check":
            print("Running one complete PyTorch denoiser call", flush=True)
            started = time.perf_counter()
            cleanup = watch_first_nonfinite(wrapper, torch) if args.locate_nonfinite else None
            try:
                with torch.inference_mode():
                    actual = wrapper(*inputs).cpu().numpy()
            except NonFiniteActivation as exc:
                print(json.dumps({"mode": "locate-nonfinite", **exc.report}, indent=2), flush=True)
                raise SystemExit(1) from None
            finally:
                if cleanup is not None:
                    cleanup()
            print(f"PyTorch denoiser completed in {time.perf_counter() - started:.1f}s", flush=True)
            if args.finite_only:
                finite = np.isfinite(actual)
                print(json.dumps({"mode": "check", "dtype": str(dtype),
                                  "finite_values": int(finite.sum()),
                                  "total_values": int(finite.size),
                                  "passes_finite_check": bool(finite.all())}, indent=2), flush=True)
                if not finite.all():
                    raise SystemExit(1)
                return
            reference = read_fixture_array(fixture, manifest, "output", np)
            metrics = error_metrics(actual, reference, np)
            print(json.dumps({"mode": "check", "dtype": str(dtype), **metrics}, indent=2), flush=True)
            if metrics["normalized_rmse"] > args.max_nrmse:
                raise SystemExit("PyTorch output does not match the GGML fixture")
            return

        if not args.output or args.output.suffix != ".pt" or args.output.exists():
            parser.error("trace requires a new --output path ending in .pt")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print("Tracing one complete PyTorch denoiser call", flush=True)
        started = time.perf_counter()
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, inputs, strict=True, check_trace=False)
        print(f"PyTorch tracing completed in {time.perf_counter() - started:.1f}s", flush=True)
        del wrapper, model
        gc.collect()
        print("Checking traced denoiser against the GGML fixture", flush=True)
        with torch.inference_mode():
            actual = traced(*inputs).cpu().numpy()
        reference = read_fixture_array(fixture, manifest, "output", np)
        metrics = error_metrics(actual, reference, np)
        print(json.dumps({"mode": "trace", "dtype": str(dtype), **metrics}, indent=2), flush=True)
        if metrics["normalized_rmse"] > args.max_nrmse:
            raise SystemExit("traced model does not match the GGML fixture")
        traced.save(str(args.output.resolve()))
        print(f"Saved {args.output.resolve()}", flush=True)
        return

    if not args.trace or not args.trace.is_file():
        parser.error("convert requires an existing --trace .pt file")
    if not args.output or args.output.suffix != ".mlpackage" or args.output.exists():
        parser.error("convert requires a new --output path ending in .mlpackage")
    import coremltools as ct

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.load(str(args.trace.resolve()), map_location="cpu")
    gc.collect()
    print("Converting traced denoiser to an ML Program", flush=True)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        # Keep the official model's explicit float32 normalization, RoPE, and
        # timestep math. The all-float16 ML Program produced non-finite values
        # on the second denoising step of a four-step M4 run.
        compute_precision=ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="latent", shape=latent_shape, dtype=np.float32),
            ct.TensorType(name="timesteps", shape=timestep_shape, dtype=np.float32),
            ct.TensorType(name="context", shape=context_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        package_dir=str(args.output.resolve()),
        skip_model_load=True,
    )
    if not args.output.is_dir():
        raise RuntimeError("Core ML conversion returned without writing the .mlpackage")
    del converted
    print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
