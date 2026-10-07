#!/usr/bin/env python3
"""Validate and export the complete FLUX.2-klein 4B denoiser for Core ML.

The implementation is imported from Black Forest Labs' official flux2 source
checkout. This script only adapts the engine's fixed-shape tensor interface.
Run each mode in a separate process to limit peak memory.
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


def load_official_model(source, weights, dtype, device, torch):
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
            value = handle.get_tensor(name)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"weight shape mismatch for {name}")
            module_name, attribute = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            setattr(module, attribute,
                    torch.nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False))
            del value
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
            t = timesteps.reshape(1).to(dtype)
            out = self.model(x, self.image_ids, t, ctx, self.text_ids, None)
            return out.reshape(1, height, width, EXPECTED_CHANNELS).permute(0, 3, 1, 2).float()

    return EngineInputs().eval()


def read_fixture_array(fixture, manifest, name, np):
    entry = manifest["tensors"][name]
    if entry["type"] != "f32":
        raise ValueError(f"{name} must be float32")
    return np.fromfile(fixture / entry["file"], dtype=np.float32).reshape(
        tuple(reversed(entry["ne"])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "trace", "convert", "compile"))
    parser.add_argument("--source", type=Path, help="checkout of black-forest-labs/flux2")
    parser.add_argument("--weights", type=Path, help="FLUX.2-klein 4B safetensors")
    parser.add_argument("--fixture", type=Path, help="captured GGML denoiser invocation")
    parser.add_argument("--output", type=Path, help="new .pt, .mlpackage, or .mlmodelc path")
    parser.add_argument("--trace", type=Path, help="input .pt for conversion")
    parser.add_argument("--package", type=Path, help="input .mlpackage for compile")
    parser.add_argument("--max-nrmse", type=float, default=0.03)
    parser.add_argument("--check-dtype", choices=("bf16", "fp16"), default="bf16",
                        help="weight and activation precision for check; trace always uses fp16")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="PyTorch device for check/trace; use mps if CPU inference is too slow")
    args = parser.parse_args(argv)

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
        parser.error("check/trace/convert require --fixture")
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
    if args.mode in ("check", "trace"):
        if not args.source or not args.weights or not args.weights.is_file():
            parser.error("check/trace require an existing --source and --weights")
        arrays = tuple(read_fixture_array(fixture, manifest, name, np)
                       for name in ("latent", "timesteps", "context"))
        inputs = tuple(torch.from_numpy(array.copy()).to(args.device) for array in arrays)
        dtype = torch.bfloat16 if args.mode == "check" and args.check_dtype == "bf16" else torch.float16
        print(f"Loading complete FLUX.2-klein 4B checkpoint as {dtype} on {args.device}", flush=True)
        started = time.perf_counter()
        model = load_official_model(args.source, args.weights.resolve(), dtype, args.device, torch)
        print(f"Checkpoint loaded in {time.perf_counter() - started:.1f}s", flush=True)
        wrapper = make_wrapper(model, latent_shape[2], latent_shape[3], context_shape[2], args.device, torch)

        if args.mode == "check":
            print("Running one complete PyTorch denoiser call", flush=True)
            started = time.perf_counter()
            with torch.inference_mode():
                actual = wrapper(*inputs).cpu().numpy()
            print(f"PyTorch denoiser completed in {time.perf_counter() - started:.1f}s", flush=True)
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
