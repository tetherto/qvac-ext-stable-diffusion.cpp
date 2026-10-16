#!/usr/bin/env python3
"""Export and validate a mixed-precision FLUX.2 VAE decoder without PyTorch."""

import argparse
import json
import math
import shutil
import statistics
import time
from pathlib import Path

from compare_flux_coreml_fixture import error_metrics


def validate_dimensions(width, height):
    if any(type(value) is not int or value < 16 or value % 16 for value in (width, height)):
        raise ValueError("image dimensions must be positive multiples of 16")


def validate_fixture(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or manifest.get("kind") != "flux2-vae-decode":
        raise ValueError("expected a FLUX.2 VAE decode fixture")
    tensors = manifest["tensors"]
    for name in ("latent", "output"):
        entry = tensors[name]
        shape = entry["ne"]
        if entry["type"] != "f32" or len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f"invalid float32 dimensions for {name}")
        file = directory / entry["file"]
        expected = math.prod(shape) * 4
        if file.resolve().parent != directory or not file.is_file() or file.stat().st_size != expected or entry["bytes"] != expected:
            raise ValueError(f"invalid tensor file or size for {name}")
    w, h, c, n = tensors["latent"]["ne"]
    if c != 128 or n != 1 or tensors["output"]["ne"] != [w * 16, h * 16, 3, 1]:
        raise ValueError("fixture must map [W,H,128,1] to [16*W,16*H,3,1]")
    return manifest


def build_decoder(weights, width, height):
    import numpy as np
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    validate_dimensions(width, height)
    consumed = set()
    conv_type, conv_dtype = np.float16, "fp16"
    norm_type, norm_dtype = np.float32, "fp32"

    def weight(name, dtype):
        consumed.add(name)
        value = weights[name].astype(dtype)
        if not np.isfinite(value).all():
            raise ValueError(f"decoder weight is non-finite at {name}")
        return value

    def cast(x, dtype):
        return mb.cast(x=x, dtype=dtype)

    def conv(x, name):
        w = weight(name + ".weight", conv_type)
        b = weight(name + ".bias", conv_type)
        return mb.conv(x=cast(x, conv_dtype), weight=w, bias=b, pad_type="custom",
                       pad=[w.shape[2] // 2] * 2 + [w.shape[3] // 2] * 2,
                       name=name.replace(".", "_"))

    def norm(x, name):
        n, c, h, w = x.shape
        y = mb.reshape(x=cast(x, norm_dtype), shape=[n, 32, c // 32, h, w])
        mean = mb.reduce_mean(x=y, axes=[2, 3, 4], keep_dims=True)
        centered = mb.sub(x=y, y=mean)
        variance = mb.reduce_mean(x=mb.mul(x=centered, y=centered), axes=[2, 3, 4], keep_dims=True)
        y = mb.mul(x=centered, y=mb.rsqrt(x=mb.add(x=variance, y=norm_type(1e-6))))
        y = mb.reshape(x=y, shape=[n, c, h, w])
        y = mb.mul(x=y, y=weight(name + ".weight", norm_type).reshape(1, c, 1, 1))
        return mb.add(x=y, y=weight(name + ".bias", norm_type).reshape(1, c, 1, 1))

    def silu(x):
        return mb.mul(x=x, y=mb.sigmoid(x=x))

    def residual(x, name):
        h = conv(silu(norm(x, name + ".norm1")), name + ".conv1")
        h = conv(silu(norm(h, name + ".norm2")), name + ".conv2")
        if name + ".nin_shortcut.weight" in weights:
            x = conv(x, name + ".nin_shortcut")
        return mb.add(x=cast(h, norm_dtype), y=cast(x, norm_dtype))

    def attention(x):
        name = "decoder.mid.attn_1"
        n, c, h, w = x.shape
        y = norm(x, name + ".norm")
        qkv = []
        for projection in ["q", "k", "v"]:
            p = conv(y, name + "." + projection)
            p = mb.reshape(x=p, shape=[n, c, h * w])
            qkv.append(cast(mb.transpose(x=p, perm=[0, 2, 1]), norm_dtype))
        y = mb.scaled_dot_product_attention(query=qkv[0], key=qkv[1], value=qkv[2])
        y = mb.reshape(x=mb.transpose(x=y, perm=[0, 2, 1]), shape=[n, c, h, w])
        y = conv(y, name + ".proj_out")
        return mb.add(x=cast(x, norm_dtype), y=cast(y, norm_dtype))

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 128, height // 16, width // 16), dtype=types.fp32)],
                opset_version=ct.target.macOS15)
    def decoder(latent):
        x = mb.pixel_shuffle(x=latent, upscale_factor=2)
        x = conv(x, "decoder.post_quant_conv")
        x = conv(x, "decoder.conv_in")
        x = residual(x, "decoder.mid.block_1")
        x = attention(x)
        x = residual(x, "decoder.mid.block_2")
        for level in reversed(range(4)):
            for block in range(3):
                x = residual(x, f"decoder.up.{level}.block.{block}")
            if level:
                x = mb.upsample_nearest_neighbor(x=x, scale_factor_height=2.0, scale_factor_width=2.0)
                x = conv(x, f"decoder.up.{level}.upsample.conv")
        x = conv(silu(norm(x, "decoder.norm_out")), "decoder.conv_out")
        return mb.cast(x=x, dtype="fp32", name="output")

    expected = {k for k in weights if k.startswith("decoder.")}
    if consumed != expected:
        raise ValueError(f"unconsumed decoder weights: {expected - consumed}; extra: {consumed - expected}")
    return decoder


def export_model(args):
    import coremltools as ct
    from safetensors import safe_open

    width, height = args.width, args.height
    if args.fixture:
        entry = validate_fixture(args.fixture)["tensors"]["latent"]
        width, height = entry["ne"][0] * 16, entry["ne"][1] * 16
    validate_dimensions(width, height)
    if args.output.exists():
        raise ValueError("export output already exists; use a fresh .mlpackage path")
    if args.output.suffix != ".mlpackage":
        raise ValueError("export output must end in .mlpackage")
    with safe_open(str(args.weights), framework="numpy") as checkpoint:
        weights = {name: checkpoint.get_tensor(name) for name in checkpoint.keys() if name.startswith("decoder.")}
    decoder = build_decoder(weights, width, height)
    model = ct.convert(decoder, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32, skip_model_load=True)
    model.user_defined_metadata["sdcpp.flux2_vae.input"] = "denormalized packed latents"
    model.user_defined_metadata["sdcpp.flux2_vae.precision"] = "fp16 convolutions; fp32 normalization, attention, residuals"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"Saved mixed-precision {width}x{height} VAE decoder: {args.output}", flush=True)


def compile_model(args):
    import coremltools as ct

    if not args.package.is_dir() or args.package.suffix != ".mlpackage":
        raise ValueError("compile requires an existing .mlpackage")
    if args.output.exists() or args.output.suffix != ".mlmodelc":
        raise ValueError("use a fresh .mlmodelc output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compiled = ct.models.utils.compile_model(str(args.package.resolve()))
    shutil.copytree(compiled, args.output)
    print(f"Compiled VAE decoder: {args.output}", flush=True)


def check_model(args):
    import coremltools as ct
    import numpy as np

    if not args.model.is_dir() or args.model.suffix != ".mlmodelc":
        raise ValueError("check requires an existing compiled .mlmodelc")
    if args.runs < 1 or args.warmup < 0 or not math.isfinite(args.max_nrmse) or args.max_nrmse <= 0:
        raise ValueError("runs must be positive; warmup nonnegative; max-nrmse finite and positive")
    if args.output.exists():
        raise ValueError("check report already exists; use a fresh output path")
    manifest = validate_fixture(args.fixture)
    tensors = manifest["tensors"]
    def read(name):
        entry = tensors[name]
        return np.fromfile(args.fixture / entry["file"], dtype=np.float32).reshape(tuple(reversed(entry["ne"])))
    latent, reference = read("latent"), read("output")
    if not np.isfinite(latent).all():
        raise ValueError("latent fixture contains a non-finite value")
    started = time.perf_counter()
    model = ct.models.CompiledMLModel(str(args.model.resolve()), compute_units=getattr(ct.ComputeUnit, args.compute_units))
    load_s = time.perf_counter() - started
    durations = []
    for index in range(args.warmup + args.runs):
        started = time.perf_counter()
        actual = model.predict({"latent": latent})["output"]
        elapsed = time.perf_counter() - started
        metrics = error_metrics(actual, reference, np)
        if index >= args.warmup:
            durations.append(elapsed)
        print(f"VAE call {index + 1}: {elapsed:.4f}s", flush=True)
    rgb_error = np.clip((actual.astype(np.float64) + 1) / 2, 0, 1) - np.clip((reference.astype(np.float64) + 1) / 2, 0, 1)
    mse = float(np.mean(rgb_error ** 2))
    report = {"model": str(args.model.resolve()), "fixture": str(args.fixture.resolve()),
              "compute_units": args.compute_units, "load_s": load_s,
              "warmup": args.warmup, "prediction_times_s": durations,
              "median_prediction_s": statistics.median(durations), **metrics,
              "rgb_psnr_db": -10 * math.log10(mse) if mse else None,
              "passes_reference": metrics["normalized_rmse"] <= args.max_nrmse}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passes_reference"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="build a fixed-shape mixed-precision MIL decoder")
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--width", type=int, default=1024)
    export.add_argument("--height", type=int, default=1024)
    export.add_argument("--fixture", type=Path, help="derive dimensions from a captured decode instead")
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--package", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="compare raw RGB and time a compiled Core ML decoder")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--fixture", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--compute-units", choices=["CPU_AND_GPU", "ALL", "CPU_AND_NE"], default="CPU_AND_GPU")
    check.add_argument("--warmup", type=int, default=1)
    check.add_argument("--runs", type=int, default=5)
    check.add_argument("--max-nrmse", type=float, default=0.001)
    args = parser.parse_args(argv)
    try:
        {"export": export_model, "compile": compile_model, "check": check_model}[args.command](args)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
