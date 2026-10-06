#!/usr/bin/env python3
"""Compare a compiled FLUX Core ML denoiser with one captured ggml call."""

import argparse
import json
import math
from pathlib import Path


TYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4}
REQUIRED = ("latent", "timesteps", "context", "output")


def validate_capture(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1:
        raise ValueError("unsupported capture format")
    tensors = manifest.get("tensors", {})
    for name in REQUIRED:
        if name not in tensors:
            raise ValueError(f"missing required tensor {name}")
    for name, entry in tensors.items():
        dtype = entry["type"]
        if dtype not in TYPE_BYTES:
            raise ValueError(f"unsupported type for {name}: {dtype}")
        shape = entry["ne"]
        if len(shape) != 4 or any(not isinstance(n, int) or n < 1 for n in shape):
            raise ValueError(f"invalid ggml dimensions for {name}: {shape}")
        expected = math.prod(shape) * TYPE_BYTES[dtype]
        file = directory / entry["file"]
        if file.parent != directory or not file.is_file():
            raise ValueError(f"missing tensor file for {name}: {file}")
        if file.stat().st_size != expected or entry["bytes"] != expected:
            raise ValueError(f"size mismatch for {name}: expected {expected} bytes")
    if tensors["latent"]["ne"] != tensors["output"]["ne"]:
        raise ValueError("denoiser output dimensions differ from latent dimensions")
    return manifest


def error_metrics(actual, reference, np):
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if actual.shape != reference.shape:
        raise ValueError(f"Core ML output shape {actual.shape} != ggml {reference.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
        raise ValueError("Core ML and ggml outputs must contain only finite values")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    target = reference.astype(np.float64)
    denominator = float(np.mean(target * target))
    if denominator == 0:
        raise ValueError("ggml output has zero energy")
    return {
        "normalized_rmse": float(np.sqrt(np.mean(delta * delta) / denominator)),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "max_abs_error": float(np.max(np.abs(delta))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="compiled .mlmodelc")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON comparison report")
    parser.add_argument("--inputs", default="latent,timesteps,context",
                        help="comma-separated Core ML inputs; add pooled or guidance if exported")
    parser.add_argument("--max-nrmse", type=float, default=0.03)
    args = parser.parse_args()
    try:
        import coremltools as ct
        import numpy as np
    except ImportError as exc:
        parser.error(f"missing Python dependency: {exc}; install numpy and coremltools")
    if not args.model.is_dir():
        parser.error(f"compiled Core ML model not found: {args.model}")
    if args.max_nrmse <= 0:
        parser.error("--max-nrmse must be positive")
    try:
        manifest = validate_capture(args.fixture.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    tensors = manifest["tensors"]
    names = args.inputs.split(",")
    if len(names) != len(set(names)) or not {"latent", "timesteps", "context"}.issubset(names) or \
            any(name not in ("latent", "timesteps", "context", "pooled", "guidance") for name in names):
        parser.error("--inputs must contain latent,timesteps,context and optional pooled,guidance")
    inputs = {}
    for name in names:
        if name not in tensors:
            parser.error(f"fixture does not contain Core ML input {name}")
        entry = tensors[name]
        if entry["type"] != "f32":
            parser.error(f"{name} must be float32 for this Core ML interface")
        shape = tuple(reversed(entry["ne"]))
        inputs[name] = np.fromfile(args.fixture / entry["file"], dtype=np.float32).reshape(shape)
    model = ct.models.CompiledMLModel(str(args.model.resolve()), compute_units=ct.ComputeUnit.ALL)
    prediction = model.predict(inputs)
    output_entry = tensors["output"]
    reference = np.fromfile(args.fixture / output_entry["file"], dtype=np.float32).reshape(
        tuple(reversed(output_entry["ne"])))
    try:
        metrics = error_metrics(prediction["output"], reference, np)
    except ValueError as exc:
        parser.error(str(exc))
    report = {"model": str(args.model.resolve()), "fixture": str(args.fixture.resolve()),
              "input_shapes": {name: list(value.shape) for name, value in inputs.items()},
              "output_shape": list(reference.shape), **metrics,
              "passes_reference": metrics["normalized_rmse"] <= args.max_nrmse}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passes_reference"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
