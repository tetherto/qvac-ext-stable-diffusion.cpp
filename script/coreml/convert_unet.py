"""Export a diffusers SD 1.x/2.x UNet to CoreML for sd-cli --coreml-unet.

The model takes float32 inputs and returns float32, with the names the runtime expects:
  sample [N, 4, H/8, W/8], timestep [N], encoder_hidden_states [N, T, D] -> noise_pred [N, 4, H/8, W/8]

Every input shape is fixed, because the Neural Engine only runs fixed-shape graphs: with
EnumeratedShapes or RangeDim the traced UNet keeps runtime reshapes and CoreML moves the whole
graph to the CPU. To support several resolutions with one file, pass several --sizes: each one
becomes a fixed-shape function of a multifunction model (iOS 18+), and all functions share one
copy of the weights. The runtime picks the function whose input shapes match.

Only list sizes the model was trained at. SDXS-512 is a one-step model distilled at 512x512 only:
at 512x768, 768x512 and 768x768 it duplicates features (extra eyes and noses) on every backend,
so this script refuses other sizes for it unless --allow-untrained-sizes is given. For other
models it warns when a size differs from the one in the UNet config. See docs/coreml.md.

Examples (SDXS-512-DreamShaper, fp16):
  python convert_unet.py --unet sdxs/unet --out sdxs_unet_fp16.mlpackage --compile
  python convert_unet.py --unet sdxs/unet --out sdxs_unet_multi.mlpackage --sizes 512x512,512x768,768x512 --compile

pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from diffusers import UNet2DConditionModel


def silu_via_tanh(self, x):
    # Same function as x * sigmoid(x). The Neural Engine's fp16 silu op (which coremltools also
    # fuses x * sigmoid(x) into) is inaccurate enough to visibly change one-step SDXS images
    # (cosine 0.99939 vs fp32 on M2), while this form stays on the ANE at cosine 0.999995.
    return 0.5 * x * (1.0 + torch.tanh(0.5 * x))


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states, return_dict=False)[0]


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
    parser.add_argument("--precision", choices=["fp16", "w8"], default="fp16",
                        help="fp16 weights and compute, or int8 weights (linear, per channel) with fp16 compute")
    parser.add_argument("--batch", type=int, default=1, help="UNet batch: 1, or 2 for batched CFG")
    parser.add_argument("--sizes", default="512x512",
                        help="comma-separated image sizes as WIDTHxHEIGHT, only ones the model was trained at "
                             "(SDXS: 512x512); more than one makes a multifunction model")
    parser.add_argument("--allow-untrained-sizes", action="store_true",
                        help="export sizes other than the model's native one even for SDXS (expect bad images)")
    parser.add_argument("--context-len", type=int, default=77, help="text tokens; sd.cpp uses 77 per prompt chunk")
    parser.add_argument("--target", choices=["iOS17", "iOS18"], default="iOS17",
                        help="op set. iOS18 uses the native scaled_dot_product_attention op, but a multifunction "
                             "model built with it fails to load on the ANE (macOS 27.2); iOS17 runs at the same speed")
    parser.add_argument("--compile", action="store_true", help="also write a compiled .mlmodelc next to the package")
    parser.add_argument("--keep-silu", action="store_true", help="keep the native silu op (inaccurate on the ANE)")
    args = parser.parse_args()

    sizes = [parse_size(s) for s in args.sizes.split(",")]
    if not args.keep_silu:
        torch.nn.SiLU.forward = silu_via_tanh

    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float32).eval()
    wrapper = UNetWrapper(unet).eval()
    cfg = unet.config
    check_sizes(cfg, sizes, args.allow_untrained_sizes)
    multi = len(sizes) > 1
    parts_dir = Path(tempfile.mkdtemp(prefix="unet_parts_")) if multi else None
    checks = []  # (function name or None, inputs, fp32 reference output)

    for width, height in sizes:
        shapes = {
            "sample": (args.batch, cfg.in_channels, height // 8, width // 8),
            "timestep": (args.batch,),
            "encoder_hidden_states": (args.batch, args.context_len, cfg.cross_attention_dim),
        }
        example = (
            torch.randn(shapes["sample"]),
            torch.full(shapes["timestep"], 999.0),
            torch.randn(shapes["encoder_hidden_states"]),
        )
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, strict=False)
            reference = wrapper(*example).numpy()

        t0 = time.time()
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name=name, shape=shape, dtype=np.float32) for name, shape in shapes.items()],
            outputs=[ct.TensorType(name="noise_pred", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=getattr(ct.target, args.target),
        )
        if args.precision == "w8":
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                                   granularity="per_channel", weight_threshold=2048)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel,
                                                                 ct.optimize.coreml.OptimizationConfig(op_config))
        print(f"{width}x{height}: converted in {time.time() - t0:.1f} s")

        inputs = {name: tensor.numpy() for name, tensor in zip(shapes, example)}
        if multi:
            mlmodel.save(str(parts_dir / f"{function_name(width, height)}.mlpackage"))
            checks.append((function_name(width, height), inputs, reference))
        else:
            mlmodel.short_description = f"UNet {Path(args.unet).resolve()} ({args.precision}) for sd-cli --coreml-unet"
            mlmodel.save(args.out)
            checks.append((None, inputs, reference))

    if multi:
        desc = ct.utils.MultiFunctionDescriptor()
        for width, height in sizes:
            name = function_name(width, height)
            desc.add_function(str(parts_dir / f"{name}.mlpackage"), src_function_name="main", target_function_name=name)
        desc.default_function_name = function_name(*sizes[0])
        ct.utils.save_multifunction(desc, args.out)
        shutil.rmtree(parts_dir)
        print(f"saved {len(sizes)} functions to {args.out}")

    # The coremltools loader rejects function_name on an .mlpackage, so check the compiled model.
    out_dir = Path(args.out).resolve().parent
    compiled = out_dir / (Path(args.out).stem + ".mlmodelc")
    if args.compile or multi:
        if compiled.exists():
            shutil.rmtree(compiled)
        subprocess.run(["xcrun", "coremlcompiler", "compile", args.out, str(out_dir)], check=True,
                       stdout=subprocess.DEVNULL)

    # Check against PyTorch fp32 on this Mac. The ANE runs fp16, so expect small differences.
    for name, inputs, reference in checks:
        for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU):
            if name is None:
                model = ct.models.MLModel(args.out, compute_units=units)
            else:
                model = ct.models.CompiledMLModel(str(compiled), compute_units=units, function_name=name)
            model.predict(inputs)  # warm-up
            t0 = time.time()
            out = model.predict(inputs)["noise_pred"]
            dt = time.time() - t0
            cos = float(np.dot(out.ravel(), reference.ravel()) / (np.linalg.norm(out) * np.linalg.norm(reference)))
            print(f"{name or 'model'} {units.name}: {dt * 1000:.1f} ms, "
                  f"max abs err {np.abs(out - reference).max():.4f}, cosine {cos:.6f}")

    if args.compile:
        print(f"compiled to {compiled}")
    elif multi:
        shutil.rmtree(compiled)


if __name__ == "__main__":
    main()
