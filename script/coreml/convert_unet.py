"""Export a diffusers SD 1.x/2.x UNet to CoreML for sd-cli --coreml-unet.

The model takes float32 inputs and returns float32, with the names the runtime expects:
  sample [N, 4, H/8, W/8], timestep [N], encoder_hidden_states [N, T, D] -> noise_pred [N, 4, H/8, W/8]

Example (SDXS-512-DreamShaper, fp16):
  python convert_unet.py --unet sdxs/unet --out sdxs_unet_fp16.mlpackage --precision fp16

pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4
"""

import argparse
import subprocess
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unet", required=True, help="diffusers UNet folder (the pipeline's unet/ directory)")
    parser.add_argument("--out", required=True, help="output .mlpackage path")
    parser.add_argument("--precision", choices=["fp16", "w8"], default="fp16",
                        help="fp16 weights and compute, or int8 weights (linear, per channel) with fp16 compute")
    parser.add_argument("--batch", type=int, default=1, help="UNet batch: 1, or 2 for batched CFG")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--context-len", type=int, default=77, help="text tokens; sd.cpp uses 77 per prompt chunk")
    parser.add_argument("--target", choices=["iOS17", "iOS18"], default="iOS18",
                        help="iOS18 maps attention to the native scaled_dot_product_attention op")
    parser.add_argument("--compile", action="store_true", help="also write a compiled .mlmodelc next to the package")
    parser.add_argument("--keep-silu", action="store_true", help="keep the native silu op (inaccurate on the ANE)")
    args = parser.parse_args()

    if not args.keep_silu:
        torch.nn.SiLU.forward = silu_via_tanh

    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float32).eval()
    cfg = unet.config
    shapes = {
        "sample": (args.batch, cfg.in_channels, args.height // 8, args.width // 8),
        "timestep": (args.batch,),
        "encoder_hidden_states": (args.batch, args.context_len, cfg.cross_attention_dim),
    }
    example = (
        torch.randn(shapes["sample"]),
        torch.full(shapes["timestep"], 999.0),
        torch.randn(shapes["encoder_hidden_states"]),
    )

    wrapper = UNetWrapper(unet).eval()
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
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, ct.optimize.coreml.OptimizationConfig(op_config))
    print(f"converted in {time.time() - t0:.1f} s")

    mlmodel.short_description = f"UNet {Path(args.unet).resolve()} ({args.precision}) for sd-cli --coreml-unet"
    mlmodel.save(args.out)

    # Check against PyTorch fp32 on this Mac. The ANE runs fp16, so expect small differences.
    inputs = {name: tensor.numpy() for name, tensor in zip(shapes, example)}
    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU):
        model = ct.models.MLModel(args.out, compute_units=units)
        model.predict(inputs)  # warm-up
        t0 = time.time()
        out = model.predict(inputs)["noise_pred"]
        dt = time.time() - t0
        cos = float(np.dot(out.ravel(), reference.ravel()) / (np.linalg.norm(out) * np.linalg.norm(reference)))
        print(f"{units.name}: {dt * 1000:.1f} ms, max abs err {np.abs(out - reference).max():.4f}, cosine {cos:.6f}")

    if args.compile:
        out_dir = Path(args.out).resolve().parent
        subprocess.run(["xcrun", "coremlcompiler", "compile", args.out, str(out_dir)], check=True)
        print(f"compiled to {out_dir / (Path(args.out).stem + '.mlmodelc')}")


if __name__ == "__main__":
    main()
