"""Export a tiny autoencoder's decoder (diffusers AutoencoderTiny) to CoreML for sd-cli --coreml-tae.

Works for the decoder built into SDXS (its vae/ folder) and for TAESD (madebyollin/taesd). The model
takes and returns float32 with the names the runtime expects:
  latent [1, 4, H/8, W/8] -> image [1, 3, H, W], in [0, 1] before sd.cpp clamps it

This is the same graph as sd.cpp's TinyDecoder: the tanh clamp, then the decoder layers. It leaves
out the x * 2 - 1 that diffusers' DecoderTiny.forward ends with; exporting that by mistake gives
[-1, 1] images that sd.cpp's clamp turns dark. The decoder is only convolutions, ReLU, nearest
upsampling and one tanh, so it runs entirely on the Neural Engine and needs no op rewrites.

Example (SDXS-512-DreamShaper, fp16):
  python convert_tae_decoder.py --vae sdxs-512-dreamshaper/vae --out sdxs_tae_decoder.mlpackage --compile

pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4
"""

import argparse
import shutil
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from diffusers import AutoencoderTiny


class DecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.layers = vae.decoder.layers

    def forward(self, latent):
        # DecoderTiny.forward without its final x * 2 - 1, so the image stays in [0, 1].
        return self.layers(torch.tanh(latent / 3) * 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", required=True, help="diffusers AutoencoderTiny folder (SDXS's vae/ or madebyollin/taesd)")
    parser.add_argument("--out", required=True, help="output .mlpackage path")
    parser.add_argument("--size", default="512x512", help="image size as WIDTHxHEIGHT (multiples of 8)")
    parser.add_argument("--target", choices=["iOS17", "iOS18"], default="iOS17", help="op set")
    parser.add_argument("--compile", action="store_true", help="also write a compiled .mlmodelc next to the package")
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    if width % 8 or height % 8:
        raise SystemExit(f"error: {args.size}: width and height must be multiples of 8")

    vae = AutoencoderTiny.from_pretrained(args.vae, torch_dtype=torch.float32).eval()
    if vae.config.latent_channels != 4:
        raise SystemExit(f"error: expected 4 latent channels, got {vae.config.latent_channels}")
    wrapper = DecoderWrapper(vae).eval()

    torch.manual_seed(0)
    latent = torch.randn(1, 4, height // 8, width // 8)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (latent,), strict=False)
        reference = wrapper(latent).numpy()

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="latent", shape=latent.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="image", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=getattr(ct.target, args.target),
    )
    print(f"{width}x{height}: converted in {time.time() - t0:.1f} s")
    mlmodel.short_description = f"Tiny autoencoder decoder {Path(args.vae).resolve()} for sd-cli --coreml-tae"
    mlmodel.save(args.out)

    # Check against PyTorch fp32 on this Mac, as PSNR of the [0, 1] image after clamping.
    inputs = {"latent": latent.numpy()}
    ref = np.clip(reference, 0, 1)
    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU):
        model = ct.models.MLModel(args.out, compute_units=units)
        model.predict(inputs)  # warm-up
        t0 = time.time()
        out = model.predict(inputs)["image"]
        dt = time.time() - t0
        mse = float(((np.clip(out, 0, 1) - ref) ** 2).mean())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        print(f"{units.name}: {dt * 1000:.1f} ms, max abs err {np.abs(out - reference).max():.4f}, PSNR vs fp32 {psnr:.1f} dB")

    if args.compile:
        out_dir = Path(args.out).resolve().parent
        compiled = out_dir / (Path(args.out).stem + ".mlmodelc")
        if compiled.exists():
            shutil.rmtree(compiled)
        ct.utils.compile_model(args.out, str(compiled))  # the CoreML framework's compiler; no Xcode needed
        print(f"compiled to {compiled}")


if __name__ == "__main__":
    main()
