"""Export a tiny autoencoder's decoder (diffusers AutoencoderTiny) to CoreML for sd-cli --coreml-tae.

Works for the decoder built into SDXS (its vae/ folder), for TAESD (madebyollin/taesd) and for
TAEF2 (madebyollin/taef2, FLUX.2). --vae is a diffusers AutoencoderTiny folder or a .safetensors file
with the original taesd.py key names (decoder.layers.N...), which is also what sd.cpp's --taesd reads.
The model takes and returns float32 with the names the runtime expects:
  latent [1, 4, H/8, W/8] -> image [1, 3, H, W], in [0, 1] before sd.cpp clamps it
  TAEF2: latent [1, 128, H/16, W/16] -> image [1, 3, H, W]
TAEF2 decodes 32-channel latents; FLUX.2 hands the decoder its 128-channel latent with 2x2 patches
folded into channels, so the graph starts with the same unpatchify (a pixel shuffle) as sd.cpp's TAESD.

This is the same graph as sd.cpp's TinyDecoder: the tanh clamp, then the decoder layers. It leaves
out the x * 2 - 1 that diffusers' DecoderTiny.forward ends with; exporting that by mistake gives
[-1, 1] images that sd.cpp's clamp turns dark. The decoder is only convolutions, ReLU, nearest
upsampling and one tanh, so it runs entirely on the Neural Engine and needs no op rewrites.

Examples:
  python convert_tae_decoder.py --vae sdxs-512-dreamshaper/vae --out sdxs_tae_decoder.mlpackage --compile
  python convert_tae_decoder.py --vae taef2.safetensors --size 512x512 --out taef2_decoder.mlpackage --compile

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


class TAEBlock(torch.nn.Module):
    """taesd.py's Block, with TAEF2's optional GroupNorm pool branch in the first three decoder blocks."""

    def __init__(self, channels, gn):
        super().__init__()
        c = torch.nn.Conv2d
        self.conv = torch.nn.Sequential(c(channels, channels, 3, padding=1), torch.nn.ReLU(),
                                        c(channels, channels, 3, padding=1), torch.nn.ReLU(),
                                        c(channels, channels, 3, padding=1))
        self.pool = None
        if gn:
            self.pool = torch.nn.Sequential(c(channels, 4 * channels, 1, bias=False),
                                            torch.nn.GroupNorm(4, 4 * channels), torch.nn.ReLU(),
                                            c(4 * channels, channels, 1, bias=False))

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return torch.relu(self.conv(x) + x)


class RawTinyDecoder(torch.nn.Module):
    """sd.cpp's TinyDecoder (src/model/vae/tae.hpp), loaded from original taesd.py key names."""

    def __init__(self, state):
        super().__init__()
        z_channels = state["decoder.layers.0.weight"].shape[1]
        self.gn = "decoder.layers.2.pool.0.weight" in state
        self.patch = 2 if self.gn else 1  # TAEF2
        c = torch.nn.Conv2d
        layers = [c(z_channels, 64, 3, padding=1), torch.nn.ReLU()]
        for stage in range(3):
            layers += [TAEBlock(64, self.gn and stage == 0) for _ in range(3)]
            layers += [torch.nn.Upsample(scale_factor=2), c(64, 64, 3, padding=1, bias=False)]
        layers += [TAEBlock(64, False), c(64, 3, 3, padding=1)]
        self.layers = torch.nn.Sequential(*layers)
        own = {"decoder.layers." + k: v for k, v in self.layers.state_dict().items()}
        missing = sorted(set(own) - set(state))
        if missing:
            raise SystemExit(f"error: weights miss {missing[:4]}")
        self.layers.load_state_dict({k[len("decoder.layers."):]: v for k, v in state.items() if k in own})
        self.latent_channels = z_channels * self.patch * self.patch

    def forward(self, latent):
        if self.patch > 1:
            latent = torch.nn.functional.pixel_shuffle(latent, self.patch)
        return self.layers(torch.tanh(latent / 3) * 3)


def load_raw_decoder(path):
    from safetensors.torch import load_file

    state = {k: v.float() for k, v in load_file(path).items() if k.startswith("decoder.layers.")}
    if not state:
        raise SystemExit(f"error: {path} has no decoder.layers.* tensors")
    return RawTinyDecoder(state).eval()


class DecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.layers = vae.decoder.layers

    def forward(self, latent):
        # DecoderTiny.forward without its final x * 2 - 1, so the image stays in [0, 1].
        return self.layers(torch.tanh(latent / 3) * 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", required=True, help="diffusers AutoencoderTiny folder (SDXS's vae/, madebyollin/taesd) or taesd/taef2 .safetensors")
    parser.add_argument("--out", required=True, help="output .mlpackage path")
    parser.add_argument("--size", default="512x512", help="image size as WIDTHxHEIGHT (multiples of 8)")
    parser.add_argument("--target", choices=["iOS17", "iOS18"], default="iOS17", help="op set")
    parser.add_argument("--compile", action="store_true", help="also write a compiled .mlmodelc next to the package")
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    if args.vae.endswith(".safetensors"):
        wrapper = load_raw_decoder(args.vae)
        channels, factor = wrapper.latent_channels, 8 * wrapper.patch
    else:
        vae = AutoencoderTiny.from_pretrained(args.vae, torch_dtype=torch.float32).eval()
        if vae.config.latent_channels != 4:
            raise SystemExit(f"error: expected 4 latent channels, got {vae.config.latent_channels}")
        wrapper = DecoderWrapper(vae).eval()
        channels, factor = 4, 8
    if width % factor or height % factor:
        raise SystemExit(f"error: {args.size}: width and height must be multiples of {factor}")

    torch.manual_seed(0)
    latent = torch.randn(1, channels, height // factor, width // factor)
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
