"""Regenerate the mobile (iOS) diffusion artifacts from pinned upstream sources.

Every file below is rebuilt from the original weights at a fixed Hugging Face revision, with the converters in this
directory and sd-cli. Steps run one at a time (the 6-bit SD 2.1 export alone needs about 8 GB of RAM).

  sdxs        SDXS-512-DreamShaper, 512x512, 1 step, no CFG
    sdxs-f16.gguf                  text encoder + UNet + tiny decoder for sd.cpp (sd-cli -M convert)
    sdxs_unet_fp16.mlmodelc        UNet on CoreML, fp16                       --coreml-unet
    sdxs_tae_decoder.mlmodelc      built-in tiny decoder on CoreML             --coreml-tae
  sd21base    SD 2.1-base, 512x512, 20 steps, CFG > 1 (batched), --prediction eps with a CoreML UNet
    sd21base-Q8_0.gguf             text encoder + UNet + VAE for sd.cpp (sd-cli -M convert); -m
    sd21base_unet_p6.mlmodelc      UNet on CoreML, 6-bit palette, batch 2, split-einsum attention  --coreml-unet
    taesd.safetensors              TAESD weights (sd.cpp still loads them)     --taesd
    taesd_decoder.mlmodelc         TAESD decoder on CoreML                     --coreml-tae

Run on a Mac with the CoreML toolchain:
  pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.3 coremltools==8.3.0 numpy==1.26.4 scikit-learn==1.5.2 huggingface_hub
  python script/coreml/export_artifacts.py --out artifacts --sd-cli build/bin/sd-cli
  python script/coreml/export_artifacts.py --out artifacts --models sd21base --dry-run

Measured with these artifacts (512x512, warm, mean of 3 images; see docs/coreml.md):
  SDXS on iPhone 16, UNet + decoder on the ANE: 105 ms per image (ggml Metal: 1,758 ms).
  SD 2.1-base on an M2, UNet + decoder on the ANE: 7.5 s per image at 20 steps (ggml Metal, SD 2.1: 60 s).
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Pinned sources. Licenses: SDXS and SD 2.1-base are OpenRAIL++, TAESD is MIT.
SOURCES = {
    "sdxs": {"repo": "IDKiro/sdxs-512-dreamshaper", "revision": "76f720262bb051da75666b22c902a78c8e16c763",
             "include": ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "unet/*", "vae/*",
                         "feature_extractor/*"]},
    "sd21base": {"repo": "sd2-community/stable-diffusion-2-1-base", "revision": "4e63672c03103b6c636b8fb4119ba982469b2955",
                 "include": ["model_index.json", "scheduler/*", "tokenizer/*", "unet/config.json",
                             "unet/diffusion_pytorch_model.fp16.safetensors", "text_encoder/config.json",
                             "text_encoder/model.fp16.safetensors", "vae/config.json",
                             "vae/diffusion_pytorch_model.fp16.safetensors"]},
    "taesd": {"repo": "madebyollin/taesd", "revision": "614f76814bbe30edbe2e627ace1c2234c81a2c0e",
              "include": ["config.json", "diffusion_pytorch_model.safetensors"]},
}


SKIP_EXISTING = False


def exists(path):
    if SKIP_EXISTING and Path(path).exists():
        print(f"  exists, skipped: {path}")
        return True
    return False


def run(cmd, dry_run):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def download(name, sources_dir, dry_run):
    src = SOURCES[name]
    dest = sources_dir / name
    run(["hf", "download", src["repo"], "--revision", src["revision"], "--local-dir", dest, "--include", *src["include"]],
        dry_run)
    for file, expected in src.get("sha256", {}).items():
        if dry_run:
            continue
        digest = hashlib.sha256()
        with open(dest / file, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise SystemExit(f"error: {dest / file} has sha256 {digest.hexdigest()}, expected {expected}")
        print(f"  sha256 ok: {file}")
    return dest


def convert(script, args, dry_run):
    run([sys.executable, HERE / script, *args], dry_run)


def keep_compiled(package, dry_run):
    """The converters write both .mlpackage and .mlmodelc; only the compiled model ships."""
    print(f"+ rm -rf {package}")
    if not dry_run:
        shutil.rmtree(package, ignore_errors=True)


def fp16_view(src, dest, dry_run):
    """sd-cli reads unet/diffusion_pytorch_model.safetensors etc.; link the pipeline's fp16 variants under those names."""
    links = {"unet/diffusion_pytorch_model.safetensors": "unet/diffusion_pytorch_model.fp16.safetensors",
             "vae/diffusion_pytorch_model.safetensors": "vae/diffusion_pytorch_model.fp16.safetensors",
             "text_encoder/model.safetensors": "text_encoder/model.fp16.safetensors"}
    for name, target in links.items():
        print(f"+ ln -sf {src / target} {dest / name}")
        if not dry_run:
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).unlink(missing_ok=True)
            (dest / name).symlink_to(src / target)
    for name in ("model_index.json", "scheduler", "tokenizer", "unet/config.json", "vae/config.json", "text_encoder/config.json"):
        if not dry_run and not (dest / name).exists():
            (dest / name).symlink_to(src / name)
    return dest


def export_sdxs(out, sources, sd_cli, dry_run):
    src = download("sdxs", sources, dry_run)
    if not exists(out / "sdxs-f16.gguf"):
        run([sd_cli, "-M", "convert", "-m", src, "-o", out / "sdxs-f16.gguf", "--type", "f16"], dry_run)
    if not exists(out / "sdxs_unet_fp16.mlmodelc"):
        convert("convert_unet.py", ["--unet", src / "unet", "--out", out / "sdxs_unet_fp16.mlpackage", "--compile"], dry_run)
        keep_compiled(out / "sdxs_unet_fp16.mlpackage", dry_run)
    if not exists(out / "sdxs_tae_decoder.mlmodelc"):
        convert("convert_tae_decoder.py", ["--vae", src / "vae", "--out", out / "sdxs_tae_decoder.mlpackage", "--compile"],
                dry_run)
        keep_compiled(out / "sdxs_tae_decoder.mlpackage", dry_run)


def export_sd21base(out, sources, sd_cli, dry_run, kmeans_workers):
    src = download("sd21base", sources, dry_run)
    if not exists(out / "sd21base-Q8_0.gguf"):
        # Q8_0 like the addon's SD 2.1 file; sd-cli keeps convolutions, norms and the text encoder in F16.
        view = fp16_view(src, sources / "sd21base-diffusers", dry_run)
        run([sd_cli, "-M", "convert", "-m", view, "-o", out / "sd21base-Q8_0.gguf", "--type", "q8_0"], dry_run)
    taesd = download("taesd", sources, dry_run)
    if not exists(out / "taesd.safetensors"):
        run(["cp", taesd / "diffusion_pytorch_model.safetensors", out / "taesd.safetensors"], dry_run)
    if not exists(out / "taesd_decoder.mlmodelc"):
        convert("convert_tae_decoder.py", ["--vae", taesd, "--out", out / "taesd_decoder.mlpackage", "--compile"], dry_run)
        keep_compiled(out / "taesd_decoder.mlpackage", dry_run)
    if not exists(out / "sd21base_unet_p6.mlmodelc"):
        convert("convert_unet.py", ["--unet", src / "unet", "--out", out / "sd21base_unet_p6.mlpackage", "--precision",
                                    "p6", "--batch", "2", "--attention", "split_einsum_v2", "--kmeans-workers",
                                    str(kmeans_workers), "--compile"], dry_run)
        keep_compiled(out / "sd21base_unet_p6.mlpackage", dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for the artifacts")
    parser.add_argument("--sources", help="directory for downloaded sources (default: <out>/sources)")
    parser.add_argument("--models", default="sdxs,sd21base", help="comma-separated: sdxs, sd21base")
    parser.add_argument("--sd-cli", default="build/bin/sd-cli", help="sd-cli, used to convert the models to GGUF")
    parser.add_argument("--skip-existing", action="store_true", help="skip artifacts that already exist in --out")
    parser.add_argument("--kmeans-workers", type=int, default=2, help="k-means workers for the 6-bit SD 2.1 export")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    global SKIP_EXISTING
    SKIP_EXISTING = args.skip_existing
    out = Path(args.out).resolve()
    sources = Path(args.sources).resolve() if args.sources else out / "sources"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - {"sdxs", "sd21base"}
    if unknown:
        raise SystemExit(f"error: unknown models {sorted(unknown)}")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        if not Path(args.sd_cli).exists():
            raise SystemExit(f"error: {args.sd_cli} not found; build sd-cli first or pass --sd-cli")

    if "sdxs" in models:
        export_sdxs(out, sources, Path(args.sd_cli).resolve(), args.dry_run)
    if "sd21base" in models:
        export_sd21base(out, sources, Path(args.sd_cli).resolve(), args.dry_run, args.kmeans_workers)
    print(f"done: artifacts in {out}")


if __name__ == "__main__":
    main()
