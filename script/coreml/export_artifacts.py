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
  sdxl        SDXL 1.0 base, 768x768, 20 steps, CFG > 1 (batched); not in the default --models
    sd_xl_base_1.0.safetensors     text encoders + VAE for sd.cpp (its UNet is not loaded); -m
    sdxl_unet_mixed/               UNet on CoreML, 4/6-bit mixed palette (4.5 bits on average), batch 2,
                                   split-einsum attention, two chunks        --coreml-unet
    taesdxl.safetensors            TAESDXL weights                             --taesd
    taesdxl_decoder_768.mlmodelc   TAESDXL decoder on CoreML, 768x768          --coreml-tae
    The UNet export traces 10 GB of fp32 weights and peaks at about 30 GB of RAM and 15 GB of temporary disk:
    run it on a Mac with 32 GB or more. Its per-layer sensitivity pass takes about 20 minutes on an M3 Ultra.

The GGUF steps need an sd-cli whose `-M convert` reads a whole diffusers folder. On the 2026-11-01 branch it
reads only text_encoder/ (610 tensors for SDXS, no UNet or VAE); build sd-cli from master for --sd-cli.

Run on a Mac with the CoreML toolchain:
  pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.3 coremltools==8.3.0 numpy==1.26.4 scikit-learn==1.5.2 huggingface_hub
  python script/coreml/export_artifacts.py --out artifacts --sd-cli build/bin/sd-cli
  python script/coreml/export_artifacts.py --out artifacts --models sd21base --dry-run

Measured with these artifacts (512x512, warm, mean of 3 images; see docs/coreml.md):
  SDXS on iPhone 16, UNet + decoder on the ANE: 105 ms per image (ggml Metal: 1,758 ms).
  SD 2.1-base on an M2, UNet + decoder on the ANE: 7.5 s per image at 20 steps (ggml Metal, SD 2.1: 60 s).
  SDXL on an M2 Air at 768x768, UNet + decoder on the ANE: 26 s per image at 20 steps (ggml Metal, Q8_0: 162-232 s).
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
    # Licenses: SDXL is OpenRAIL++, TAESDXL is MIT.
    "sdxl": {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "revision": "462165984030d82259a11f4367a4eed129e94a7b",
             "include": ["sd_xl_base_1.0.safetensors", "scheduler/scheduler_config.json", "unet/config.json",
                         "unet/diffusion_pytorch_model.fp16.safetensors"]},
    "taesdxl": {"repo": "madebyollin/taesdxl", "revision": "b20258aaef75ef61e659c1e0f14f251cf0ad153e",
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


def export_sdxl(out, sources, dry_run, kmeans_workers):
    src = download("sdxl", sources, dry_run)
    if not exists(out / "sd_xl_base_1.0.safetensors"):
        run(["ln", "-sf", src / "sd_xl_base_1.0.safetensors", out / "sd_xl_base_1.0.safetensors"], dry_run)
    if not exists(out / "sdxl_unet_mixed"):
        convert("convert_unet.py", ["--unet", src / "unet", "--out", out / "sdxl_unet_mixed.mlpackage", "--precision",
                                    "mixed", "--mixed-bits", "4.5", "--batch", "2", "--attention", "split_einsum_v2",
                                    "--sizes", "768x768", "--chunks", "2", "--kmeans-workers", str(kmeans_workers),
                                    "--sensitivity", sources / "sdxl_unet_sensitivity.json", "--compile"], dry_run)
        for chunk in ("sdxl_unet_mixed_chunk1.mlpackage", "sdxl_unet_mixed_chunk2.mlpackage"):
            keep_compiled(out / "sdxl_unet_mixed" / chunk, dry_run)
    taesdxl = download("taesdxl", sources, dry_run)
    if not exists(out / "taesdxl.safetensors"):
        run(["cp", taesdxl / "diffusion_pytorch_model.safetensors", out / "taesdxl.safetensors"], dry_run)
    if not exists(out / "taesdxl_decoder_768.mlmodelc"):
        convert("convert_tae_decoder.py", ["--vae", taesdxl, "--out", out / "taesdxl_decoder_768.mlpackage",
                                           "--size", "768x768", "--compile"], dry_run)
        keep_compiled(out / "taesdxl_decoder_768.mlpackage", dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for the artifacts")
    parser.add_argument("--sources", help="directory for downloaded sources (default: <out>/sources)")
    parser.add_argument("--models", default="sdxs,sd21base", help="comma-separated: sdxs, sd21base, sdxl")
    parser.add_argument("--sd-cli", default="build/bin/sd-cli", help="sd-cli, used to convert the models to GGUF")
    parser.add_argument("--skip-existing", action="store_true", help="skip artifacts that already exist in --out")
    parser.add_argument("--kmeans-workers", type=int, default=2, help="k-means workers for the palettized exports")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    global SKIP_EXISTING
    SKIP_EXISTING = args.skip_existing
    out = Path(args.out).resolve()
    sources = Path(args.sources).resolve() if args.sources else out / "sources"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - {"sdxs", "sd21base", "sdxl"}
    if unknown:
        raise SystemExit(f"error: unknown models {sorted(unknown)}")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        if set(models) - {"sdxl"} and not Path(args.sd_cli).exists():
            raise SystemExit(f"error: {args.sd_cli} not found; build sd-cli first or pass --sd-cli")

    if "sdxs" in models:
        export_sdxs(out, sources, Path(args.sd_cli).resolve(), args.dry_run)
    if "sd21base" in models:
        export_sd21base(out, sources, Path(args.sd_cli).resolve(), args.dry_run, args.kmeans_workers)
    if "sdxl" in models:
        export_sdxl(out, sources, args.dry_run, args.kmeans_workers)
    print(f"done: artifacts in {out}")


if __name__ == "__main__":
    main()
