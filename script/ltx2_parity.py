#!/usr/bin/env python3
"""
ltx2_parity.py — M2 parity / reproducibility harness for LTX-2 video generation.

Validates the LTX-2 T2V and I2V paths produce consistent output across backends:

  * same-backend reproducibility — STRICT. Re-running the same seed on the same
    backend must yield (near-)identical frames. Catches nondeterminism bugs.
  * cross-backend similarity     — LOOSE. CPU vs Vulkan vs Metal are compared
    within a perceptual band only. Exact pixel parity is NOT expected: LTX-2 is
    a multi-step diffusion model, and FP16 math + differing kernel/reduction
    order across ggml backends compound over the denoising loop. We assert the
    *content* matches (PSNR above a floor), not bit-equality.

It drives the `sd` CLI (-M vid_gen), extracts frames with ffmpeg, and compares
them against a committed reference set.

Workflow
--------
1. Produce the golden reference once, on the CPU backend (the portable baseline):

     ltx2_parity.py --bin build/bin/sd-cli --models-dir "$LTX2_MODELS" \\
         --mode both --update-ref --ref-dir test/refs/ltx2

2. On each accelerated build, check against that reference:

     ltx2_parity.py --bin build-vulkan/bin/sd-cli --models-dir "$LTX2_MODELS" \\
         --mode both --ref-dir test/refs/ltx2 --backend vulkan \\
         --min-psnr 18

   And a strict self-check (same backend, two runs must match):

     ltx2_parity.py --bin build-vulkan/bin/sd-cli --models-dir "$LTX2_MODELS" \\
         --mode t2v --self-check --min-psnr 45

Exit code is non-zero if any comparison falls below threshold.

Deps: numpy, Pillow (pip install numpy pillow) and ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Fixed generation parameters — identical across every backend so the only
# variable under test is the compute backend itself. Keep small/fast.
SEED = 42
WIDTH = 512
HEIGHT = 288
FRAMES = 25
FPS = 8
PROMPT = "a lovely cat sitting on a sunny windowsill, high quality"

# sd-cli appends ".avi" to whatever -o path you give it (see test_m2.sh).
AVI_SUFFIX = ".avi"


def model_paths(models_dir: Path) -> dict[str, Path]:
    return {
        "diffusion": models_dir / "distilled/ltx-2.3-22b-distilled-Q4_0.gguf",
        "vae": models_dir / "vae/ltx-2.3-22b-distilled_video_vae.safetensors",
        "llm": models_dir / "gemma-3-12b-it-Q4_K_S.gguf",
        "conn": models_dir / "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
    }


def run_generation(bin_path: Path, models: dict[str, Path], mode: str,
                   out_base: Path, init_image: Path | None) -> Path:
    """Run sd-cli for one t2v/i2v generation. Returns the produced .avi path."""
    cmd = [
        str(bin_path), "-M", "vid_gen",
        "--diffusion-model", str(models["diffusion"]),
        "--vae", str(models["vae"]),
        "--llm", str(models["llm"]),
        "--embeddings-connectors", str(models["conn"]),
        "-p", PROMPT,
        "--cfg-scale", "3.5", "--sampling-method", "euler",
        "-W", str(WIDTH), "-H", str(HEIGHT),
        "--video-frames", str(FRAMES), "--fps", str(FPS),
        "--diffusion-fa", "--offload-to-cpu",
        "-s", str(SEED),
        "-o", str(out_base),
    ]
    if mode == "i2v":
        if init_image is None or not init_image.exists():
            sys.exit(f"i2v mode needs --init-image (got {init_image})")
        cmd += ["-i", str(init_image)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    produced = Path(str(out_base) + AVI_SUFFIX)
    if not produced.exists():
        sys.exit(f"expected output {produced} not produced")
    return produced


def extract_frames(video: Path, dest: Path) -> list[Path]:
    """Decode a video to PNG frames via ffmpeg, deterministically named."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
         "-start_number", "0", str(dest / "f%04d.png")],
        check=True,
    )
    frames = sorted(dest.glob("f*.png"))
    if not frames:
        sys.exit(f"ffmpeg produced no frames from {video}")
    return frames


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB. inf if identical."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def compare_dirs(ref_dir: Path, cand_dir: Path, min_psnr: float) -> bool:
    ref = sorted(ref_dir.glob("f*.png"))
    cand = sorted(cand_dir.glob("f*.png"))
    if not ref:
        sys.exit(f"no reference frames in {ref_dir} — run with --update-ref first")
    if len(ref) != len(cand):
        print(f"  FAIL frame count: ref={len(ref)} cand={len(cand)}")
        return False

    worst = math.inf
    failures = 0
    for r, c in zip(ref, cand):
        ra, ca = load_rgb(r), load_rgb(c)
        if ra.shape != ca.shape:
            print(f"  FAIL {c.name}: shape {ca.shape} != ref {ra.shape}")
            failures += 1
            continue
        d = psnr(ra, ca)
        worst = min(worst, d)
        if d < min_psnr:
            failures += 1
            print(f"  FAIL {c.name}: PSNR {d:.2f} dB < {min_psnr}")
    worst_str = "inf (identical)" if worst == math.inf else f"{worst:.2f} dB"
    print(f"  worst-frame PSNR: {worst_str}   failures: {failures}/{len(ref)}")
    return failures == 0


def run_mode(args, mode: str, init_image: Path | None) -> bool:
    models = model_paths(Path(args.models_dir))
    ref_dir = Path(args.ref_dir) / mode

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        avi = run_generation(Path(args.bin), models, mode, td / mode, init_image)
        frames_dir = td / f"{mode}_frames"
        extract_frames(avi, frames_dir)

        if args.update_ref:
            ref_dir.mkdir(parents=True, exist_ok=True)
            for f in ref_dir.glob("f*.png"):
                f.unlink()
            for f in sorted(frames_dir.glob("f*.png")):
                Image.open(f).save(ref_dir / f.name)
            print(f"  updated reference -> {ref_dir} ({len(list(ref_dir.glob('f*.png')))} frames)")
            return True

        if args.self_check:
            # generate a second time and compare the two live runs (strict)
            avi2 = run_generation(Path(args.bin), models, mode, td / f"{mode}_2", init_image)
            frames2 = td / f"{mode}_frames2"
            extract_frames(avi2, frames2)
            print(f"[{mode}] self-check ({args.backend}, run1 vs run2, strict):")
            return compare_dirs(frames_dir, frames2, args.min_psnr)

        print(f"[{mode}] cross-backend vs reference ({args.backend}):")
        return compare_dirs(ref_dir, frames_dir, args.min_psnr)


def main() -> int:
    ap = argparse.ArgumentParser(description="LTX-2 M2 parity harness")
    ap.add_argument("--bin", required=True, help="path to the sd CLI binary")
    ap.add_argument("--models-dir", default=os.environ.get("LTX2_MODELS", ""),
                    help="dir with LTX-2 model files (or $LTX2_MODELS)")
    ap.add_argument("--ref-dir", default="test/refs/ltx2",
                    help="reference frames root (per-mode subdirs)")
    ap.add_argument("--mode", choices=["t2v", "i2v", "both"], default="both")
    ap.add_argument("--init-image", help="init frame for i2v")
    ap.add_argument("--backend", default="unknown",
                    help="label for logs (cpu/vulkan/metal)")
    ap.add_argument("--update-ref", action="store_true",
                    help="write current output as the golden reference (run on CPU)")
    ap.add_argument("--self-check", action="store_true",
                    help="strict: run twice on this backend and compare (reproducibility)")
    ap.add_argument("--min-psnr", type=float, default=18.0,
                    help="min per-frame PSNR (cross-backend ~18, same-backend ~45)")
    args = ap.parse_args()

    if not args.models_dir:
        return _die("--models-dir or $LTX2_MODELS is required")
    if not Path(args.bin).exists():
        return _die(f"binary not found: {args.bin}")

    modes = ["t2v", "i2v"] if args.mode == "both" else [args.mode]
    init = Path(args.init_image) if args.init_image else None
    ok = True
    for m in modes:
        ok &= run_mode(args, m, init)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
