import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

CASES = [("square", 512, 512, 42, []), ("flash", 512, 512, 42, ["--diffusion-fa"]),
         ("landscape", 512, 384, 123, [])]
PROMPT = "A red ceramic teapot on a wooden table beside a window, soft morning light, detailed photograph"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare matching preflight engine/package pairs")
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--fabric", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=["cpu", "cuda0", "mtl0"], required=True)
    return parser.parse_args()


def run_image(args, variant, case, repeat):
    name, width, height, seed, flags = case
    root = getattr(args, variant).resolve()
    prefix = root / "install"
    image = args.output / (variant + "-" + name + "-" + str(repeat) + ".png")
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env = {**os.environ, variable: str(prefix / "lib") + os.pathsep + os.environ.get(variable, "")}
    env.pop("GGML_BACKEND_PATH", None)
    command = [str(root / "engine-build/bin/sd-cli"), "--model", str(args.model.resolve()),
               "--output", str(image), "--backend", args.backend, "--prompt", PROMPT,
               "--negative-prompt", "", "--width", str(width), "--height", str(height),
               "--seed", str(seed), "--steps", "12", "--cfg-scale", "7", "--sampling-method", "euler",
               "--rng", "cpu", "--sampler-rng", "cpu", "--threads", "4", "--verbose", *flags]
    start = time.monotonic()
    with image.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=prefix / "lib", env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with Image.open(image) as opened:
        if opened.size != (width, height):
            raise RuntimeError("Unexpected output dimensions")
        pixels = opened.convert("RGB").tobytes()
    return {"variant": variant, "case": name, "repeat": repeat,
            "seconds": time.monotonic() - start, "pixel_sha256": hashlib.sha256(pixels).hexdigest()}, pixels


def compare_case(args, case):
    records, pixels = [], {}
    for repeat in range(2):
        for variant in (["control", "fabric"] if repeat == 0 else ["fabric", "control"]):
            record, data = run_image(args, variant, case, repeat)
            records.append(record)
            pixels[(variant, repeat)] = data
    differences = [abs(a - b) for a, b in zip(pixels[("control", 0)], pixels[("fabric", 0)])]
    return {"case": case[0], "runs": records, "identical": max(differences) == 0,
            "repeatable": pixels[("control", 0)] == pixels[("control", 1)] and
                          pixels[("fabric", 0)] == pixels[("fabric", 1)],
            "mean_abs_pixel_error": sum(differences) / len(differences),
            "max_abs_pixel_error": max(differences)}


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in CASES:
        results.append(compare_case(args, case))
        (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(x["identical"] and x["repeatable"] for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
