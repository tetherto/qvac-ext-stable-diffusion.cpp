#!/usr/bin/env python3
"""Repeat an image-generation workload and summarize the engine's phase timings."""

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


TIMINGS = {
    "conditioning_s": (r"get_learned_condition completed, taking ([\d.]+)s", 1.0),
    "sampling_s": (r"sampling completed, taking ([\d.]+)s", 1.0),
    "vae_decode_s": (r"decode_first_stage completed, taking ([\d.]+)s", 1.0),
    "generation_s": (r"generate_image completed in ([\d.]+)s", 1.0),
}
FLUX_CALL = r"flux denoiser compute completed, taking (\d+) ms"


def parse_timings(log):
    # Older engine logs used milliseconds and a global backend banner.
    log = re.sub(r"get_learned_condition completed, taking (\d+) ms",
                 lambda match: f"get_learned_condition completed, taking {int(match[1]) / 1000}s", log)
    if "Using Metal backend" not in log and not re.search(r"Diffusion model runtime backend: Metal", log):
        raise ValueError("Metal backend was not reported; check the build and run log")
    if ("VAE Autoencoder: Using CPU backend" in log or
            "VAE runtime backend: CPU" in log or "VAE CPU fallback complete" in log):
        raise ValueError("the VAE ran on CPU; remove --vae-on-cpu")

    result = {}
    for name, (pattern, scale) in TIMINGS.items():
        matches = re.findall(pattern, log)
        if len(matches) != 1:
            raise ValueError(f"expected one {name} measurement, found {len(matches)}")
        result[name] = float(matches[0]) * scale
    if result["generation_s"] <= 0:
        raise ValueError("generation time must be positive")
    result["vae_share_pct"] = 100 * result["vae_decode_s"] / result["generation_s"]
    flux_calls_ms = [int(value) for value in re.findall(FLUX_CALL, log)]
    if flux_calls_ms:
        result["flux_denoiser_calls"] = len(flux_calls_ms)
        result["flux_denoiser_s"] = sum(flux_calls_ms) / 1000
        if result["flux_denoiser_s"] > result["sampling_s"] + 0.05:
            raise ValueError("FLUX denoiser time exceeds total sampling time")
        result["sampling_other_s"] = max(0.0, result["sampling_s"] - result["flux_denoiser_s"])
        result["flux_denoiser_share_pct"] = 100 * result["flux_denoiser_s"] / result["sampling_s"]
    return result


def git_revision(repo):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark ggml Metal image generation before a Core ML sidecar is added."
    )
    parser.add_argument("--binary", type=Path, default=Path("build/bin/sd-cli"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--require-flux-profile", action="store_true",
                        help="fail unless the instrumented FLUX denoiser reported per-call timings")
    args, cli_args = parser.parse_known_args(argv)
    if cli_args and cli_args[0] == "--":
        cli_args = cli_args[1:]
    if not cli_args:
        parser.error("supply sd-cli model and generation arguments after --")
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >= 0 and --runs must be >= 1")
    forbidden = {"-o", "--output", "-b", "--batch-count", "-M", "--mode"}
    if forbidden.intersection(cli_args):
        parser.error("the runner sets output, batch count, and image generation mode")
    if not ({"-m", "--model", "--diffusion-model"} & set(cli_args)):
        parser.error("pass --model or --diffusion-model to sd-cli after --")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"sd-cli binary not found: {binary}")
    output_dir = args.output_dir.resolve()
    if (output_dir / "report.json").exists():
        parser.error(f"report.json already exists in {output_dir}; use a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": git_revision(Path(__file__).resolve().parent.parent),
        "machine": platform.platform(),
        "processor": platform.processor(),
        "binary": str(binary),
        "sd_cli_args": cli_args,
        "warmup": args.warmup,
        "runs": args.runs,
        "measurements": [],
    }
    for index in range(args.warmup + args.runs):
        warmup = index < args.warmup
        label = f"{'warmup' if warmup else 'run'}-{(index + 1) if warmup else (index - args.warmup + 1):02d}"
        image = output_dir / f"{label}.png"
        log_path = output_dir / f"{label}.log"
        command = [str(binary), "-M", "img_gen", *cli_args, "-b", "1", "-v", "-o", str(image)]
        print(f"{label}: running", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        wall_s = time.perf_counter() - started
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(f"{label} exited with status {completed.returncode}; see {log_path}")
        if not image.is_file() or image.stat().st_size == 0:
            raise RuntimeError(f"{label} produced no image; see {log_path}")
        try:
            timings = parse_timings(log)
        except ValueError as exc:
            raise RuntimeError(f"{label}: {exc}; see {log_path}") from exc
        if args.require_flux_profile and "flux_denoiser_s" not in timings:
            raise RuntimeError(f"{label}: no FLUX denoiser timings; rebuild sd-cli; see {log_path}")
        measurement = {
            "label": label,
            "warmup": warmup,
            "command": command,
            "log": str(log_path),
            "image": str(image),
            "wall_s": wall_s,
            **timings,
        }
        report["measurements"].append(measurement)
        print(
            f"{label}: generation {timings['generation_s']:.2f}s, "
            f"sampling {timings['sampling_s']:.2f}s, "
            f"VAE {timings['vae_decode_s']:.2f}s ({timings['vae_share_pct']:.1f}%)"
            + (
                f", FLUX denoiser {timings['flux_denoiser_s']:.2f}s "
                f"in {timings['flux_denoiser_calls']} calls"
                if "flux_denoiser_s" in timings else ""
            ),
            flush=True,
        )

    measured = [item for item in report["measurements"] if not item["warmup"]]
    fields = ("conditioning_s", "sampling_s", "vae_decode_s", "generation_s", "wall_s", "vae_share_pct")
    report["median"] = {field: statistics.median(item[field] for item in measured) for field in fields}
    flux_fields = ("flux_denoiser_calls", "flux_denoiser_s", "sampling_other_s", "flux_denoiser_share_pct")
    if any("flux_denoiser_s" in item for item in measured):
        if not all("flux_denoiser_s" in item for item in measured):
            raise RuntimeError("FLUX denoiser timing was missing from some runs")
        report["median"].update({
            field: statistics.median(item[field] for item in measured) for field in flux_fields
        })
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Median timings saved to {report_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)
