#!/usr/bin/env python3
"""Benchmark fresh-process MiniMax-H3 generation on physical CUDA GPU 0.

Pass sd-cli model, prompt, seed, size, frame and sampling arguments after ``--``.
The runner owns backend selection, output, mode and batch count. Native AVI
outputs retain both video and audio; decoding and provenance capture run outside
the measured sd-cli process. Media checks establish structural validity only;
prompt fidelity, motion and perceived audio quality need separate review.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


PHASES = {
    "conditioning_s": r"get_learned_condition completed, taking ([\d.]+)\s*(ms|s)",
    "sampling_s": r"sampling completed, taking ([\d.]+)\s*(ms|s)",
    "audio_decode_s": r"decoding audio latent completed, taking ([\d.]+)\s*(ms|s)",
    "video_decode_s": r"decode_first_stage completed, taking ([\d.]+)\s*(ms|s)",
    "generation_s": r"generate_video completed in ([\d.]+)\s*(ms|s)",
}
MODEL_FLAGS = ("--diffusion-model", "--llm", "--vae", "--audio-vae")
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GGML_VK_VISIBLE_DEVICES", "CUDA_MODULE_LOADING",
    "SD_CONVROT_MODE", "SD_CONVROT_H256_MODE", "GGML_CUDA_DISABLE_GRAPHS",
    "GGML_CUDA_FORCE_CUBLAS", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_PDL",
    "OMP_NUM_THREADS", "LD_LIBRARY_PATH",
)
FORBIDDEN_FLAGS = {
    "-o", "--output", "-b", "--batch-count", "-M", "--mode", "--backend",
    "--auto-fit", "--split-mode", "--tensor-split", "--list-devices",
    "--config", "--preview", "--preview-path", "--imat-out",
}
GPU_FIELDS = "timestamp,uuid,pci.bus_id,memory.used,utilization.gpu,clocks.sm,temperature.gpu"


class GpuTelemetry:
    """Sample the whole physical GPU; measurements are not per-process peaks."""

    def __init__(self, folder, executable="nvidia-smi", allow_busy=False):
        self.folder = folder
        self.executable = executable
        self.allow_busy = allow_busy
        self.process = None
        self.output = None
        self.errors = None
        self.metadata = {
            "status": "unavailable", "physical_gpu_index": 0, "interval_ms": 250,
            "csv": str(folder / "gpu.csv"), "stderr": str(folder / "gpu-sampler.log"),
            "before": str(folder / "gpu-before.json"),
            "memory_scope": "whole-device used memory, including other processes; samples can miss the true peak",
            "timing_note": "nvidia-smi sampler runs during sd-cli; startup and shutdown are outside wall_s",
        }

    def query(self, fields, resource="gpu"):
        command = [self.executable, "-i", "0", f"--query-{resource}={fields}", "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or f"nvidia-smi query exited {result.returncode}")
        return command, result.stdout.strip()

    def start(self):
        before = {"physical_gpu_index": 0, "idle_check": "unavailable"}
        try:
            command, identity = self.query("uuid,pci.bus_id,name,driver_version,memory.total")
            rows = list(csv.reader(identity.splitlines(), skipinitialspace=True))
            if len(rows) != 1 or len(rows[0]) != 5 or not rows[0][0].startswith("GPU-"):
                raise ValueError("GPU 0 identity query did not identify exactly one GPU")
            names = ("uuid", "pci_bus_id", "name", "driver_version", "memory_total_mib")
            before.update(identity=dict(zip(names, rows[0])), identity_command=command)
            self.metadata["identity"] = before["identity"]
            try:
                command, processes = self.query("pid,process_name,used_memory", "compute-apps")
                process_rows = list(csv.reader(processes.splitlines(), skipinitialspace=True))
                if any(len(row) != 3 or not row[0].isdigit() for row in process_rows):
                    raise ValueError("GPU 0 process query did not return parseable compute process rows")
                before.update(idle_check="busy" if processes else "idle", compute_processes=processes,
                              process_query_command=command, allow_busy_gpu=self.allow_busy)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                before["idle_check_reason"] = str(exc)
            if before["idle_check"] == "busy" and not self.allow_busy:
                self.metadata["reason"] = "GPU 0 has running compute processes"
                raise RuntimeError("GPU 0 has running compute processes; wait for it to be idle or pass --allow-busy-gpu")
            command = [self.executable, "-i", "0", f"--query-gpu={GPU_FIELDS}",
                       "--format=csv,noheader,nounits", "-lms", "250"]
            self.metadata["command"] = command
            self.output = (self.folder / "gpu.csv").open("w", encoding="utf-8")
            self.errors = (self.folder / "gpu-sampler.log").open("w", encoding="utf-8")
            self.process = subprocess.Popen(command, stdout=self.output, stderr=self.errors, text=True)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            self.metadata["reason"] = str(exc)
        finally:
            (self.folder / "gpu-before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
            self.metadata["idle_check"] = before["idle_check"]

    def stop(self):
        stopped = False
        if self.process is not None:
            if self.process.poll() is None:
                stopped = True
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except ProcessLookupError:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.metadata["sampler_exit_code"] = self.process.returncode
        for stream in (self.output, self.errors):
            if stream is not None:
                stream.close()
        path = self.folder / "gpu.csv"
        if self.process is None or not path.is_file():
            return self.metadata
        values = {key: [] for key in ("memory_used_mib", "utilization_percent", "clock_sm_mhz", "temperature_c")}
        valid, invalid = 0, 0
        with path.open(encoding="utf-8", errors="replace") as stream:
            for row in csv.reader(stream, skipinitialspace=True):
                identity = self.metadata.get("identity", {})
                if len(row) != 7 or row[1] != identity.get("uuid") or row[2] != identity.get("pci_bus_id"):
                    invalid += 1
                    continue
                valid += 1
                for key, raw in zip(values, row[3:]):
                    try:
                        value = float(raw)
                        if math.isfinite(value):
                            values[key].append(value)
                    except ValueError:
                        pass
        self.metadata.update(samples=valid, invalid_samples=invalid)
        if invalid:
            self.metadata["status"] = "invalid"
            self.metadata["reason"] = "sampler produced malformed or mismatched GPU identity rows"
            return self.metadata
        if not valid:
            self.metadata["reason"] = "sampler produced missing or mismatched GPU identity rows"
        elif not stopped:
            self.metadata["status"] = "partial"
            self.metadata["reason"] = "sampler exited early; retained values cover only part of the process"
        else:
            self.metadata["status"] = "available"
        for key, samples in values.items():
            if samples:
                self.metadata[f"sampled_min_{key}"] = min(samples)
                self.metadata[f"sampled_max_{key}"] = max(samples)
        return self.metadata


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path, hash_contents=False):
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file() or stat.st_size == 0:
        raise ValueError(f"missing or empty file: {path}")
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        result["sha256"] = sha256_file(path)
    return result


def git_output(folder, arguments):
    result = subprocess.run(["git", "-C", str(folder), *arguments], capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def source_identity(folder):
    revision = git_output(folder, ["rev-parse", "HEAD"])
    changes = git_output(folder, ["diff", "HEAD", "--binary"])
    status = git_output(folder, ["status", "--porcelain"])
    ggml = git_output(folder, ["rev-parse", "HEAD:ggml"])
    ggml_folder = Path(folder) / "ggml"
    ggml_revision = git_output(ggml_folder, ["rev-parse", "HEAD"]) if (ggml_folder / ".git").exists() else None
    return {
        "path": str(Path(folder).resolve()),
        "revision": revision.decode().strip() if revision else None,
        "dirty": bool(status) if status is not None else None,
        "tracked_diff_sha256": hashlib.sha256(changes).hexdigest() if changes is not None else None,
        "ggml_gitlink": ggml.decode().strip() if ggml else None,
        "ggml_revision": ggml_revision.decode().strip() if ggml_revision else None,
    }


def option_value(arguments, names):
    values = []
    for index, argument in enumerate(arguments):
        if argument in names:
            if index + 1 == len(arguments):
                raise ValueError(f"missing value after {argument}")
            values.append(arguments[index + 1])
        elif any(argument.startswith(name + "=") for name in names):
            raise ValueError("use separate option and value tokens for sd-cli")
    if len(values) != 1:
        raise ValueError(f"supply exactly one {'/'.join(names)}")
    return values[0]


def validate_arguments(arguments):
    for argument in arguments:
        if argument.split("=", 1)[0] in FORBIDDEN_FLAGS:
            raise ValueError(f"runner controls {argument.split('=', 1)[0]}")
    models = {flag: option_value(arguments, (flag,)) for flag in MODEL_FLAGS}
    seed = int(option_value(arguments, ("-s", "--seed")))
    width = int(option_value(arguments, ("-W", "--width")))
    height = int(option_value(arguments, ("-H", "--height")))
    frames = int(option_value(arguments, ("--video-frames",)))
    steps = int(option_value(arguments, ("--steps",)))
    option_value(arguments, ("-p", "--prompt"))
    if seed < 0 or min(width, height, steps) < 1 or frames < 5:
        raise ValueError("use a nonnegative seed, positive dimensions/steps and at least five frames")
    if "--fps" in arguments and int(option_value(arguments, ("--fps",))) != 24:
        raise ValueError("MiniMax-H3 requires 24 fps")
    if "--params-backend" in arguments:
        placement = option_value(arguments, ("--params-backend",))
        for assignment in placement.split(","):
            device = assignment.rsplit("=", 1)[-1].strip().lower()
            if device not in ("disk", "cpu", "cuda0"):
                raise ValueError("parameter backends are restricted to disk, cpu or cuda0")
    return models, {"width": (width + 31) // 32 * 32, "height": (height + 31) // 32 * 32,
                    "frames": 5 + (frames - 5 + 16) // 17 * 17, "fps": 24}


def parse_timings(log):
    if not re.search(r"Diffusion model runtime backend:\s*CUDA0\b", log, re.I):
        raise ValueError("diffusion runtime did not report CUDA0")
    timings = {}
    for name, pattern in PHASES.items():
        matches = re.findall(pattern, log)
        if len(matches) != 1:
            raise ValueError(f"expected one {name} measurement, found {len(matches)}")
        value, unit = matches[0]
        timings[name] = float(value) / (1000 if unit == "ms" else 1)
        if not math.isfinite(timings[name]) or timings[name] < 0:
            raise ValueError(f"invalid {name}")
    if timings["generation_s"] <= 0:
        raise ValueError("generation time must be positive")
    if sum(timings[name] for name in PHASES if name != "generation_s") > timings["generation_s"] + 0.1:
        raise ValueError("phase timings exceed total generation time")
    return timings


def media_metadata(probe, expected):
    streams = probe.get("streams", [])
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ValueError("expected exactly one video and one audio stream")
    video, audio = video[0], audio[0]
    try:
        frame_count = int(video.get("nb_read_frames") or video.get("nb_frames"))
        fps = float(Fraction(video["r_frame_rate"]))
        actual = {"width": int(video["width"]), "height": int(video["height"]),
                  "frames": frame_count, "fps": fps}
        if actual != expected:
            raise ValueError(f"video metadata {actual} differs from expected {expected}")
        if int(audio["channels"]) != 2 or int(audio["sample_rate"]) != 32000:
            raise ValueError("expected stereo 32000 Hz MiniMax-H3 audio")
        duration = float(probe.get("format", {}).get("duration", "nan"))
        if not math.isfinite(duration) or abs(duration - frame_count / fps) > 0.15:
            raise ValueError("container duration differs from video frame count / fps")
        audio_duration = audio.get("duration")
        if audio_duration not in (None, "N/A"):
            audio_duration = float(audio_duration)
            if not math.isfinite(audio_duration) or abs(audio_duration - frame_count / fps) > 0.15:
                raise ValueError("audio duration differs from video duration")
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"incomplete media metadata: {exc}") from exc
    return {**actual, "duration_s": duration, "audio_channels": 2, "audio_sample_rate": 32000,
            "video_codec": video.get("codec_name"), "audio_codec": audio.get("codec_name")}


def inspect_media(path, expected, ffprobe="ffprobe", ffmpeg="ffmpeg", timeout=120):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty output video: {path}")
    probe = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
                            "-of", "json", str(path)], capture_output=True, text=True, timeout=timeout, check=False)
    if probe.returncode != 0:
        raise ValueError(f"ffprobe failed: {probe.stderr.strip()}")
    metadata = media_metadata(json.loads(probe.stdout), expected)
    decoded = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-hwaccel", "none", "-threads", "1",
                              "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=timeout, check=False)
    if decoded.returncode != 0:
        raise ValueError(f"media decode failed: {decoded.stderr.strip()}")
    return {**metadata, "decode_check": "passed", "semantic_quality": "not assessed"}


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--hash-models", action="store_true", help="SHA256 model contents before timing; otherwise record paths, sizes and mtimes")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--nvidia-smi", default="nvidia-smi", help="optional GPU 0 telemetry executable")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="allow an explicitly recorded concurrent compute process on GPU 0")
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cli_args = args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args
    if args.warmup < 0 or args.runs < 1 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("warmup must be nonnegative; runs and timeout must be positive")
    try:
        models, expected = validate_arguments(cli_args)
        binary = file_identity(args.binary, True)
        if not os.access(binary["path"], os.X_OK):
            raise ValueError("binary is not executable")
        model_files = {flag: file_identity(path, args.hash_models) for flag, path in models.items()}
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty; existing evidence will not be overwritten")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", GGML_VK_VISIBLE_DEVICES="0")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "timing_scope": "fresh process including model load and output writing; warmup processes excluded",
              "runner": file_identity(__file__, True),
              "source": source_identity(args.source_dir), "binary": binary, "models": model_files,
              "model_hashes_recorded": args.hash_models, "machine": platform.platform(),
              "environment": {key: environment[key] for key in ENVIRONMENT_KEYS if key in environment},
              "sd_cli_args": cli_args, "expected_media": expected, "warmup": args.warmup, "runs": args.runs,
              "gpu_telemetry": "optional nvidia-smi GPU 0 sampling at 250 ms; unavailable or partial data is labelled per run",
              "measurements": []}
    report_path = output / "report.json"
    write_report(report_path, report)
    try:
        for index in range(args.warmup + args.runs):
            warmup = index < args.warmup
            label = f"warmup-{index + 1:02d}" if warmup else f"run-{index - args.warmup + 1:02d}"
            folder = output / label
            folder.mkdir()
            media = folder / "output.avi"
            log_path = folder / "sd-cli.log"
            command = [binary["path"], "-M", "vid_gen", *cli_args, "--backend", "cuda0", "-b", "1", "-v", "-o", str(media)]
            measurement = {"label": label, "warmup": warmup, "command": command, "log": str(log_path), "media": str(media), "status": "running"}
            report["measurements"].append(measurement)
            write_report(report_path, report)
            print(f"{label}: running on physical CUDA GPU 0", flush=True)
            telemetry = GpuTelemetry(folder, args.nvidia_smi, args.allow_busy_gpu)
            try:
                telemetry.start()
                start = time.perf_counter()
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout_seconds, check=False)
                measurement.update(wall_s=time.perf_counter() - start, returncode=completed.returncode)
            finally:
                measurement["gpu_telemetry"] = telemetry.stop()
            if completed.returncode != 0:
                raise RuntimeError(f"{label} exited {completed.returncode}; see {log_path}")
            measurement.update(parse_timings(log_path.read_text(encoding="utf-8", errors="replace")))
            measurement["media_validation"] = inspect_media(media, expected, args.ffprobe, args.ffmpeg)
            measurement["status"] = "passed"
            write_report(report_path, report)
        measured = [item for item in report["measurements"] if not item["warmup"]]
        report["median"] = {field: statistics.median(item[field] for item in measured) for field in (*PHASES, "wall_s")}
        report["status"] = "passed"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report.update(status="failed", error=str(exc))
        if report["measurements"]:
            report["measurements"][-1]["status"] = "failed"
        write_report(report_path, report)
        raise RuntimeError(f"{exc}; evidence retained at {report_path}") from exc
    write_report(report_path, report)
    print(f"Median timings saved to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
