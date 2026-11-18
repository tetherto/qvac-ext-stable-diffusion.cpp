#!/usr/bin/env python3
"""Model-free regression checks for MiniMax-H3 benchmark evidence and isolation."""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench_minimax_h3 as bench


LOG = """Diffusion model runtime backend: CUDA0
get_learned_condition completed, taking 500 ms
sampling completed, taking 8.00s
decoding audio latent completed, taking 0.20s
decode_first_stage completed, taking 1.00s
generate_video completed in 10.00s
"""
EXPECTED = {"width": 480, "height": 288, "frames": 5, "fps": 24}


def probe_data(**video_overrides):
    video = {"codec_type": "video", "codec_name": "mjpeg", "width": 480, "height": 288,
             "r_frame_rate": "24/1", "nb_read_frames": "5"}
    video.update(video_overrides)
    return {"streams": [video, {"codec_type": "audio", "codec_name": "pcm_s16le", "channels": 2,
                                "sample_rate": "32000", "duration": "0.208333"}],
            "format": {"duration": "0.208333"}}


def cli_args(models):
    return [item for flag, path in models.items() for item in (flag, str(path))] + [
        "-p", "A paper boat on a lake, with piano music.", "--seed", "42", "-W", "480", "-H", "288",
        "--video-frames", "5", "--steps", "1", "--diffusion-fa", "--params-backend", "disk"]


class TimingsTest(unittest.TestCase):
    def test_parses_phases_and_keeps_process_total_separate(self):
        values = bench.parse_timings(LOG)
        self.assertEqual(values["conditioning_s"], 0.5)
        self.assertEqual(values["sampling_s"], 8)
        self.assertEqual(values["audio_decode_s"], 0.2)
        self.assertEqual(values["video_decode_s"], 1)
        self.assertEqual(values["generation_s"], 10)

    def test_rejects_missing_duplicate_or_inconsistent_phases(self):
        for text, error in [(LOG.replace("decode_first_stage", "other"), "video_decode_s"),
                            (LOG + "sampling completed, taking 8.00s\n", "sampling_s"),
                            (LOG.replace("10.00s", "1.00s"), "exceed"),
                            (LOG.replace("CUDA0", "CUDA1"), "CUDA0"),
                            (LOG.replace("CUDA0", "CPU"), "CUDA0")]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                bench.parse_timings(text)


class ArgumentsTest(unittest.TestCase):
    def setUp(self):
        self.arguments = cli_args({flag: "m.gguf" for flag in bench.MODEL_FLAGS})

    def test_native_alignment_is_recorded(self):
        arguments = self.arguments.copy()
        arguments[arguments.index("480")] = "481"
        arguments[arguments.index("5")] = "120"
        _, expected = bench.validate_arguments(arguments)
        self.assertEqual(expected, {"width": 512, "height": 288, "frames": 124, "fps": 24})

    def test_backend_output_and_auto_placement_cannot_be_overridden(self):
        for argument in ("--backend", "--backend=cuda1", "--auto-fit", "--split-mode", "--output", "-M", "--config"):
            with self.subTest(argument=argument), self.assertRaisesRegex(ValueError, "runner controls"):
                bench.validate_arguments(self.arguments + [argument, "x"])

    def test_fixed_seed_models_dimensions_and_samples_are_required(self):
        for flag in ("--audio-vae", "--seed", "--steps"):
            arguments = self.arguments.copy()
            index = arguments.index(flag)
            del arguments[index:index + 2]
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                bench.validate_arguments(arguments)
        arguments = self.arguments.copy()
        arguments[arguments.index("42")] = "-1"
        with self.assertRaisesRegex(ValueError, "nonnegative seed"):
            bench.validate_arguments(arguments)

    def test_parameter_placement_cannot_select_another_gpu_backend(self):
        for value in ("cuda1", "diffusion=cuda0&cuda1", "vae=vulkan1"):
            arguments = self.arguments.copy()
            arguments[-1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "restricted"):
                bench.validate_arguments(arguments)


class MediaTest(unittest.TestCase):
    def test_expected_native_video_and_audio_are_valid(self):
        actual = bench.media_metadata(probe_data(), EXPECTED)
        self.assertEqual(actual["frames"], 5)
        self.assertEqual(actual["audio_channels"], 2)

    def test_missing_audio_and_wrong_shape_fps_or_frame_count_fail(self):
        for overrides in ({"width": 512}, {"nb_read_frames": "4"}, {"r_frame_rate": "30/1"}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "differs"):
                bench.media_metadata(probe_data(**overrides), EXPECTED)
        missing = probe_data()
        missing["streams"].pop()
        with self.assertRaisesRegex(ValueError, "one video and one audio"):
            bench.media_metadata(missing, EXPECTED)

    def test_audio_configuration_and_duration_are_checked(self):
        for field, value in (("channels", 1), ("sample_rate", "16000"), ("duration", "2.0"), ("duration", "nan")):
            data = probe_data()
            data["streams"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bench.media_metadata(data, EXPECTED)
        data = probe_data()
        data["format"]["duration"] = "nan"
        with self.assertRaisesRegex(ValueError, "container duration"):
            bench.media_metadata(data, EXPECTED)

    def test_absent_output_and_decode_errors_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.avi"
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                bench.inspect_media(path, EXPECTED)
            path.write_bytes(b"placeholder")
            responses = [subprocess.CompletedProcess([], 0, json.dumps(probe_data()), ""),
                         subprocess.CompletedProcess([], 1, "", "corrupt frame")]
            with mock.patch.object(bench.subprocess, "run", side_effect=responses) as run:
                with self.assertRaisesRegex(ValueError, "corrupt frame"):
                    bench.inspect_media(path, EXPECTED)
                command = run.call_args_list[1].args[0]
                self.assertIn("none", command)
                self.assertIn("-xerror", command)


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = {}
        for index, flag in enumerate(bench.MODEL_FLAGS):
            path = self.root / f"model-{index}.gguf"
            path.write_bytes(b"synthetic model metadata fixture")
            self.models[flag] = path
        self.binary = self.root / "sd-cli-stub"
        self.output = self.root / "evidence"

    def write_stub(self, body):
        self.binary.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
        self.binary.chmod(0o755)

    def arguments(self, *options):
        return ["--binary", str(self.binary), "--output-dir", str(self.output), "--warmup", "1", "--runs", "2",
                "--source-dir", str(self.root), "--nvidia-smi", str(self.root / "missing-nvidia-smi"),
                *options, "--", *cli_args(self.models)]

    def test_actual_process_receives_gpu_zero_and_preserves_command_and_identity(self):
        self.write_stub("import json,os,pathlib,sys\n"
                        "assert os.environ['CUDA_VISIBLE_DEVICES']=='0'\n"
                        "assert os.environ['GGML_VK_VISIBLE_DEVICES']=='0'\n"
                        "assert sys.argv[sys.argv.index('--backend')+1]=='cuda0'\n"
                        "pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'AVI fixture')\n"
                        f"print({LOG!r})\n")
        with mock.patch.object(bench, "inspect_media", return_value={"decode_check": "passed"}), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1", "PRIVATE_TEST_SECRET": "do not record"}):
            self.assertEqual(bench.main(self.arguments("--hash-models")), 0)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["measurements"]), 3)
        self.assertEqual([item["warmup"] for item in report["measurements"]], [True, False, False])
        self.assertEqual(report["median"]["generation_s"], 10)
        self.assertEqual(report["environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("PRIVATE_TEST_SECRET", report["environment"])
        self.assertEqual(report["binary"]["sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(report["runner"]["sha256"], hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest())
        self.assertEqual(report["models"]["--vae"]["sha256"], hashlib.sha256(self.models["--vae"].read_bytes()).hexdigest())
        self.assertEqual(report["sd_cli_args"], cli_args(self.models))
        self.assertTrue(all(Path(item["log"]).is_file() for item in report["measurements"]))
        self.assertEqual(len({item["media"] for item in report["measurements"]}), 3)

    def test_nonzero_process_keeps_failure_evidence_and_never_reports_a_median(self):
        self.write_stub("import sys\nprint('native allocation failed',flush=True)\nsys.exit(7)\n")
        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            bench.main(self.arguments())
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["measurements"][0]["returncode"], 7)
        self.assertNotIn("median", report)
        self.assertIn("native allocation failed", Path(report["measurements"][0]["log"]).read_text())

    def test_timeout_and_missing_media_keep_failure_report(self):
        self.write_stub("import time\ntime.sleep(10)\n")
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            bench.main(self.arguments("--timeout-seconds", "0.02"))
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        other_output = self.root / "missing-media"
        self.write_stub(f"print({LOG!r})\n")
        arguments = self.arguments()
        arguments[arguments.index(str(self.output))] = str(other_output)
        with self.assertRaisesRegex(RuntimeError, "missing or empty"):
            bench.main(arguments)
        self.assertEqual(json.loads((other_output / "report.json").read_text())["status"], "failed")

    def test_existing_evidence_is_never_overwritten(self):
        self.write_stub("raise RuntimeError('must not execute')\n")
        self.output.mkdir()
        marker = self.output / "run-01.log"
        marker.write_text("keep this evidence")
        with self.assertRaises(SystemExit):
            bench.main(self.arguments())
        self.assertEqual(marker.read_text(), "keep this evidence")

    def test_sampler_is_stopped_when_native_process_fails_or_times_out(self):
        for index, body in enumerate(("import sys\nsys.exit(7)\n", "import time\ntime.sleep(10)\n")):
            with self.subTest(index=index):
                self.output = self.root / f"failure-{index}"
                self.write_stub(body)
                with mock.patch.object(bench, "GpuTelemetry") as telemetry:
                    telemetry.return_value.stop.return_value = {"status": "unavailable", "reason": "fixture"}
                    with self.assertRaises(RuntimeError):
                        bench.main(self.arguments("--timeout-seconds", "0.05"))
                    telemetry.return_value.stop.assert_called_once()
                report = json.loads((self.output / "report.json").read_text())
                self.assertEqual(report["measurements"][0]["gpu_telemetry"]["status"], "unavailable")


class TelemetryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.identity = subprocess.CompletedProcess([], 0, "GPU-zero, 00000000:01:00.0, RTX 5090, 590.0, 32607\n", "")
        self.idle = subprocess.CompletedProcess([], 0, "", "")
        self.row = "2026/11/06 13:00:00.000, GPU-zero, 00000000:01:00.0, 1200, 82, 2400, 64\n"

    def sampler_mock(self, command, **kwargs):
        self.command = command
        kwargs["stdout"].write(self.row)
        kwargs["stdout"].flush()
        process = mock.Mock()
        process.poll.return_value = None
        process.returncode = -15
        process.wait.return_value = -15
        return process

    def test_only_gpu_zero_is_queried_and_sampled_memory_is_explicitly_whole_device(self):
        with mock.patch.object(bench.subprocess, "run", side_effect=[self.identity, self.idle]) as query, \
                mock.patch.object(bench.subprocess, "Popen", side_effect=self.sampler_mock) as launch:
            telemetry = bench.GpuTelemetry(self.folder)
            telemetry.start()
            result = telemetry.stop()
        for call in (*query.call_args_list, launch.call_args):
            command = call.args[0]
            self.assertEqual(command[command.index("-i") + 1], "0")
        self.assertEqual(result["idle_check"], "idle")
        self.assertEqual(result["identity"]["uuid"], "GPU-zero")
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["sampled_max_memory_used_mib"], 1200)
        self.assertEqual(result["sampled_max_temperature_c"], 64)
        self.assertIn("whole-device", result["memory_scope"])
        self.assertIn("miss the true peak", result["memory_scope"])
        telemetry.process.terminate.assert_called_once()
        telemetry.process.wait.assert_called_once()
        self.assertTrue(telemetry.output.closed)
        self.assertTrue(telemetry.errors.closed)

    def test_missing_nvidia_smi_is_explicitly_unavailable(self):
        with mock.patch.object(bench.subprocess, "run", side_effect=FileNotFoundError("missing nvidia-smi")):
            telemetry = bench.GpuTelemetry(self.folder)
            telemetry.start()
            result = telemetry.stop()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["idle_check"], "unavailable")
        self.assertIn("missing nvidia-smi", result["reason"])
        self.assertNotIn("sampled_max_memory_used_mib", result)

    def test_busy_gpu_is_refused_without_terminating_any_existing_process(self):
        busy = subprocess.CompletedProcess([], 0, "123, existing-job, 2000\n", "")
        with mock.patch.object(bench.subprocess, "run", side_effect=[self.identity, busy]), \
                mock.patch.object(bench.subprocess, "Popen") as launch:
            telemetry = bench.GpuTelemetry(self.folder)
            try:
                with self.assertRaisesRegex(RuntimeError, "running compute processes"):
                    telemetry.start()
            finally:
                telemetry.stop()
            launch.assert_not_called()
        before = json.loads((self.folder / "gpu-before.json").read_text())
        self.assertEqual(before["idle_check"], "busy")
        self.assertIn("123", before["compute_processes"])

    def test_busy_opt_in_and_early_exit_are_recorded_as_partial_sampling(self):
        busy = subprocess.CompletedProcess([], 0, "123, existing-job, 2000\n", "")
        with mock.patch.object(bench.subprocess, "run", side_effect=[self.identity, busy]), \
                mock.patch.object(bench.subprocess, "Popen", side_effect=self.sampler_mock):
            telemetry = bench.GpuTelemetry(self.folder, allow_busy=True)
            telemetry.start()
            telemetry.process.poll.return_value = 1
            telemetry.process.returncode = 1
            result = telemetry.stop()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["idle_check"], "busy")
        self.assertTrue(json.loads((self.folder / "gpu-before.json").read_text())["allow_busy_gpu"])

    def test_mismatched_gpu_identity_never_yields_a_memory_maximum(self):
        self.row = self.row.replace("GPU-zero", "GPU-other")
        with mock.patch.object(bench.subprocess, "run", side_effect=[self.identity, self.idle]), \
                mock.patch.object(bench.subprocess, "Popen", side_effect=self.sampler_mock):
            telemetry = bench.GpuTelemetry(self.folder)
            telemetry.start()
            result = telemetry.stop()
        self.assertEqual(result["status"], "invalid")
        self.assertNotIn("sampled_max_memory_used_mib", result)

    def test_sampler_ignoring_termination_is_killed_and_reaped(self):
        with mock.patch.object(bench.subprocess, "run", side_effect=[self.identity, self.idle]), \
                mock.patch.object(bench.subprocess, "Popen", side_effect=self.sampler_mock):
            telemetry = bench.GpuTelemetry(self.folder)
            telemetry.start()
            telemetry.process.wait.side_effect = [subprocess.TimeoutExpired("sampler", 5), -9]
            telemetry.stop()
        telemetry.process.kill.assert_called_once()
        self.assertEqual(telemetry.process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
