#!/usr/bin/env python3
"""Opt-in M4 integration checks for a complete FLUX.2 Core ML sidecar."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "SDCPP_TEST_SD_CLI",
    "SDCPP_TEST_FLUX2_DIFFUSION",
    "SDCPP_TEST_FLUX2_VAE",
    "SDCPP_TEST_FLUX2_LLM",
    "SDCPP_TEST_FLUX2_COREML",
    "SDCPP_TEST_FLUX2_FIXTURE",
)


@unittest.skipUnless(all(os.environ.get(name) for name in NAMES),
                     "set all SDCPP_TEST_* paths for the M4 integration test")
class Flux2CoreMLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.paths = {name: Path(os.environ[name]).resolve() for name in NAMES}
        for name, path in self.paths.items():
            self.assertTrue(path.exists(), f"{name} is missing: {path}")

    def command(self, output):
        return [
            str(self.paths["SDCPP_TEST_SD_CLI"]), "-M", "img_gen",
            "--diffusion-model", str(self.paths["SDCPP_TEST_FLUX2_DIFFUSION"]),
            "--vae", str(self.paths["SDCPP_TEST_FLUX2_VAE"]),
            "--llm", str(self.paths["SDCPP_TEST_FLUX2_LLM"]),
            "-p", "a lovely cat", "--seed", "42",
            "-W", "1024", "-H", "1024", "--steps", "4",
            "--sampling-method", "euler", "--cfg-scale", "1",
            "--diffusion-fa", "--clip-on-cpu", "--offload-to-cpu",
            "-v", "-o", str(output),
        ]

    def test_fixture_matches_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "parity.json"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "script/compare_flux_coreml_fixture.py"),
                 "--model", str(self.paths["SDCPP_TEST_FLUX2_COREML"]),
                 "--fixture", str(self.paths["SDCPP_TEST_FLUX2_FIXTURE"]),
                 "--output", str(report)],
                cwd=ROOT, capture_output=True, text=True, timeout=900,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(report.read_text())["passes_reference"])

    def test_multistep_generation_uses_coreml_without_ggml_diffusion_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "image.png"
            env = os.environ.copy()
            env["SDCPP_FLUX2_COREML_MODEL"] = str(self.paths["SDCPP_TEST_FLUX2_COREML"])
            completed = subprocess.run(
                self.command(output), cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=900,
            )
            log = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, log[-4000:])
            self.assertIn("Using Core ML FLUX.2 denoiser", log)
            self.assertRegex(log, r"diffusion_model 0\.00MB")
            self.assertTrue(output.is_file() and output.stat().st_size > 0)

    def test_invalid_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["SDCPP_FLUX2_COREML_MODEL"] = str(Path(directory) / "missing.mlmodelc")
            completed = subprocess.run(
                self.command(Path(directory) / "image.png"),
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
            )
            log = completed.stdout + completed.stderr
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("could not be loaded", log)

    def test_lora_requests_fail_before_application(self):
        for mode in ("immediately", "at_runtime", "auto"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                # An existing path makes the CLI preserve the request. Its
                # contents must never be loaded by either LoRA apply path.
                lora = Path(directory) / "unsupported.safetensors"
                lora.touch()
                output = Path(directory) / "image.png"
                command = self.command(output)
                command[command.index("-p") + 1] = f"a lovely cat <lora:{lora}:1>"
                command.extend(["--lora-apply-mode", mode])
                env = os.environ.copy()
                env["SDCPP_FLUX2_COREML_MODEL"] = str(self.paths["SDCPP_TEST_FLUX2_COREML"])
                completed = subprocess.run(
                    command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
                )
                log = completed.stdout + completed.stderr
                self.assertNotEqual(completed.returncode, 0, log[-4000:])
                self.assertIn("LoRA requests are not supported by the Core ML FLUX.2 denoiser", log)
                self.assertNotIn("attempting to apply", log)
                self.assertNotIn("flux denoiser compute completed", log)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
