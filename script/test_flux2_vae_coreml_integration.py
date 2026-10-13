#!/usr/bin/env python3
"""Opt-in macOS engine checks for a compiled FLUX.2 VAE decoder."""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from compare_flux_coreml_fixture import error_metrics
from export_flux2_vae_coreml import validate_fixture


NAMES = ("SDCPP_TEST_SD_CLI", "SDCPP_TEST_FLUX2_DIFFUSION", "SDCPP_TEST_FLUX2_VAE",
         "SDCPP_TEST_FLUX2_LLM", "SDCPP_TEST_FLUX2_COREML_VAE")


@unittest.skipUnless(all(os.environ.get(name) for name in NAMES), "set all SDCPP_TEST_* model and CLI paths")
class Flux2VAECoreMLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.paths = {name: Path(os.environ[name]).resolve() for name in NAMES}
        for path in self.paths.values():
            self.assertTrue(path.exists(), str(path))

    def command(self, output, steps=1):
        return [str(self.paths["SDCPP_TEST_SD_CLI"]), "-M", "img_gen",
                "--diffusion-model", str(self.paths["SDCPP_TEST_FLUX2_DIFFUSION"]),
                "--vae", str(self.paths["SDCPP_TEST_FLUX2_VAE"]),
                "--llm", str(self.paths["SDCPP_TEST_FLUX2_LLM"]),
                "--backend", "metal", "--params-backend", "cpu", "--threads", "4",
                "-p", "a lovely cat", "--seed", "42", "-W", "1024", "-H", "1024",
                "--steps", str(steps), "--sampling-method", "euler", "--cfg-scale", "1",
                "--diffusion-fa", "-v", "-o", str(output)]

    def run_cli(self, command, sidecar, capture=None):
        env = os.environ.copy()
        for name in list(env):
            if name.startswith("SDCPP_") and not name.startswith("SDCPP_TEST_"):
                env.pop(name)
        if sidecar:
            env["SDCPP_FLUX2_COREML_VAE_MODEL"] = str(sidecar)
        if capture:
            env["SDCPP_VAE_CAPTURE_DIR"] = str(capture)
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
        return result.returncode, result.stdout + result.stderr

    def test_generation_matches_metal_without_native_decoder_compute(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, sidecar in [("metal", None), ("coreml", self.paths["SDCPP_TEST_FLUX2_COREML_VAE"])]:
                status, log = self.run_cli(self.command(root / (name + ".png"), steps=4), sidecar, root / name)
                self.assertEqual(status, 0, log[-6000:])
                self.assertTrue((root / (name + ".png")).is_file())
                validate_fixture(root / name)
                if sidecar:
                    self.assertIn("Using Core ML FLUX.2 VAE decoder (CPU+GPU)", log)
                    self.assertNotIn("vae compute buffer size:", log)
                for measurement in re.findall(r"(?:decode_first_stage completed, taking [\d.]+s|generate_image completed in [\d.]+s)", log):
                    print(f"{name}: {measurement}", flush=True)
            native = np.fromfile(root / "metal/output.bin", dtype=np.float32)
            coreml = np.fromfile(root / "coreml/output.bin", dtype=np.float32)
            self.assertEqual((root / "metal/latent.bin").read_bytes(), (root / "coreml/latent.bin").read_bytes())
            metrics = error_metrics(coreml, native, np)
            print(f"engine VAE normalized RMSE: {metrics['normalized_rmse']:.6g}", flush=True)
            self.assertLess(metrics["normalized_rmse"], 0.001)

    def test_reference_image_keeps_native_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "image.png"
            reference = Path(__file__).resolve().parents[1] / "assets/flux2/flux2-klein-4b.png"
            self.assertTrue(reference.is_file())
            status, log = self.run_cli(self.command(output) + ["-r", str(reference)], self.paths["SDCPP_TEST_FLUX2_COREML_VAE"])
            self.assertEqual(status, 0, log[-6000:])
            self.assertIn("computing vae encode graph completed", log)
            self.assertIn("Using Core ML FLUX.2 VAE decoder (CPU+GPU)", log)
            self.assertTrue(output.is_file())

    def test_missing_sidecar_fails_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "image.png"
            status, log = self.run_cli(self.command(output), Path(directory) / "missing.mlmodelc")
            self.assertNotEqual(status, 0)
            self.assertIn("Core ML FLUX.2 VAE could not be loaded", log)
            self.assertNotIn("flux denoiser compute completed", log)
            self.assertFalse(output.exists())

    def test_wrong_resolution_fails_without_metal_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "image.png"
            command = self.command(output)
            command[command.index("-W") + 1] = "512"
            command[command.index("-H") + 1] = "512"
            status, log = self.run_cli(command, self.paths["SDCPP_TEST_FLUX2_COREML_VAE"])
            self.assertNotEqual(status, 0)
            self.assertIn("latent shape does not match the compiled decoder", log)
            self.assertNotIn("vae compute buffer size:", log)
            self.assertFalse(output.exists())

    def test_tiling_fails_without_metal_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "image.png"
            status, log = self.run_cli(self.command(output) + ["--vae-tiling"], self.paths["SDCPP_TEST_FLUX2_COREML_VAE"])
            self.assertNotEqual(status, 0)
            self.assertIn("does not support video, tiling, or circular padding", log)
            self.assertNotIn("vae compute buffer size:", log)
            self.assertFalse(output.exists())

    def test_malformed_model_is_rejected_at_load(self):
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 128, 64, 64), dtype=types.fp32)], opset_version=ct.target.macOS15)
            def bad_decoder(latent):
                return mb.mul(x=latent, y=np.float32(0.5), name="output")
            model = ct.convert(bad_decoder, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)
            package = root / "bad.mlpackage"
            model.save(str(package))
            compiled = ct.models.utils.compile_model(str(package))
            status, log = self.run_cli(self.command(root / "image.png"), compiled)
            self.assertNotEqual(status, 0)
            self.assertIn("must map [1,128,H,W]", log)
            self.assertNotIn("flux denoiser compute completed", log)

    def test_nonfinite_output_is_rejected_before_clamping(self):
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 128, 64, 64), dtype=types.fp32)], opset_version=ct.target.macOS15)
            def bad_decoder(latent):
                x = mb.reduce_mean(x=latent, axes=[1], keep_dims=True)
                x = mb.upsample_nearest_neighbor(x=x, scale_factor_height=16.0, scale_factor_width=16.0)
                x = mb.tile(x=x, reps=[1, 3, 1, 1])
                return mb.mul(x=x, y=np.float32(float("nan")), name="output")
            model = ct.convert(bad_decoder, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)
            package = root / "bad.mlpackage"
            model.save(str(package))
            compiled = ct.models.utils.compile_model(str(package))
            status, log = self.run_cli(self.command(root / "image.png"), compiled)
            self.assertNotEqual(status, 0)
            self.assertIn("returned a non-finite value", log)
            self.assertFalse((root / "image.png").exists())


if __name__ == "__main__":
    unittest.main()
