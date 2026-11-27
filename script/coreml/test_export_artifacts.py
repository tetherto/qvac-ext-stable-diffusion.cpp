"""Checks the commands export_artifacts.py plans for each model, without downloading or converting anything.

Run: python -m unittest script/coreml/test_export_artifacts.py
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "export_artifacts.py"
SD_TURBO_REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"


def planned_commands(models):
    with tempfile.TemporaryDirectory() as out:
        result = subprocess.run([sys.executable, str(SCRIPT), "--out", out, "--models", models, "--dry-run"],
                                capture_output=True, text=True)
    return result


def command_with(commands, text):
    matches = [line for line in commands.splitlines() if text in line]
    return matches[0] if matches else ""


class SdTurboPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commands = planned_commands("sdturbo").stdout

    def test_downloads_the_pinned_revision(self):
        download = command_with(self.commands, "hf download stabilityai/sd-turbo")
        self.assertIn(f"--revision {SD_TURBO_REVISION}", download)
        self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", download)

    def test_converts_the_whole_pipeline_to_gguf(self):
        convert = command_with(self.commands, "-M convert")
        self.assertIn("sdturbo-Q8_0.gguf", convert)
        self.assertIn("--type q8_0", convert)

    def test_exports_an_eight_bit_unet_for_the_neural_engine(self):
        unet = command_with(self.commands, "sdturbo_unet_p8.mlpackage --precision")
        self.assertIn("--precision p8", unet)
        self.assertNotIn("--attention", unet)
        self.assertNotIn("--batch", unet)

    def test_exports_an_fp16_unet_with_standard_attention_for_the_gpu(self):
        unet = command_with(self.commands, "sdturbo_unet_fp16.mlpackage --compile")
        self.assertIn("convert_unet.py", unet)
        self.assertNotIn("--attention", unet)
        self.assertNotIn("--precision", unet)
        self.assertNotIn("--batch", unet)

    def test_exports_the_tiny_decoder(self):
        self.assertIn("taesd_decoder.mlpackage", command_with(self.commands, "convert_tae_decoder.py"))


class ModelSelection(unittest.TestCase):
    def test_default_models_leave_sd_turbo_out(self):
        self.assertNotIn("sd-turbo", planned_commands("sdxs,sd21base").stdout)

    def test_sd21base_and_sdturbo_share_one_tiny_decoder(self):
        commands = planned_commands("sd21base,sdturbo").stdout
        self.assertIn("sd21base_unet_p6.mlpackage", commands)
        self.assertIn("sdturbo_unet_fp16.mlpackage", commands)

    def test_unknown_model_is_rejected(self):
        result = planned_commands("sdturbo,nope")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown models", result.stderr)


if __name__ == "__main__":
    unittest.main()
