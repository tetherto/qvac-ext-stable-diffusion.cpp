"""Checks the commands export_artifacts.py plans for each model without downloading anything, and its GGUF step
on a small random diffusers pipeline (needs sd-cli: build/bin/sd-cli, or the path in SD_CLI).

Run: python -m unittest script/coreml/test_export_artifacts.py
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "export_artifacts.py"
SD_CLI = Path(os.environ.get("SD_CLI", HERE.parents[1] / "build" / "bin" / "sd-cli")).resolve()
SD_TURBO_REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"

GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
UNET_PREFIX = "unet."
VAE_PREFIX = "vae."
TEXT_ENCODER_PREFIX = "te."

sys.path.insert(0, str(HERE))
import export_artifacts  # noqa: E402


def planned_commands(models=None):
    selection = ["--models", models] if models is not None else []
    with tempfile.TemporaryDirectory() as out:
        result = subprocess.run([sys.executable, str(SCRIPT), "--out", out, *selection, "--dry-run"],
                                capture_output=True, text=True)
    return result


def commands_with(commands, text):
    return [line for line in commands.splitlines() if text in line]


def read_number(file, fmt):
    return struct.unpack(fmt, file.read(struct.calcsize(fmt)))[0]


def read_string(file):
    return file.read(read_number(file, "<Q")).decode()


def skip_array(file):
    item_type = read_number(file, "<I")
    for _ in range(read_number(file, "<Q")):
        skip_value(file, item_type)


def skip_value(file, value_type):
    if value_type == GGUF_STRING:
        read_string(file)
    elif value_type == GGUF_ARRAY:
        skip_array(file)
    else:
        file.seek(GGUF_SCALAR_SIZES[value_type], os.SEEK_CUR)


def skip_metadata(file, count):
    for _ in range(count):
        read_string(file)
        skip_value(file, read_number(file, "<I"))


def read_tensor_name(file):
    name = read_string(file)
    dims = read_number(file, "<I")
    file.seek(dims * 8 + 4 + 8, os.SEEK_CUR)
    return name


def gguf_tensor_names(path):
    with open(path, "rb") as file:
        file.seek(8)
        tensor_count = read_number(file, "<Q")
        skip_metadata(file, read_number(file, "<Q"))
        return [read_tensor_name(file) for _ in range(tensor_count)]


def write_tiny_fp16_pipeline(root):
    """An SD 2.x-shaped diffusers folder with the fp16 file names SD-Turbo ships: random UNet and VAE, and a
    text encoder reduced to its token embedding."""
    import torch
    from diffusers import AutoencoderKL
    from safetensors.torch import save_file
    from test_convert_unet import tiny_unet

    tiny_unet(sd2=True).save_pretrained(root / "unet", variant="fp16")
    AutoencoderKL(block_out_channels=(32,), norm_num_groups=8).save_pretrained(root / "vae", variant="fp16")
    (root / "text_encoder").mkdir()
    save_file({"text_model.embeddings.token_embedding.weight": torch.zeros(8, 32)},
              str(root / "text_encoder" / "model.fp16.safetensors"))
    (root / "text_encoder" / "config.json").write_text("{}")
    (root / "model_index.json").write_text("{}")
    return root


def count_with_prefix(names, prefix):
    return sum(name.startswith(prefix) for name in names)


def command_with(commands, text):
    matches = commands_with(commands, text)
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


@unittest.skipUnless(SD_CLI.exists(), f"{SD_CLI} is not built")
class GgufConversion(unittest.TestCase):
    def test_gguf_holds_the_unet_the_text_encoder_and_the_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tiny_fp16_pipeline(Path(tmp) / "pipeline")
            gguf = Path(tmp) / "tiny-Q8_0.gguf"
            export_artifacts.export_q8_gguf(src, Path(tmp) / "view", gguf, SD_CLI, dry_run=False)
            names = gguf_tensor_names(gguf)
        for prefix in (UNET_PREFIX, VAE_PREFIX, TEXT_ENCODER_PREFIX):
            with self.subTest(prefix=prefix):
                self.assertGreater(count_with_prefix(names, prefix), 0)


class ModelSelection(unittest.TestCase):
    def test_default_models_leave_sd_turbo_out(self):
        result = planned_commands()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sdxs_unet_fp16.mlpackage", result.stdout)
        self.assertIn("sd21base_unet_p6.mlpackage", result.stdout)
        self.assertNotIn("sdturbo", result.stdout)
        self.assertNotIn("sd-turbo", result.stdout)

    def test_sd21base_and_sdturbo_share_one_tiny_decoder(self):
        commands = planned_commands("sd21base,sdturbo").stdout
        self.assertIn("sd21base_unet_p6.mlpackage", commands)
        self.assertIn("sdturbo_unet_fp16.mlpackage", commands)
        self.assertEqual(len(commands_with(commands, "convert_tae_decoder.py")), 1)
        self.assertEqual(len(commands_with(commands, "hf download madebyollin/taesd")), 1)

    def test_unknown_model_is_rejected(self):
        result = planned_commands("sdturbo,nope")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown models", result.stderr)


if __name__ == "__main__":
    unittest.main()
