#!/usr/bin/env python3
"""Validate the VAE fixture boundary used by direct MIL conversion."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from export_flux2_vae_coreml import validate_dimensions, validate_fixture


class VAEFixtureTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {"format": 1, "kind": "flux2-vae-decode", "tensors": {}}
        for name, shape in [("latent", [2, 1, 128, 1]), ("output", [32, 16, 3, 1])]:
            size = math.prod(shape) * 4
            filename = name + ".bin"
            with (self.root / filename).open("wb") as file:
                file.truncate(size)
            self.manifest["tensors"][name] = {"file": filename, "ne": shape, "type": "f32", "bytes": size}

    def validate(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        return validate_fixture(self.root)

    def test_non_square_fixture(self):
        self.assertEqual(self.validate(), self.manifest)

    def test_rejects_denoiser_fixture(self):
        self.manifest["kind"] = "flux-denoiser"
        with self.assertRaisesRegex(ValueError, "VAE decode fixture"):
            self.validate()

    def test_rejects_normalized_unpacked_or_batched_latents(self):
        for shape in ([2, 4, 32, 1], [2, 1, 64, 2]):
            with self.subTest(shape=shape):
                self.manifest["tensors"]["latent"]["ne"] = shape
                with self.assertRaisesRegex(ValueError, "must map"):
                    self.validate()

    def test_rejects_wrong_rgb_shape(self):
        self.manifest["tensors"]["output"]["ne"] = [16, 32, 3, 1]
        with self.assertRaisesRegex(ValueError, "must map"):
            self.validate()

    def test_rejects_truncated_tensor(self):
        (self.root / "output.bin").write_bytes(b"\0")
        with self.assertRaisesRegex(ValueError, "file or size"):
            self.validate()

    def test_rejects_tensor_outside_fixture(self):
        self.manifest["tensors"]["latent"]["file"] = "../latent.bin"
        with self.assertRaisesRegex(ValueError, "file or size"):
            self.validate()

    def test_image_dimensions(self):
        validate_dimensions(1024, 768)
        for width, height in [(0, 16), (17, 16), (16, -16), (True, 16), (16.0, 16)]:
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                validate_dimensions(width, height)


if __name__ == "__main__":
    unittest.main()
