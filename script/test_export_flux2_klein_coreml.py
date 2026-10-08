"""Checks the fixed Core ML interface before loading a multi-gigabyte model."""

import unittest

from export_flux2_klein_coreml import fixture_shapes


def manifest(latent=(64, 64, 128, 1), context=(7680, 512, 1, 1)):
    return {"tensors": {
        "latent": {"ne": latent},
        "timesteps": {"ne": (1, 1, 1, 1)},
        "context": {"ne": context},
        "output": {"ne": latent},
    }}


class FixtureShapeTests(unittest.TestCase):
    def test_captured_klein_shapes(self):
        self.assertEqual(fixture_shapes(manifest()),
                         ((1, 128, 64, 64), (1, 1, 1, 1), (1, 1, 512, 7680)))

    def test_rejects_other_denoiser(self):
        with self.assertRaisesRegex(ValueError, "128 channels"):
            fixture_shapes(manifest(latent=(64, 64, 64, 1)))

    def test_rejects_other_text_encoder(self):
        with self.assertRaisesRegex(ValueError, "7680"):
            fixture_shapes(manifest(context=(4096, 512, 1, 1)))

    def test_rejects_pooled_conditioning(self):
        captured = manifest()
        captured["tensors"]["pooled"] = {"ne": (768, 1, 1, 1)}
        with self.assertRaisesRegex(ValueError, "pooled"):
            fixture_shapes(captured)


if __name__ == "__main__":
    unittest.main()
