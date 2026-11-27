"""Checks the fixed Core ML interface before loading a multi-gigabyte model."""

import unittest

from export_flux2_klein_coreml import DEFAULT_CUTS, chunk_bounds, chunk_io, chunk_weight_filter, fixture_shapes


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


class ChunkTests(unittest.TestCase):
    def test_default_cuts(self):
        self.assertEqual(chunk_bounds(DEFAULT_CUTS), [(0, 3), (3, 6), (6, 13), (13, 20), (20, 25)])

    def test_rejects_bad_cuts(self):
        for cuts in ((3, 3), (0,), (25,)):
            with self.assertRaises(ValueError):
                chunk_bounds(cuts)

    def test_chunks_chain_by_name(self):
        available = {"latent", "timesteps", "context"}
        for lo, hi in chunk_bounds(DEFAULT_CUTS):
            inputs, outputs = chunk_io(lo, hi)
            self.assertTrue(set(inputs) <= available, (lo, hi, inputs))
            available |= set(outputs)
        self.assertEqual(chunk_io(20, 25)[1], ["output"])
        self.assertEqual(chunk_io(0, 3)[1], ["img_3", "txt_3", "vec"])
        self.assertEqual(chunk_io(3, 6), (["img_3", "txt_3", "vec"], ["hidden_6"]))

    def test_weights_per_chunk(self):
        first, middle, last = chunk_weight_filter(0, 3), chunk_weight_filter(6, 13), chunk_weight_filter(20, 25)
        self.assertTrue(first("txt_in.weight") and first("double_blocks.2.img_mlp.0.weight"))
        self.assertFalse(first("double_blocks.3.img_mlp.0.weight") or first("final_layer.linear.weight"))
        self.assertTrue(middle("single_blocks.1.linear1.weight") and middle("single_stream_modulation.lin.weight"))
        self.assertFalse(middle("single_blocks.8.linear1.weight") or middle("txt_in.weight")
                         or middle("double_stream_modulation_img.lin.weight"))
        self.assertTrue(last("final_layer.linear.weight") and last("single_blocks.19.linear2.weight"))


if __name__ == "__main__":
    unittest.main()
