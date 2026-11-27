"""Parity tests for the ANE rewrites in convert_unet.py against the original diffusers implementations.

Small random UNets and attention layers, float32 on the CPU, no model download; about 10 s:
  python -m unittest script/coreml/test_convert_unet.py

pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4 scikit-learn==1.5.2
"""

import sys
import unittest
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_unet  # noqa: E402

TOL = convert_unet.REWRITE_TOLERANCE


def tiny_unet(sd2):
    """A three-level SD-shaped UNet: SD 1.x conv projections, or SD 2.x linear projections and per-level heads."""
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=8,
        block_out_channels=(32, 64, 64),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=1,
        cross_attention_dim=48,
        attention_head_dim=(2, 4, 4) if sd2 else 8,
        use_linear_projection=sd2,
        norm_num_groups=8,
    ).eval()


def tiny_sdxl_unet():
    """An SDXL-shaped UNet: linear projections, two transformer layers at the lowest level, and text_time
    conditioning (16 pooled values + 6 ids embedded into 8 values each)."""
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=8,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        layers_per_block=1,
        transformer_layers_per_block=(1, 1, 2),
        cross_attention_dim=48,
        attention_head_dim=(2, 4, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=16 + 6 * 8,
        norm_num_groups=8,
    ).eval()


PALETTIZED_OP = "constexpr_lut_to_dense"
CONV_SHAPE = (1, 48, 8, 8)


def converted_conv():
    """One 3x3 convolution whose weight is large enough to be palettized, as an fp16 CoreML program."""
    torch.manual_seed(4)
    conv = torch.nn.Conv2d(CONV_SHAPE[1], 64, 3).eval()
    assert conv.weight.numel() >= convert_unet.PALETTE_THRESHOLD
    traced = torch.jit.trace(conv, torch.zeros(CONV_SHAPE))
    return ct.convert(traced, convert_to="mlprogram", inputs=[ct.TensorType(name="x", shape=CONV_SHAPE, dtype=np.float32)],
                      compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17,
                      skip_model_load=True)


def palette_sizes(mlmodel):
    operations = mlmodel._mil_program.functions["main"].operations
    return [op.lut.val.size for op in operations if op.op_type == PALETTIZED_OP]


def rel_err(a, b):
    return convert_unet.max_rel_err(a.detach().numpy(), b.detach().numpy())


class AttentionParity(unittest.TestCase):
    def check(self, cross, batch, query_chunk, tokens=40):
        torch.manual_seed(1)
        attn = Attention(query_dim=64, cross_attention_dim=48 if cross else None, heads=4, dim_head=16).eval()
        x = torch.randn(batch, tokens, 64)
        context = torch.randn(batch, 7, 48) if cross else None
        with torch.no_grad():
            attn.set_processor(AttnProcessor2_0())
            reference = attn(x, encoder_hidden_states=context)
            attn.set_processor(convert_unet.SplitEinsumAttention(query_chunk))
            out = attn(x, encoder_hidden_states=context)
        self.assertLess(rel_err(out, reference), TOL)

    def test_self_and_cross_batches_and_chunks(self):
        for cross in (False, True):
            for batch in (1, 2):
                # 40 tokens: one chunk, an exact split (8 x 5) and a ragged split (16 + 16 + 8).
                for chunk in (512, 8, 16):
                    with self.subTest(cross=cross, batch=batch, chunk=chunk):
                        self.check(cross, batch, chunk)


class UNetParity(unittest.TestCase):
    def check(self, sd2, attention, batch, query_chunk):
        unet = tiny_unet(sd2)
        torch.manual_seed(2)
        sample = torch.randn(batch, 4, 8, 8)
        timestep = torch.full((batch,), 999.0)
        context = torch.randn(batch, 77, 48)
        with torch.no_grad():
            reference = unet(sample, timestep, context).sample
            convert_unet.apply_attention(unet, attention, query_chunk)
            out = unet(sample, timestep, context).sample
        self.assertLess(rel_err(out, reference), TOL)

    def test_rewrites_match_diffusers(self):
        for sd2 in (False, True):
            for attention in ("split_einsum_v2", "ane"):
                for batch in (1, 2):
                    # The 8x8 top level has 64 query tokens: 512 is one chunk, 16 splits it into four.
                    for chunk in (512, 16):
                        with self.subTest(sd2=sd2, attention=attention, batch=batch, chunk=chunk):
                            self.check(sd2, attention, batch, chunk)

    def test_sdxl_embedded_time_ids(self):
        """The export takes y with the ids already embedded; it must match diffusers given the raw ids."""
        for attention in ("standard", "split_einsum_v2", "ane"):
            with self.subTest(attention=attention):
                unet = tiny_sdxl_unet()
                wrapper = convert_unet.UNetWrapper(unet)
                torch.manual_seed(2)
                sample = torch.randn(2, 4, 8, 8)
                timestep = torch.full((2,), 999.0)
                context = torch.randn(2, 77, 48)
                y, added = convert_unet.sdxl_example(unet, 2, 64, 64)
                self.assertEqual(y.shape, (2, convert_unet.adm_dim(unet.config)))
                with torch.no_grad():
                    reference = unet(sample, timestep, context, added_cond_kwargs=added).sample
                    convert_unet.apply_attention(unet, attention, 16)
                    convert_unet.use_embedded_time_ids(unet)
                    out = wrapper(sample, timestep, context, y)
                self.assertLess(rel_err(out, reference), TOL)

    def test_palettize_tensor(self):
        torch.manual_seed(3)
        w = torch.randn(64, 48, 3, 3)
        for nbits in (4, 6):
            with self.subTest(nbits=nbits):
                q = convert_unet.palettize_tensor(w, nbits)
                self.assertEqual(q.shape, w.shape)
                self.assertLessEqual(q.unique().numel(), 2 ** nbits)
        # More bits, less error.
        err4 = (convert_unet.palettize_tensor(w, 4) - w).abs().mean()
        err6 = (convert_unet.palettize_tensor(w, 6) - w).abs().mean()
        self.assertLess(err6, err4 / 2)

    def test_mixed_bits_stays_within_target(self):
        unet = tiny_sdxl_unet()
        names = [n for n, p in unet.named_parameters() if p.numel() >= convert_unet.PALETTE_THRESHOLD]
        sensitivity = {n: 30.0 + i for i, n in enumerate(names)}  # the first layers are the most sensitive
        sizes = {n: p.numel() for n, p in unet.named_parameters() if n in sensitivity}
        bits = convert_unet.mixed_bits(unet, sensitivity, 4.5, 4, 6)
        average = sum(bits[n] * sizes[n] for n in sizes) / sum(sizes.values())
        self.assertLessEqual(average, 4.5)
        self.assertGreater(average, 4.0)
        self.assertEqual(bits[names[0]], 6)

    def test_p8_palettizes_large_weights_to_eight_bits(self):
        self.assertEqual(palette_sizes(converted_conv()), [])
        compressed = convert_unet.compress_weights(converted_conv(), "p8", "kmeans", 1)
        self.assertEqual(palette_sizes(compressed), [2 ** 8])

    def test_fp16_leaves_weights_uncompressed(self):
        self.assertEqual(palette_sizes(convert_unet.compress_weights(converted_conv(), "fp16", "kmeans", 1)), [])

    def test_non_finite_output_names_the_compute_units_to_avoid(self):
        finite = torch.zeros(1, 4, 8, 8).numpy()
        broken = torch.full((1, 4, 8, 8), float("nan")).numpy()
        self.assertIsNone(convert_unet.unusable_units_warning("CPU_AND_GPU", finite))
        self.assertIn("SD_COREML_COMPUTE_UNITS=cpu_gpu", convert_unet.unusable_units_warning("CPU_AND_GPU", broken))
        self.assertIn("SD_COREML_COMPUTE_UNITS=cpu_ne", convert_unet.unusable_units_warning("CPU_AND_NE", broken))

    def test_silu_rewrite(self):
        x = torch.linspace(-20, 20, 4001)
        self.assertLess(rel_err(convert_unet.silu_via_tanh(None, x), torch.nn.functional.silu(x)), 1e-6)

    def test_unsupported_attention_is_rejected(self):
        attn = Attention(query_dim=64, heads=4, dim_head=16)
        with self.assertRaises(ValueError):
            convert_unet.SplitEinsumAttention()(attn, torch.randn(1, 64, 4, 4))


if __name__ == "__main__":
    unittest.main()
