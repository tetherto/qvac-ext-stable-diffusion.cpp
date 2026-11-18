"""Parity tests for the ANE rewrites in convert_unet.py against the original diffusers implementations.

Small random UNets and attention layers, float32 on the CPU, no model download; about 10 s:
  python -m unittest script/coreml/test_convert_unet.py

pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4
"""

import sys
import unittest
from pathlib import Path

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

    def test_silu_rewrite(self):
        x = torch.linspace(-20, 20, 4001)
        self.assertLess(rel_err(convert_unet.silu_via_tanh(None, x), torch.nn.functional.silu(x)), 1e-6)

    def test_unsupported_attention_is_rejected(self):
        attn = Attention(query_dim=64, heads=4, dim_head=16)
        with self.assertRaises(ValueError):
            convert_unet.SplitEinsumAttention()(attn, torch.randn(1, 64, 4, 4))


if __name__ == "__main__":
    unittest.main()
