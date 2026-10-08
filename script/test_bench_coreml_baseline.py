#!/usr/bin/env python3
"""Model-free checks for the benchmark's result validation."""

import unittest

from bench_coreml_baseline import parse_timings


LOG = """[DEBUG] Using Metal backend
[INFO] get_learned_condition completed, taking 125 ms
[INFO] sampling completed, taking 9.50s
[INFO] decode_first_stage completed, taking 1.25s
[INFO] generate_image completed in 11.00s
"""


class ParseTimingsTest(unittest.TestCase):
    def test_extracts_all_phases(self):
        result = parse_timings(LOG)
        self.assertEqual(result["conditioning_s"], 0.125)
        self.assertEqual(result["sampling_s"], 9.5)
        self.assertEqual(result["vae_decode_s"], 1.25)
        self.assertEqual(result["generation_s"], 11.0)
        self.assertAlmostEqual(result["vae_share_pct"], 100 * 1.25 / 11)

    def test_runner_api_log_format(self):
        log = LOG.replace("Using Metal backend", "Diffusion model runtime backend: Metal")
        log = log.replace("125 ms", "0.125s")
        self.assertEqual(parse_timings(log)["conditioning_s"], 0.125)
        with self.assertRaisesRegex(ValueError, "VAE ran on CPU"):
            parse_timings(log + "VAE runtime backend: CPU\n")
        with self.assertRaisesRegex(ValueError, "VAE ran on CPU"):
            parse_timings(log + "vae VAE CPU fallback complete; restored runtime backend Metal\n")

    def test_rejects_cpu_or_missing_measurement(self):
        with self.assertRaisesRegex(ValueError, "Metal backend"):
            parse_timings(LOG.replace("Using Metal backend", "Using CPU backend"))
        with self.assertRaisesRegex(ValueError, "VAE ran on CPU"):
            parse_timings(LOG + "VAE Autoencoder: Using CPU backend\n")
        with self.assertRaisesRegex(ValueError, "vae_decode_s"):
            parse_timings(LOG.replace("decode_first_stage", "other_stage"))

    def test_sums_flux_model_calls_inside_sampling(self):
        log = LOG.replace("9.50s", "9.50s\nflux denoiser compute completed, taking 4000 ms\n"
                          "flux denoiser compute completed, taking 3000 ms")
        result = parse_timings(log)
        self.assertEqual(result["flux_denoiser_calls"], 2)
        self.assertEqual(result["flux_denoiser_s"], 7.0)
        self.assertEqual(result["sampling_other_s"], 2.5)
        with self.assertRaisesRegex(ValueError, "exceeds total sampling"):
            parse_timings(log.replace("4000 ms", "8000 ms"))


if __name__ == "__main__":
    unittest.main()
