#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from compare_flux_coreml_fixture import error_metrics, validate_capture


class FluxCaptureTest(unittest.TestCase):
    def test_complete_fixture_and_truncated_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tensors = {}
            for name in ("latent", "timesteps", "context", "output"):
                filename = name + ".bin"
                (directory / filename).write_bytes(b"\0" * 4)
                tensors[name] = {"file": filename, "type": "f32",
                                 "ne": [1, 1, 1, 1], "bytes": 4}
            (directory / "manifest.json").write_text(
                json.dumps({"format": 1, "tensors": tensors}), encoding="utf-8")
            self.assertEqual(set(validate_capture(directory)["tensors"]), set(tensors))
            (directory / "output.bin").write_bytes(b"\0" * 3)
            with self.assertRaisesRegex(ValueError, "size mismatch for output"):
                validate_capture(directory)


@unittest.skipIf(np is None, "numpy is required")
class ErrorMetricsTest(unittest.TestCase):
    def test_identical_output_has_zero_error(self):
        reference = np.array([1.0, -2.0], dtype=np.float32)
        result = error_metrics(reference.copy(), reference, np)
        self.assertEqual(result["normalized_rmse"], 0.0)
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "output shape"):
            error_metrics(np.zeros((1, 2), dtype=np.float32),
                          np.zeros((2, 1), dtype=np.float32), np)

    def test_zero_energy_reference_fails(self):
        with self.assertRaisesRegex(ValueError, "zero energy"):
            error_metrics(np.ones(2, dtype=np.float32),
                          np.zeros(2, dtype=np.float32), np)

    def test_nonfinite_output_fails(self):
        with self.assertRaisesRegex(ValueError, "finite values"):
            error_metrics(np.array([float("nan")], dtype=np.float32),
                          np.array([1.0], dtype=np.float32), np)


if __name__ == "__main__":
    unittest.main()
