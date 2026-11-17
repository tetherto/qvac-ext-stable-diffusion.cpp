"""Builds the tiny CoreML models that tests/coreml/test_coreml.cpp runs.

They have the I/O of the real sidecars but a few ops each, with outputs test_coreml.cpp computes exactly:
  unet:  noise_pred[b] = 0.5 * sample[b] + sum_{t,d} ehs[b, t, d] * W[t, d] + 0.001 * timestep[b]
         with W[t, d] = 0.1 * (t + 1) + 0.01 * (d + 1), so a wrong context layout changes the result
  tae:   image = nearest 8x upsampling of latent channels 0..2

  python tests/coreml/make_fixtures.py OUT_DIR

pip install coremltools==8.3.0 numpy==1.26.4 (no torch needed)
"""

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

C, H, W = 4, 8, 8  # latent
T, D = 5, 6  # text tokens, embedding width; T != D so a transposed context gives a different result


def weights(channels_first):
    w = 0.1 * (np.arange(T)[:, None] + 1) + 0.01 * (np.arange(D)[None, :] + 1)  # [T, D]
    return (w.T[None, :, None, :] if channels_first else w[None]).astype(np.float32)  # [1, D, 1, T] or [1, T, D]


def unet(batch, dtype=types.fp32, channels_first=False, prediction=None):
    ctx_shape = (batch, D, 1, T) if channels_first else (batch, T, D)

    @mb.program(input_specs=[mb.TensorSpec((batch, C, H, W), dtype), mb.TensorSpec((batch,), dtype),
                             mb.TensorSpec(ctx_shape, dtype)],
                 opset_version=ct.target.iOS17)
    def prog(sample, timestep, encoder_hidden_states):
        if dtype == types.fp16:
            sample = mb.cast(x=sample, dtype="fp32")
            timestep = mb.cast(x=timestep, dtype="fp32")
            encoder_hidden_states = mb.cast(x=encoder_hidden_states, dtype="fp32")
        c = mb.reduce_sum(x=mb.mul(x=encoder_hidden_states, y=weights(channels_first)),
                          axes=[1, 2, 3] if channels_first else [1, 2], keep_dims=False)
        bias = mb.add(x=c, y=mb.mul(x=timestep, y=np.float32(0.001)))
        bias = mb.reshape(x=bias, shape=(batch, 1, 1, 1))
        return mb.add(x=mb.mul(x=sample, y=np.float32(0.5)), y=bias, name="noise_pred")

    model = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                       minimum_deployment_target=ct.target.iOS17)
    if prediction:
        model.user_defined_metadata["prediction_type"] = prediction
    return model


def tae(scale):
    @mb.program(input_specs=[mb.TensorSpec((1, C, H, W), types.fp32)], opset_version=ct.target.iOS17)
    def prog(latent):
        rgb = mb.slice_by_index(x=latent, begin=[0, 0, 0, 0], end=[1, 3, H, W])
        return mb.upsample_nearest_neighbor(x=rgb, scale_factor_height=scale, scale_factor_width=scale, name="image")

    return ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                      minimum_deployment_target=ct.target.iOS17)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "unet_b1.mlpackage": unet(1),
        "unet_b1_fp16.mlpackage": unet(1, dtype=types.fp16),
        "unet_b1_channels_first.mlpackage": unet(1, channels_first=True),
        "unet_b2_vpred.mlpackage": unet(2, prediction="v_prediction"),
        "tae.mlpackage": tae(8),
        "tae_wrong_scale.mlpackage": tae(4),  # [1, 4, 8, 8] -> [1, 3, 32, 32]: not a tiny decoder's 8x
    }
    for name, model in fixtures.items():
        model.save(str(out / name))
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
