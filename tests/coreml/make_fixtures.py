"""Builds the tiny CoreML models that tests/coreml/test_coreml.cpp runs.

They have the I/O of the real sidecars but a few ops each, with outputs test_coreml.cpp computes exactly:
  unet:  noise_pred[b] = 0.5 * sample[b] + sum_{t,d} ehs[b, t, d] * W[t, d] + 0.001 * timestep[b]
         with W[t, d] = 0.1 * (t + 1) + 0.01 * (d + 1), so a wrong context layout changes the result
         SDXL-style models add sum_a y[b, a] * V[a] with V[a] = 0.02 * (a + 1), where y is either an input (A values)
         or text_embeds (P values) followed by time_ids (K values); time_ids use V[k] = 0.001 * (k + 1)
         unet_chunks/ computes the same as unet_b1 in two chunks: unet_chunk1 returns 0.5 * sample + 0.001 * timestep,
         unet_chunk2 takes that and encoder_hidden_states
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
A = 7  # y values for the SDXL-style fixture with a y input
P, K = 3, 2  # text_embeds and time_ids values for the Apple-style SDXL fixture


def weights(channels_first):
    w = 0.1 * (np.arange(T)[:, None] + 1) + 0.01 * (np.arange(D)[None, :] + 1)  # [T, D]
    return (w.T[None, :, None, :] if channels_first else w[None]).astype(np.float32)  # [1, D, 1, T] or [1, T, D]


def unet(batch, dtype=types.fp32, channels_first=False, prediction=None, vector=None):
    """vector: None, "y" (a y input) or "text_time" (text_embeds and time_ids inputs)."""
    ctx_shape = (batch, D, 1, T) if channels_first else (batch, T, D)
    specs = [mb.TensorSpec((batch, C, H, W), dtype), mb.TensorSpec((batch,), dtype), mb.TensorSpec(ctx_shape, dtype)]
    if vector == "y":
        specs.append(mb.TensorSpec((batch, A), dtype))
    elif vector == "text_time":
        specs += [mb.TensorSpec((batch, P), dtype), mb.TensorSpec((batch, K), dtype)]

    def body(sample, timestep, encoder_hidden_states, *vectors):
        if dtype == types.fp16:
            sample = mb.cast(x=sample, dtype="fp32")
            timestep = mb.cast(x=timestep, dtype="fp32")
            encoder_hidden_states = mb.cast(x=encoder_hidden_states, dtype="fp32")
            vectors = [mb.cast(x=v, dtype="fp32") for v in vectors]
        c = mb.reduce_sum(x=mb.mul(x=encoder_hidden_states, y=weights(channels_first)),
                          axes=[1, 2, 3] if channels_first else [1, 2], keep_dims=False)
        bias = mb.add(x=c, y=mb.mul(x=timestep, y=np.float32(0.001)))
        scales = {"y": [0.02 * (np.arange(A) + 1)], "text_time": [0.02 * (np.arange(P) + 1), 0.001 * (np.arange(K) + 1)]}
        for v, scale in zip(vectors, scales.get(vector, [])):
            term = mb.reduce_sum(x=mb.mul(x=v, y=scale[None].astype(np.float32)), axes=[1], keep_dims=False)
            bias = mb.add(x=bias, y=term)
        bias = mb.reshape(x=bias, shape=(batch, 1, 1, 1))
        return mb.add(x=mb.mul(x=sample, y=np.float32(0.5)), y=bias, name="noise_pred")

    # mb.program names the inputs after the function's parameters.
    if vector == "y":
        def prog(sample, timestep, encoder_hidden_states, y):
            return body(sample, timestep, encoder_hidden_states, y)
    elif vector == "text_time":
        def prog(sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            return body(sample, timestep, encoder_hidden_states, text_embeds, time_ids)
    else:
        def prog(sample, timestep, encoder_hidden_states):
            return body(sample, timestep, encoder_hidden_states)

    model = ct.convert(mb.program(input_specs=specs, opset_version=ct.target.iOS17)(prog), convert_to="mlprogram",
                       compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS17)
    if prediction:
        model.user_defined_metadata["prediction_type"] = prediction
    return model


def unet_chunks():
    """unet_b1's function in two chunks: the second takes the first's output and a UNet input."""

    @mb.program(input_specs=[mb.TensorSpec((1, C, H, W), types.fp32), mb.TensorSpec((1,), types.fp32)],
                opset_version=ct.target.iOS17)
    def chunk1(sample, timestep):
        t = mb.reshape(x=mb.mul(x=timestep, y=np.float32(0.001)), shape=(1, 1, 1, 1))
        return mb.add(x=mb.mul(x=sample, y=np.float32(0.5)), y=t, name="hidden")

    @mb.program(input_specs=[mb.TensorSpec((1, C, H, W), types.fp32), mb.TensorSpec((1, T, D), types.fp32)],
                opset_version=ct.target.iOS17)
    def chunk2(hidden, encoder_hidden_states):
        c = mb.reduce_sum(x=mb.mul(x=encoder_hidden_states, y=weights(False)), axes=[1, 2], keep_dims=False)
        return mb.add(x=hidden, y=mb.reshape(x=c, shape=(1, 1, 1, 1)), name="noise_pred")

    return [ct.convert(p, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                       minimum_deployment_target=ct.target.iOS17) for p in (chunk1, chunk2)]


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
        "unet_y_b1.mlpackage": unet(1, vector="y"),
        "unet_text_time_b2.mlpackage": unet(2, dtype=types.fp16, channels_first=True, vector="text_time"),
        "tae.mlpackage": tae(8),
        "tae_wrong_scale.mlpackage": tae(4),  # [1, 4, 8, 8] -> [1, 3, 32, 32]: not a tiny decoder's 8x
    }
    for i, model in enumerate(unet_chunks()):
        fixtures[f"unet_chunks/unet_chunk{i + 1}.mlpackage"] = model
    for name, model in fixtures.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out / name))
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
