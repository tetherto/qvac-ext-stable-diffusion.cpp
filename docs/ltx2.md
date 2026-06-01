# LTX-2 (LTX-2.3) video generation

Support for [Lightricks LTX-2](https://huggingface.co/Lightricks) text-to-video
(T2V) and image-to-video (I2V) generation. Only the **video** stream is in
scope; the audio stream (audio DiT, audio VAE, vocoder) is intentionally not
converted or loaded.

> Status: **work in progress.** Milestone M1 (model conversion + scaffolding +
> "model loads on CPU") is implemented. End-to-end T2V/I2V inference, the
> Gemma-3 text encoder and the CausalVideoAutoencoder land in later milestones.

## What works today (M1)

- Safetensors -> GGUF conversion tooling for the video-only DiT (+ VAE), at
  `f16`, `q8_0`, `q5_1`, `q4_0`.
- LTX-2 architecture auto-detection in the model loader.
- The 14B video DiT (`AVTransformer3DModel`, video half) loads and binds all of
  its parameters on CPU, with geometry inferred from the checkpoint shapes.
- A CI smoke test that verifies the load path on Linux x86-64 without any large
  download.

## Model conversion

The conversion script reads an LTX-2.3 `.safetensors` checkpoint, drops every
audio/vocoder tensor, and writes a video-only GGUF.

```bash
pip install -r script/requirements-ltx2.txt

# Inspect the keep/drop plan without writing anything (stdlib only):
python script/convert_ltx2_to_gguf.py --src ltx-2.3-22b-dev.safetensors --dry-run

# Convert to F16 (M1 deliverable):
python script/convert_ltx2_to_gguf.py \
    --src ltx-2.3-22b-dev.safetensors \
    --dst ltx-2.3-22b-video-f16.gguf --type f16

# Quantized variants:
python script/convert_ltx2_to_gguf.py --src ... --dst ...-q8_0.gguf --type q8_0
python script/convert_ltx2_to_gguf.py --src ... --dst ...-q4_0.gguf --type q4_0
```

Quantization notes:

- `f16` keeps every tensor in half precision (highest quality, largest file).
- `q8_0` / `q5_1` / `q4_0` quantize only the large 2D DiT linear weights; norms,
  biases, modulation tables and the VAE stay in higher precision.

## Building

Follow the standard [build guide](./build.md). The CLI binary is `sd-cli`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target sd-cli
```

## Verifying "loads on CPU" without the full weights

M1 ships a tiny synthetic checkpoint generator so the load path can be exercised
in seconds. It emits the exact DiT tensor names at drastically reduced
dimensions; the C++ side infers the geometry from the shapes, so this is the
same code path used for the real weights.

```bash
# requires numpy + gguf (see script/requirements-ltx2.txt)
bash script/ci_ltx2_load_smoke.sh            # uses build/bin/sd-cli
```

Expected: the log reports `Version: LTX-2`, the inferred DiT geometry, and
`loading tensors completed`, then exits early (generation is not available yet
in M1).

## CI

`.github/workflows/ltx2.yml` runs on Linux x86-64:

1. validates the conversion filter (`--self-test`),
2. builds `sd-cli`,
3. runs the synthetic load-on-CPU smoke test.

## Scope

In scope: video DiT, Video-VAE encoder+decoder, Gemma-3 text encoder, a noise
scheduler + CFG, T2V and I2V, GGUF conversion, CLI, C API.

Out of scope: the audio stream (audio DiT, Audio-VAE, vocoder), training /
fine-tuning, the spatial upscaler, and video-to-video.
