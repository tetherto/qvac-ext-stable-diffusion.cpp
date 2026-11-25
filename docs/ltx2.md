# LTX-2 (LTX-2.3) video generation

Support for [Lightricks LTX-2](https://huggingface.co/Lightricks) text-to-video
(T2V) and image-to-video (I2V) generation. Only the **video** stream is in
scope; the audio stream (audio DiT, audio VAE, vocoder) is intentionally not
converted or loaded.

> Status: **work in progress.** Milestone M1 (conversion + scaffolding + "loads
> on CPU") and Milestone M2 (core CPU inference) are implemented: the full
> T2V/I2V pipeline — Gemma-3 text encoder, video DiT denoising loop, the
> LinearQuadratic scheduler and the Video-VAE — runs end-to-end on CPU and
> writes a video file. See **Validation status** below for which pieces are
> numerically validated vs. structurally implemented pending reference parity.

## What works today (M1 + M2)

- Safetensors -> GGUF conversion tooling for the video-only DiT (+ VAE), at
  `f16`, `q8_0`, `q5_1`, `q4_0`.
- LTX-2 architecture auto-detection in the model loader.
- The 14B video DiT (`AVTransformer3DModel`, video half) loads on CPU and runs a
  full forward pass: `patchify_proj`, 3D RoPE, AdaLN-single modulation, gated
  self/cross attention with RMS qk-norm, the gelu FFN, the video-embeddings
  connector and `proj_out`. Geometry is inferred from the checkpoint shapes.
- A native **Gemma-3** text encoder (`src/gemma3.hpp`): GeGLU MLP, q/k RMSNorm,
  `(1 + weight)` RMSNorm, scaled token embeddings, and sliding-window vs global
  attention layers. The multi-layer feature extractor + projection feed the DiT
  cross-attention.
- A **LinearQuadratic** flow-matching scheduler (`--scheduler linear_quadratic`).
- The LTX **CausalVideoAutoencoder** geometry (32x spatial / 8x temporal, 128
  latent channels) and the T2V + I2V `generate_video` wiring, producing MJPEG
  AVI / raw-frame output from the CLI.
- A CI smoke test that runs a full synthetic end-to-end generate on both
  Linux x86-64 (AVX) and macOS ARM64 (NEON) without any large download.

## Usage

LTX-2 needs the converted LTX checkpoint (DiT + VAE + text projection) plus a
Gemma-3 GGUF for the text encoder:

```bash
# Text-to-video
./build/bin/sd-cli -M vid_gen \
    --diffusion-model ltx-2.3-video-q8_0.gguf \
    --llm gemma-3-12b.gguf \
    -p "a corgi running on the beach at sunset" \
    --scheduler linear_quadratic --steps 8 \
    -W 512 -H 768 --video-frames 49 -o out

# Image-to-video (animate / bootstrap from a reference frame)
./build/bin/sd-cli -M vid_gen \
    --diffusion-model ltx-2.3-video-q8_0.gguf --llm gemma-3-12b.gguf \
    -i first_frame.png -p "the camera slowly pans right" \
    --scheduler linear_quadratic --steps 8 \
    -W 512 -H 768 --video-frames 49 -o out
```

LTX-2 targets the **distilled** checkpoint first (8 steps, CFG=1), so no
negative-prompt pass is needed. Frame counts are aligned to `(F-1) % 8 == 0` and
spatial dimensions to multiples of 32. Memory is staged per component (encode
text, then run the DiT, then decode) — use `--diffusion-fa` / quantized weights
to fit consumer RAM.

## Validation status (read before trusting output)

M2's goal is end-to-end CPU inference that runs and produces video of the right
shape; numerical parity (PSNR/SSIM) is an M3 acceptance criterion. The following
pieces are implemented structurally and **must be validated against the
Diffusers LTX-2 reference** before claiming quality parity:

- the exact assignment of the 9 per-block + 2 prompt AdaLN modulation channels,
  the 3D RoPE axis split and `theta`, and the non-affine norm placement
  (`src/ltx2.hpp`);
- the Gemma-3 RoPE theta / sliding-window pattern / query scaling, and a real
  Gemma SentencePiece tokenizer (the current tokenizer is a byte-level
  placeholder; `src/gemma3.hpp`, `Ltx2Conditioner`);
- the LTX multi-layer feature-extractor aggregation and the
  `text_embedding_projection.video_*` tensor layout (`Ltx2TextProjection`);
- the **Video-VAE** is currently a shape-correct geometric placeholder
  (`Ltx2VAERunner` in `src/vae.hpp`); the learned causal-conv encoder/decoder,
  PixelNorm and per-channel statistics replace it next.

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

## Verifying the full pipeline without the real weights

A tiny synthetic checkpoint generator exercises the entire pipeline in seconds.
It emits the exact DiT + Gemma-3 + projection tensor names at drastically
reduced dimensions (and a separate Gemma file), so the same code path that loads
the real weights runs a full generate.

```bash
# requires numpy + gguf (see script/requirements-ltx2.txt)
python script/make_synthetic_ltx2_gguf.py --out /tmp/ltx2_tiny.gguf
# writes /tmp/ltx2_tiny.gguf (DiT + projection) and /tmp/ltx2_tiny.gguf.gemma.gguf

bash script/ci_ltx2_load_smoke.sh            # build + synthetic end-to-end generate
```

Expected: the log reports `Version: LTX-2`, the inferred Gemma-3 and DiT
geometry, `get_sigmas with LinearQuadratic scheduler`, `sampling completed`,
`decode_first_stage completed`, and writes an `.avi` video.

## CI

`.github/workflows/ltx2.yml` runs on Linux x86-64 (AVX) and macOS ARM64 (NEON):

1. validates the conversion filter (`--self-test`),
2. builds `sd-cli`,
3. runs the synthetic end-to-end generate smoke test.

## Scope

In scope: video DiT, Video-VAE encoder+decoder, Gemma-3 text encoder, a noise
scheduler + CFG, T2V and I2V, GGUF conversion, CLI, C API.

Out of scope: the audio stream (audio DiT, Audio-VAE, vocoder), training /
fine-tuning, the spatial upscaler, and video-to-video.
