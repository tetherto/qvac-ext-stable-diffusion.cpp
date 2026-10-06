## Use Flash Attention to save memory and improve speed.

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB.
eg.:
 - flux 768x768 ~600mb
 - SD2 768x768 ~1400mb

For most backends, it slows things down, but for cuda it generally speeds it up too.
At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).

Run by adding `--diffusion-fa` to the arguments and watch for:
```
[INFO ] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```
and the compute buffer shrink in the debug log:
```
[DEBUG] ggml_extend.hpp:1004 - flux compute buffer size: 650.00 MB(VRAM)
```

## Offload weights to the CPU to save VRAM without reducing generation speed.

Using `--offload-to-cpu` allows you to offload weights to the CPU, saving VRAM without reducing generation speed.

## Use quantization to reduce memory usage.

[quantization](./quantization_and_gguf.md)

## FLUX.2-klein 4B Core ML validation

The engine has an optional Core ML denoiser runtime on macOS. It accepts a
compiled `.mlmodelc` sidecar through `SDCPP_FLUX2_COREML_MODEL`. The sidecar
must contain the complete FLUX.2-klein denoiser. Without the variable, the
existing GGML path remains in use.

Build the runtime alongside Metal:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON -DSD_COREML=ON
cmake --build build --config Release -j 8
```

The current sidecar interface requires fixed-shape, contiguous float32 Core ML
multiarrays named `latent`, `timesteps`, and `context`. It may additionally
accept `pooled` and `guidance` when present. Its float32 output must be named
`output` and match the latent shape. Core ML dimensions reverse GGML's `ne`
order; for example, GGML `[W,H,C,N]` maps to Core ML `[N,C,H,W]`. The runtime
supports text-to-image without reference latents, skipped layers, or LoRA.
Invalid sidecars and unsupported inputs fail instead of falling back to GGML.
When Core ML is selected, the engine does not allocate or load GGML diffusion
weights, avoiding a second denoiser copy in memory.

The 16 GiB Apple M4 FLUX.1-schnell Q4_0 baseline spent 155.18 seconds in four
denoiser calls and 178.18 seconds on generation. Its 6.23 GiB GGUF would
require 22.15 GiB for dense FP16 weights alone, so FLUX.2-klein 4B is the
first complete-denoiser target on that machine.

Download the FLUX.2-klein 4B model, VAE, and Qwen3 4B text encoder described
in [FLUX.2 setup](flux2.md). Record a GGML Metal baseline and capture its
first denoiser invocation with the same prompt, seed, dimensions, and options
that will be used for Core ML:

```sh
SDCPP_FLUX_CAPTURE_DIR=bench-results/flux2-klein-4b-1024-fixture \
python3 script/bench_coreml_baseline.py \
  --binary build/bin/sd-cli \
  --output-dir bench-results/flux2-klein-4b-1024-metal \
  --warmup 0 --runs 1 --require-flux-profile -- \
  --diffusion-model models/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --vae models/flux2-klein-4b/split_files/vae/flux2-vae.safetensors \
  --llm models/flux2-klein-4b/Qwen3-4B-Q4_K_M.gguf \
  -p 'a lovely cat' --seed 42 -W 1024 -H 1024 \
  --steps 4 --sampling-method euler --cfg-scale 1 \
  --diffusion-fa --clip-on-cpu --offload-to-cpu
cat bench-results/flux2-klein-4b-1024-fixture/manifest.json
```

The benchmark writes a log, image, and phase timings in `report.json`. The
fixture contains exact float32 inputs and GGML output for numerical comparison.
On a 16 GiB M4 at 1024 × 1024 with four steps, one measured FLUX.2-klein 4B
run took 154.99 seconds for generation and 132.36 seconds in four denoiser
calls. This single run establishes the optimization target, not a speedup.

The exporter imports the official
[Black Forest Labs FLUX.2 implementation](https://github.com/black-forest-labs/flux2)
from a separate source checkout. Use a separate Python environment with
`torch`, `safetensors`, `einops`, `numpy`, and `coremltools`. A previously tested
combination is PyTorch 2.7 and coremltools 9. The model checkpoint is about
7.2 GiB, and tracing or conversion may exceed 16 GiB of memory. A Mac with
more RAM is preferable for export; transfer the resulting `.mlpackage` to the
M4 and compile it there if needed. Keep the captured fixture and baseline on
the M4. The command examples below use a fresh output location and should be
run from the repository root.

First validate that the official PyTorch model reproduces the GGML call. This
is a required gate before exporting. It is a full denoiser invocation and can
take several minutes on CPU. The command prints separate checkpoint loading and
denoiser timing messages. Add `--device mps` to the `check` or `trace` command
if PyTorch MPS is available and CPU inference is too slow:

```sh
git clone --depth 1 https://github.com/black-forest-labs/flux2.git ../flux2-official
python3 script/export_flux2_klein_coreml.py check \
  --source ../flux2-official \
  --weights models/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --fixture bench-results/flux2-klein-4b-1024-fixture
```

The check defaults to BF16, matching the checkpoint's precision. Repeat it
with `--check-dtype fp16` before tracing; the traced model uses FP16 for Core ML.
Both checks must meet the 0.03 normalized RMSE gate. A single-call numerical
comparison does not establish full image quality.

If both checks pass, trace, convert, and compile in separate
processes. Each output path must be new. The model has fixed dimensions from
the fixture; changing image size or text length requires another export.

```sh
python3 script/export_flux2_klein_coreml.py trace \
  --source ../flux2-official \
  --weights models/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --fixture bench-results/flux2-klein-4b-1024-fixture \
  --output bench-results/flux2-klein-4b-1024.pt
python3 script/export_flux2_klein_coreml.py convert \
  --trace bench-results/flux2-klein-4b-1024.pt \
  --fixture bench-results/flux2-klein-4b-1024-fixture \
  --output bench-results/flux2-klein-4b-1024.mlpackage
python3 script/export_flux2_klein_coreml.py compile \
  --package bench-results/flux2-klein-4b-1024.mlpackage \
  --output bench-results/flux2-klein-4b-1024.mlmodelc
```

After compilation, verify one Core ML call against the fixture before running
image generation:

```sh
python3 script/compare_flux_coreml_fixture.py \
  --model bench-results/flux2-klein-4b-1024.mlmodelc \
  --fixture bench-results/flux2-klein-4b-1024-fixture \
  --output bench-results/flux2-klein-4b-coreml-parity.json
```

A passing single-call comparison is followed by a full generation using
`SDCPP_FLUX2_COREML_MODEL=/absolute/path/to/flux2-klein-4b.mlmodelc` with the
same CLI options and a fresh output directory. Compare denoiser time,
generation time, peak memory, and the image against the Metal baseline.
