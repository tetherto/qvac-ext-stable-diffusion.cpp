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

## Use params backend to reduce VRAM or RAM usage.

`--params-backend` controls where model parameters are kept. If it is not set, parameters use the same backend as `--backend`, so a GPU runtime backend also keeps parameters in VRAM.

Use CPU params to reduce VRAM usage:

```shell
--backend cuda0 --params-backend cpu
```

This keeps model weights in system RAM and moves them to the runtime backend when needed. In the example CLI/server, `--offload-to-cpu` is a compatibility shortcut that prepends `*=cpu` to `--params-backend` before creating the context, so explicit module assignments can still override it:

```shell
--offload-to-cpu --params-backend te=disk
```

Use disk params to reduce both VRAM and RAM usage:

```shell
--backend cuda0 --params-backend disk
```

This reloads parameters from the model file on demand and releases them after use. It has the lowest memory residency, but can be slower because weights must be read again. `disk` is never selected implicitly; set it explicitly when RAM usage matters more than reload cost.

Per-module assignments can target only the largest modules:

```shell
--backend cuda0 --params-backend diffusion=disk,te=cpu,vae=cpu
```

See [backend selection](./backend.md) for full syntax.

## Run models that don't fit in VRAM (CPU streaming).

`--offload-to-cpu` alone keeps every parameter in system RAM and stages it to the runtime backend on first use, then leaves it resident there. If the diffusion model is larger than the runtime backend's free memory (e.g. Flux dev at bf16 on an 8 GiB GPU), that residency stops fitting during the sampling loop and generation fails. Two additional flags make it fit by trading a small amount of speed for room:

- `--max-vram <GiB>` sets a VRAM budget the graph-cut segmenter respects. It cuts each forward pass into segments sized to fit the budget, running them in sequence and freeing intermediate activations between them. Negative values auto-detect free VRAM and spare the given amount (`--max-vram -1` uses most of the free VRAM and keeps ~1 GiB headroom), a positive value caps the budget, `0` disables segmentation.
- `--stream-layers` streams the diffusion model's transformer blocks one at a time. Each block's parameters are copied from the CPU to the runtime backend just before it runs and evicted when the residency budget is reached. Prefetching hides most of the copy latency behind compute. This flag only takes effect when the diffusion params backend is CPU, so it must be combined with `--offload-to-cpu` (or an explicit `--params-backend diffusion=cpu`); a warning is logged and the flag is ignored otherwise.

The three flags stack. The recommended shape for "biggest model my card can host":

```shell
sd-cli --diffusion-model flux1-dev.safetensors ... \
       --offload-to-cpu --max-vram -1 --stream-layers
```

- `--offload-to-cpu`: params in RAM, staged as needed.
- `--max-vram -1`: use most of the free VRAM as the compute budget, spare 1 GiB headroom, let the graph-cut segmenter split each forward pass to fit.
- `--stream-layers`: on top of the segmenter, stream individual transformer blocks so their weights don't all need to be resident at once.

Ordered from fastest to smallest-VRAM: no flags → `--offload-to-cpu` → `--offload-to-cpu --max-vram <N>` → `--offload-to-cpu --max-vram <N> --stream-layers`. Each step down costs a few percent of throughput to buy more room; combined they can run models roughly 3-4x larger than the raw VRAM would allow.

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
For a multi-step quality investigation, set `SDCPP_FLUX_CAPTURE_ALL=1` alongside
`SDCPP_FLUX_CAPTURE_DIR`; each denoiser call is then captured under `call-1`,
`call-2`, and so on. Use a new capture directory for each run.
On a 16 GiB M4 at 1024 × 1024 with four steps, one measured FLUX.2-klein 4B
run took 154.99 seconds for generation and 132.36 seconds in four denoiser
calls. This single run establishes the optimization target, not a speedup.

The exporter imports the official
[Black Forest Labs FLUX.2 implementation](https://github.com/black-forest-labs/flux2)
from a separate source checkout. Use a separate Python environment with the
versions in `script/requirements-flux2-coreml.txt`. PyTorch 2.7 is the latest
version tested by coremltools 9, and NumPy is kept below 2.4 for conversion
compatibility. The model checkpoint is about
7.2 GiB, and tracing or conversion may exceed 16 GiB of memory. A Mac with
more RAM is preferable for export; transfer the resulting `.mlpackage` to the
M4 and compile it there if needed. Keep the captured fixture and baseline on
the M4. The command examples below use a fresh output location and should be
run from the repository root.

```sh
python3 -m venv .venv-flux2-export
source .venv-flux2-export/bin/activate
python -m pip install -r script/requirements-flux2-coreml.txt
```

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
comparison does not establish full image quality. Also check a later denoiser
call with `--finite-only` before tracing. This checks the official PyTorch
model without using the captured output as a reference:

```sh
python3 script/export_flux2_klein_coreml.py check \
  --source ../flux2-official \
  --weights models/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --fixture bench-results/flux2-klein-v4-failure-capture/call-2 \
  --check-dtype fp16 --finite-only
```

On the 16 GiB M4, this captured second call returned 0 finite values out of
524,288 in the official FP16 PyTorch model. The FP16 Core ML conversion also
returned 0 finite values on the same call. The failure therefore precedes
Core ML conversion; the current FP16 export is unsuitable for multi-step image
generation. Do not treat its first-call parity result or measured runtime as
a working speedup. A BF16 or numerically stable mixed-precision export is
needed before further end-to-end benchmarking. The same second call returned
524,288 finite values in official BF16 PyTorch. To find where FP16 first
becomes non-finite, repeat the failing check with `--locate-nonfinite`. It
stops at the first affected module or residual operation and reports its input
and output ranges, without tracing or writing another model:

```sh
python3 script/export_flux2_klein_coreml.py check \
  --source ../flux2-official \
  --weights models/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --fixture bench-results/flux2-klein-v4-failure-capture/call-2 \
  --check-dtype fp16 --finite-only --locate-nonfinite
```

The first observed FP16 overflow is the text MLP gate multiplication in
double block 4. Saturating these residuals to the finite FP16 range kept the
second call finite but missed the GGML parity gate (0.0652 normalized RMSE).
`--safe-fp16` now keeps the residual streams in float32 while retaining FP16
linear weights and casting normalized inputs at each linear layer. Check the
captured later call against a GGML capture before tracing or benchmarking;
finite output alone does not establish acceptable numerical parity or image
quality:

```sh
python3 script/export_flux2_klein_coreml.py check \
  --source ../flux2-official \
  --weights models/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --fixture bench-results/flux2-klein-4b-ggml-all-reference/call-2 \
  --check-dtype fp16 --safe-fp16
```

Only after all checks pass, trace, convert, and compile in separate
processes. Each output path must be new. The model has fixed dimensions from
the fixture; changing image size or text length requires another export. The
trace step runs the traced graph against the fixture and only saves it if that
comparison passes. It uses paired cosine and sine values for RoPE so no
intermediate tensor exceeds Core ML's rank-5 limit, and retains fractional
timesteps in float32 through their sinusoidal embedding. Older traces must be
regenerated. Conversion preserves the traced model's explicit float32 math.
The exporter remains experimental until the multi-step failure is resolved.

```sh
python3 script/export_flux2_klein_coreml.py trace \
  --source ../flux2-official \
  --weights models/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --fixture bench-results/flux2-klein-4b-1024-fixture \
  --safe-fp16 \
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

If a later denoiser call was captured, also test that exact input before a full
generation. This mode checks shape and finite values without comparing with
the captured output, so it can be used when that output contains NaNs:

```sh
python3 script/compare_flux_coreml_fixture.py \
  --model bench-results/flux2-klein-4b-1024.mlmodelc \
  --fixture bench-results/flux2-klein-v4-failure-capture/call-2 \
  --output bench-results/flux2-klein-4b-coreml-step2.json \
  --finite-only
```

After these checks pass, run a full generation using
`SDCPP_FLUX2_COREML_MODEL=/absolute/path/to/flux2-klein-4b.mlmodelc` with the
same CLI options and a fresh output directory. Compare denoiser time,
generation time, peak memory, and the image against the Metal baseline.
