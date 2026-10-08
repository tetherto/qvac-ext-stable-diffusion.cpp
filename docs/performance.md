## Use Flash Attention to save memory and improve speed.

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB.
eg.:
 - flux 768x768 ~600mb
 - SD2 768x768 ~1400mb

`--flash-attn on|off|auto` controls flash attention. Its effect on speed depends
on GPU architecture, backend, driver, model, and tensor shapes. Automatic
selection uses backend rules and, for some paths, small probes; it does not
guarantee the fastest end-to-end generation. Compare `on` and `off` with the
same model and generation settings over repeated runs, separating startup/probe
time from steady-state generation time. See [backend selection](backend.md)
for automatic selection rules and per-module overrides.

Watch for:
```
[INFO ] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```
and the compute buffer shrink in the debug log:
```
[DEBUG] ggml_runner.cpp:280 - flux compute buffer size: 650.00 MB(VRAM) on CUDA0 (peak across 1 segment)
```
This reports the actual peak compute workspace capacity per backend, including
CPU fallback. It excludes weights and cache buffers. Within a runner lifecycle,
the summary is printed only on the first graph or when backend capacities or the
segment count change.

## Tune direct convolution.

`--conv-direct on|off|auto` selects direct convolution or the im2col-based
path. Performance depends on GPU architecture, backend, driver, model, and
tensor shapes, so enabling direct convolution can either improve or reduce speed.

Automatic selection uses small benchmarks on Vulkan and implicit GEMM checks
on other backends. It does not guarantee the fastest end-to-end generation.
Compare `on` and `off` with the same model and generation settings over
repeated runs, keeping flash attention unchanged and separating startup/probe
time from steady-state generation time. See [backend selection](backend.md)
for automatic selection rules and per-module overrides.

## Run conditional and unconditional CFG in one batched UNet forward.

For UNet models, the conditional and unconditional guidance branches are
concatenated into a single batch of two and run through one UNet forward per
step instead of two separate forwards. This is enabled by default whenever the
run qualifies for it.

Use `--batched-cfg off` to force separate conditional and unconditional
forwards.

## Use VAE tiling to reduce encode and decode memory usage.

`--vae-tiling` enables spatial tiling for both VAE encoding and decoding. The
default tile size is 256x256 **image pixels**, independent of the VAE scale factor:

```shell
--vae-tiling --vae-tile-size 256x256 --vae-tile-overlap 0.5
```

`--vae-tile-size` accepts one size or `WIDTHxHEIGHT`. A zero dimension uses the
256-pixel default. Sizes are rounded down to a multiple of the VAE scale factor
and capped at the current input dimensions. Explicit sizes below four latent
pixels per axis (or the full axis when it is smaller) are rejected. Encoding and
decoding use the same spatial sizes, without an additional encoding multiplier.
Inputs that fit within a tile are processed as one tile.

For a 512x512 image with the default 50% overlap, both encoding and decoding use
3x3 tiles. A 256-pixel tile corresponds to 32 latent pixels for an 8x VAE, 16 for
a 16x VAE, and 8 for a 32x VAE. Smaller tiles reduce each graph's memory demand,
but overlapping work can increase processing time and tiling can affect image
quality, especially during encoding. Use larger tiles when more context is needed.

`--vae-relative-tile-size` overrides the absolute size on each axis with a positive
value. Values up to and including 1 specify a fraction of the current input size;
values greater than 1 specify a target number of tiles per axis, accounting for
overlap. For example, `0.5x0.5` uses half the width and height in both encode and
decode. The target overlap is clamped to 0 through 0.5 and the actual overlap is
adjusted to fit the image. Size and overlap options require `--vae-tiling`.

**Migration:** `--vae-tile-size` and the C/JSON fields `tile_size_w` and
`tile_size_h` now use image pixels instead of latent units. The C/JSON fields
`tile_size_x/y` have been renamed to `tile_size_w/h`, and `rel_size_x/y` to
`rel_size_w/h`. The command-line option names are unchanged. For example, an old
decode tile size of 32 corresponds to 256 pixels for an 8x VAE or 512 pixels for a
16x VAE. Encoding no longer enlarges explicit or relative tile sizes.

The main VAE decode path retries allocation failures with smaller tiles, even
without `--vae-tiling`. Supported video VAEs first try temporal tiling; spatial
retries use at most 256-pixel tiles initially and then halve the effective tile
dimensions down to the minimum size. Each spatial retry must reduce the effective
tile size. These runtime adjustments do not change the caller's parameters.
Execution failures are not retried, and encoding has no automatic OOM retry.

`--temporal-tiling` remains independent of spatial tiling. MiniMax H3 always uses
spatial tiling (256x256 pixels and 25% overlap by default) and its own temporal
windows. With `--vae-tiling`, its overlap follows `--vae-tile-overlap`; explicit
spatial sizes are honored.

## Offload weights to the CPU to save VRAM without reducing generation speed.

Using `--offload-to-cpu` allows you to offload weights to the CPU, saving VRAM without reducing generation speed.

## Use params backend to reduce VRAM or RAM usage.

`--params-backend` controls where model parameters are kept. If it is not set, auto-fit chooses parameter placement while preserving `--backend`. With `--auto-fit off`, parameters use the same backend as `--backend`, so a GPU runtime backend also keeps parameters in VRAM.

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

This reloads parameters from the model file on demand, retains unpinned compute copies while space permits, and releases them under pressure or at module-run completion. It has the lowest source-memory residency, but can be slower because evicted weights must be read again. `disk` is never selected implicitly; set it explicitly when RAM usage matters more than reload cost.

Per-module assignments can target only the largest modules:

```shell
--backend cuda0 --params-backend diffusion=disk,te=cpu,vae=cpu
```

See [backend selection](./backend.md) for full syntax.

## Run models that don't fit in VRAM (automatic segmented execution).

`--offload-to-cpu` keeps the source parameters in system RAM and creates compute-side GPU replicas on demand. Unpinned replicas remain resident for reuse, but automatic graph-cut execution evicts them from the last segment backward when the next weight or compute allocation needs space. Disk-backed parameters follow the same policy without retaining a RAM source copy.

When a graph has cut markers and its missing weights plus incremental compute workspace exceed the available device headroom, it runs its fixed segment list in order. A reusable monolithic compute buffer is not counted as a new allocation. An explicit `--max-vram` budget deducts already-resident managed weights and compute/cache buffers registered by every runner sharing the device, so later graph runs remain segmented when the full graph exceeds the budget. The current segment's weights are pinned during compute, and the next parameter-bearing segment is prefetched when the device supports asynchronous transfer. No opt-in streaming flag is required.

When choosing between monolithic and segmented execution, the runner requires
an additional 128 MiB of headroom in both available device memory and any explicit
managed budget. This planning headroom absorbs small allocation estimate changes;
subsequent capacity checks can consume it while still preserving the 512 MiB device
scratch reserve and respecting the managed budget.

- `--max-vram <GiB>` optionally lowers the live-memory limit. A positive value is a managed per-device budget, `0` uses the device's current free memory without an explicit budget, and a negative value snapshots free memory at startup while reserving that many GiB (`--max-vram -1` reserves about 1 GiB). Driver contexts and unrelated external allocations remain outside the managed budget.
- `--disable-prefetch` disables asynchronous next-segment prefetch while retaining synchronous loading, eviction, and segmented execution.
- `--disable-segmented-compute` forces monolithic graph execution for diagnostics or compatibility, even when the automatic memory check would select segments.

Single-device monolithic execution also reclaims unpinned weight replicas before
loading weights or allocating compute workspace, including graphs without cut
markers and runs with `--disable-segmented-compute`. It still respects the managed
device budget and fails if the graph cannot fit after reclamation.

Segment completion releases active workspace use while retaining the runner's
allocator/scheduler capacity. Compatible gallocr reservations are reused across
graphs; idle workspaces can be reclaimed under pressure and are freed at runner
completion. Cross-graph caches survive individual graphs, but cut buffers do not.

The recommended shape for "biggest model my card can host" is:

```shell
sd-cli --diffusion-model flux1-dev.safetensors ... \
       --offload-to-cpu --max-vram -1
```

- `--offload-to-cpu`: params in RAM, staged as needed.
- `--max-vram -1`: reserve about 1 GiB from the startup free-memory snapshot; live free memory can still lower the effective limit for every graph.

Use `--params-backend diffusion=disk` instead when reducing system RAM residency is more important than avoiding repeated model-file reads.

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
