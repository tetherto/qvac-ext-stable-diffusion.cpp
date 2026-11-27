# Fabric integration preflight

The engine callback helper uses public GGML graph APIs so the engine can consume
an installed GGML package without private graph definitions. This preflight
validates that boundary before any production dependency-pin change.

## Checks

All CPU, CUDA and Metal lanes build the engine against installed public headers,
with pinned Fabric and diffusion-GGML control revisions. Only the existing
`test-fit-params` backend mock receives the matching private backend implementation
header through a target-specific include directory. The engine and other tests do
not inherit that include directory.

Each job builds a shared core and dynamic CPU backend, plus the requested GPU
backend. It tests CPU initialization, required backend discovery, independent
Flux RoPE/padding/IM2COL3D values and layouts, and five model-free engine tests.
The callback regression covers empty graphs, selection masks, cancellation,
in-place execution and the no-callback path. CUDA and Metal lanes additionally
run it on the requested GPU. Missing hardware and unsupported operations fail;
CPU fallback is not accepted as GPU validation. CUDA checks CPU availability
with the GPU hidden. A CPU-only install contains no GPU module.

The trusted branch push runs the workflow. There is no pull-request-triggered
execution on self-hosted runners. Runner labels match the Fabric CI mappings.
Each job uploads commands, return codes and logs even on failure.

## Local reproduction

Use separate source checkouts for Fabric and diffusion GGML. CMake, Ninja, a C++17
compiler and Python3 are required. CUDA requires nvcc and a compatible GPU;
Metal requires Apple hardware and its SDK. Run from the engine checkout:

```sh
python3 scripts/fabric_preflight.py --ggml-source ../fabric/ggml   --backend CPU --mode package --output ../results/package
python3 scripts/fabric_preflight.py --ggml-source ../fabric/ggml   --backend CUDA --mode package --output ../results/fabric
python3 scripts/fabric_preflight.py --ggml-source ../diffusion-ggml   --backend CUDA --mode package --output ../results/control
```

Every invocation requires a fresh output directory, preventing a diagnostic
CMake cache from contaminating the clean-package check.
`--installed-prefix` can reuse a matching local package for diagnosis, but does
not verify its provenance or installation; CI always builds and installs afresh.
`--cuda-architectures` defaults to `native` and can be set explicitly.

## Model comparison

The automatic workflow is model-free. To compare a locally available SD2.1 Q8
GGUF fixture, install Pillow and run after both clean-package engine builds:

```sh
python3 scripts/compare_fabric_models.py --control ../results/control   --fabric ../results/fabric --model /path/to/sd21-q8.gguf   --backend cuda0 --output ../results/images
```

This runs square, flash-attention and landscape cases twice on each matching
engine/core pair with fixed seeds and CPU RNG. It writes images, verbose logs,
pixel hashes and differences. Exact parity and repeatability are required for
exit0; nonzero differences are evidence to investigate, not proof of quality
regression. Timing includes startup and is not a performance acceptance gate.
Use `--backend mtl0` for Metal model runs; its GGML registry is named `MTL`.
Use normal freshly installed preflight output trees, not `--installed-prefix`
trees, for this command.

## Local validation and remaining acceptance

The public-API callback fix was validated on Linux CPU and CUDA against Fabric
`ccb0962b151b8710b3c9f0b35b81c1a40243738b` and control
`2ab141bb0fb9028cd847c3f5095c453d9646c5ea`. The clean installed-Fabric engine build,
five focused engine tests, CPU/CUDA operation probes, GPU-hidden CPU discovery,
and callback tests on both cores passed.

Paired SD2.1 Q8 generations used identical prompts, seeds, sampling settings and
CPU RNG. These local full-model runs are separate from the model-free CI:

| Case | Generations per core | Mean absolute pixel difference / 255 | Repeatability |
| --- | --- | --- | --- |
| CPU, 256 x 256, 2 steps | 1 | 0 | Not measured |
| CUDA, 512 x 512, 12 steps | 2 | 3.5073 | Exact within each core |
| CUDA, flash attention, 512 x 512, 12 steps | 2 | 3.9458 | Exact within each core |
| CUDA, 512 x 384, 12 steps | 2 | 4.3632 | Exact within each core |

All fourteen generations completed. CUDA cross-core pixels are not exact; the
comparison's exact-parity requirement has not been relaxed. Per-channel maxima
were 241, 254 and 191, respectively, so the mean is not a per-pixel bound.

A separate per-node diagnostic found identical first 2178 tensors and identical
inputs to the first differing Q8_0 x F32 matrix multiplication. Its output maximum
absolute difference was 2.384e-7 and relative L2 difference was 2.416e-8. An
independent double-precision reference over 77120 sampled outputs gave essentially
equal RMSE (0.0026048558) for both cores. This supports rounding at that operation;
it does not establish final-image quality acceptance. Disabling fusion or forcing
F32 cuBLAS did not eliminate the final-image differences.

CUDA image-quality acceptance remains open because no agreed threshold or review
criterion exists. Full FLUX/video/ConvRot fixtures and native Android/Windows,
HIP and Vulkan migration validation are outside this workflow. Metal callback
validation is provided by CI and must be checked before merge.

The callback-only graph ranges lack private use-count metadata, so optimizations
that require that metadata may be inhibited. Normal execution with no callback
continues through the original direct backend compute path.

Do not infer production migration approval from passing model-free CI. Production
submodule and package pins remain unchanged.
