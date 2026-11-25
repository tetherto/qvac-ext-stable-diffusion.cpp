# Fabric integration preflight

This draft validates an external GGML package without changing the engine's
submodule, dependency pins, runtime code, or packaging. It is not a migration
approval. The workflow pins Fabric and diffusion GGML revisions for reproducible
comparisons and uploads every stage's command, exit status, and logs.

## Checks

The clean `package-fabric` job uses only the installed package. Diagnostic jobs
for CPU, CUDA and Metal explicitly expose the matching GGML private headers to
continue runtime testing despite the known engine package-boundary failure.
**A green diagnostic job does not mean the clean package is usable.**

Each job builds a shared core and dynamic CPU backend, plus the requested GPU
backend. It tests CPU initialization, required backend discovery, independent
Flux RoPE/padding/IM2COL3D values and layouts, and four model-free engine tests.
Missing requested hardware and unsupported operations fail the job; no CPU
fallback or skip is accepted as GPU validation. CUDA also checks CPU availability
with the GPU hidden. A CPU-only install contains no GPU module.

The trusted branch push runs the workflow, including while the PR is a draft.
There is no pull-request-triggered execution on self-hosted runners. Runner labels
match the Fabric CI mappings; access from this repository still requires CI
confirmation. Manual dispatch is available once GitHub registers the workflow.

## Local reproduction

Use separate source checkouts for Fabric and diffusion GGML. CMake, Ninja, a C++17
compiler and Python3 are required. CUDA requires nvcc and a compatible GPU;
Metal requires Apple hardware and its SDK. Run from the engine checkout:

```sh
python3 scripts/fabric_preflight.py --ggml-source ../fabric/ggml   --backend CPU --mode package --output ../results/package
python3 scripts/fabric_preflight.py --ggml-source ../fabric/ggml   --backend CUDA --mode diagnostic --output ../results/fabric
python3 scripts/fabric_preflight.py --ggml-source ../diffusion-ggml   --backend CUDA --mode diagnostic --output ../results/control
```

Every invocation requires a fresh output directory, preventing a diagnostic
CMake cache from contaminating the clean-package check.
`--installed-prefix` can reuse a matching local package for diagnosis, but does
not verify its provenance or installation; CI always builds and installs afresh.
`--cuda-architectures` defaults to `native` and can be set explicitly.

## Model comparison

The automatic workflow is model-free. To compare a locally available SD2.1 Q8
GGUF fixture, install Pillow and run after both diagnostic engine builds:

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

## Current acceptance gaps

- The engine includes `ggml/src/ggml-impl.h` and constructs a private graph view.
  The clean package build fails; removing the include alone is insufficient.
- Baseline Fabric rejects strided CUDA IM2COL3D. The workflow now tests the
  candidate fix from [Fabric PR594](https://github.com/tetherto/qvac-fabric-llm.cpp/pull/594).
  Keep the strict check enabled to confirm the candidate restores GPU execution.
- Local SD2.1 CPU comparisons are exact; CUDA outputs differ reproducibly.
  Stage-level numerical investigation is still needed before accepting parity.
- Full FLUX/video/ConvRot model fixtures and Android/Windows-native/HIP/Vulkan
  validation are outside this initial workflow. Model-free Metal CI is included.

The preflight deliberately does not apply fixes or relax assertions to obtain
green checks. Do not merge it as a dependency migration or infer coverage of
untested models from a successful build.
