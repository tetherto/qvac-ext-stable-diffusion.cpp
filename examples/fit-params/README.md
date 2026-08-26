# sd-fit-params

`sd-fit-params` computes the CLI arguments that make a model fit into free
device memory, using measured metadata-only dry runs: the real generation
pipeline is executed once with graph building and memory measurement only, so
no weight data is read and no ggml weight or compute buffers are allocated.
Shaped host tensors are still materialized to carry state between graph builds;
allocation failures are reported as fit errors. The measured per-module memory
is then packed against the free memory of every GPU device (minus a
512 MiB margin, or the `--max-vram` budgets) and the resulting placement is
printed to stdout as `--backend` / `--params-backend` / `--vae-tiling` /
`--stream-layers` arguments.

Because compute memory depends on the generation parameters, pass the same
width/height (and video frames) you intend to generate with. Example usage:

``` bash
# First, run sd-fit-params and store the results in a file:
> ./build/bin/sd-fit-params -m sd_v1-5.gguf -W 1024 -H 1024 --max-vram 4 | tee args.txt
[INFO ] fit_params.cpp:93   - fit-params: measured memory plan
[INFO ] fit_params.cpp:93   -   devices:
[INFO ] fit_params.cpp:93   -     MTL0         Apple M4                         free  12123 MiB, budget   4096 MiB
[INFO ] fit_params.cpp:93   -   modules (measured for this workload):
[INFO ] fit_params.cpp:93   -     diffusion    params   1398 MiB, compute   8360 MiB
[INFO ] fit_params.cpp:93   -     te           params    125 MiB, compute      1 MiB
[INFO ] fit_params.cpp:93   -     vae          params    159 MiB, compute   6656 MiB (tiled    416 MiB)
[INFO ] fit_params.cpp:93   -   placement (time-share: params load per phase and free after):
[INFO ] fit_params.cpp:93   -     diffusion    -> cpu
[INFO ] fit_params.cpp:93   -     te           -> MTL0, params on disk
[INFO ] fit_params.cpp:93   -     vae          -> MTL0, params on disk, vae tiling
[INFO ] stable-diffusion.cpp - fit-params: fitting params to free memory took 0.26s
printing fitted CLI arguments to stdout...
--backend "diffusion=cpu,te=MTL0,vae=MTL0" --params-backend "te=disk,vae=disk" --vae-tiling

# Next, use those results for sd-cli:
> cat args.txt | xargs ./build/bin/sd-cli -m sd_v1-5.gguf -p "a cat" -W 1024 -H 1024
```

Useful flags:

- `-W` / `-H` / `--video-frames`: the workload the fit must accommodate
- `--max-vram <GiB>` or `--max-vram cuda0=8,cuda1=14`: per-device budgets
  (default: free memory minus 512 MiB per device). Positive values cap the
  graph-splitting budget, negative values use auto budget detection, and `0`
  disables graph splitting.
- `--fit-print`: print the measured memory table to stdout instead of arguments
- `-p`: representative prompt (token count affects text encoder memory)
- generation inputs including init/control/reference images, LoRAs, hires, and
  model placement options flow into the measurement as they would into a real run

## Planner order

The planner tries the fastest and most resident placements first, then falls
back to progressively lower-VRAM choices. Device budgets come from the current
free GPU memory minus a 512 MiB margin, unless `--max-vram` provides an explicit
budget. If no GPU device is available, the tool keeps the default backend.

The checks run in this order:

1. Default placement: put every module on the first GPU. This succeeds when
   the sum of all module parameters plus the largest measured compute buffer
   fits that device budget. If it succeeds, the tool prints an empty line
   because no extra CLI arguments are needed.
2. Resident multi-device placement: sort modules by parameter size, largest
   first, and place each module on one GPU while keeping all parameters resident.
   For each GPU, resident parameters accumulate and only the largest compute
   buffer assigned to that GPU is counted, because module compute phases do not
   run at the same time.
3. Resident VAE tiling: while trying the resident plan, if a module has a
   measured tiled compute size and full-resolution compute does not fit, retry
   that module with tiled compute. This currently applies to VAE measurements
   and emits `--vae-tiling`.
4. Time-share single-device placement: if resident placement fails, plan each
   module as a separate phase. A module can run on a GPU with
   `--params-backend <module>=disk` when its parameters plus its compute buffer
   fit one device budget.
5. Time-share VAE tiling: if the non-tiled time-share check fails and the module
   has a tiled compute measurement, retry with the tiled compute size and emit
   `--vae-tiling` if it fits.
6. Multi-GPU split: if the module is splittable and more than one GPU exists,
   split its parameters across all GPUs when the sum of each device budget minus
   that module's compute buffer can hold the module parameters. The emitted
   backend uses `&`, for example `diffusion=CUDA0&CUDA1`, and parameters are
   loaded per phase from disk.
7. Diffusion CPU params plus layer streaming: if split placement still does not
   fit, and the module is a splittable diffusion module, choose the GPU with the
   largest graph-splitting budget and keep diffusion parameters in CPU RAM while
   streaming layers to the runtime GPU. This emits
   `--params-backend diffusion=cpu`, preserves the original `--max-vram`, and
   adds `--stream-layers`. This fallback is only considered when graph splitting
   is enabled by a positive or negative `--max-vram`; `--max-vram 0` disables it.
8. CPU runtime fallback: if none of the GPU options above fit, put the module on
   the CPU runtime backend.

If the current parameters already fit, the tool prints an empty line and
reports that no changes are needed. If `--backend` / `--params-backend` are
already set and changes would be needed, the tool fails instead of overriding
them.

See `docs/backend.md` for the placement spec syntax and the heuristic
`--auto-fit` alternative built into `sd-cli`.

## Debugging the planner

Set `SD_FIT_DEBUG_DEVICES` to plan against simulated devices instead of the
real ones, e.g. `SD_FIT_DEBUG_DEVICES="CUDA0:24,CUDA1:16"` (`name:free_gib`).
Measurement still runs on the real machine; only device enumeration is
replaced. Useful to preview placements for other hardware and to exercise
multi-device planning paths. Debug only: the emitted specs reference the
simulated device names.
