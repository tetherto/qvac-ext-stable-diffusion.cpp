# sd-fit-params

`sd-fit-params` computes the CLI arguments that make a model fit into free
device memory, using measured metadata-only dry runs: the real generation
pipeline is executed once with graph building and memory measurement only, so
no weight data is read and no buffers are allocated. The measured per-module
memory is then packed against the free memory of every GPU device (minus a
512 MiB margin, or the `--max-vram` budgets) and the resulting placement is
printed to stdout as `--backend` / `--params-backend` / `--vae-tiling`
arguments.

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
  (default: free memory minus 512 MiB per device)
- `--fit-print`: print the measured memory table to stdout instead of arguments
- `-p`: representative prompt (token count affects text encoder memory)
- model placement inputs such as `--type`, `--diffusion-fa`, `--vae-tiling`
  flow into the measurement exactly as they would into a real run

If the current parameters already fit, the tool prints an empty line and
reports that no changes are needed. If `--backend` / `--params-backend` are
already set and changes would be needed, the tool fails instead of overriding
them.

See `docs/backend.md` for the placement spec syntax and the heuristic
`--auto-fit` alternative built into `sd-cli`.
