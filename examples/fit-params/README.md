# sd-fit-params

`sd-fit-params` computes the CLI arguments that make a model fit into free
device memory, using measured metadata-only dry runs: the real generation
pipeline is executed with graph building and memory measurement only, so
no weight data is read and no ggml weight or compute buffers are allocated.
Shaped host tensors are still materialized to carry state between graph builds;
allocation failures are reported as fit errors. The measured per-module memory
includes projected persistent cache buffers and is packed against the free
memory of every GPU device. A 512 MiB safety margin is retained after applying
either detected or explicit `--max-vram` limits, and the resulting placement is
printed to stdout as `--backend` / `--params-backend` / `--vae-tiling` /
`--stream-layers` arguments.

Because compute memory depends on the generation parameters, pass the same
width/height (and video frames) you intend to generate with. Example usage:

``` bash
# First, run sd-fit-params and store the results in a file:
> ./build/bin/sd-fit-params -m sd_v1-5.gguf -W 1024 -H 1024 --max-vram 4 | tee args.txt
[INFO ] fit_params.cpp:93   - fit-params: measured memory plan
[INFO ] fit_params.cpp:93   -   devices:
[INFO ] fit_params.cpp:93   -     MTL0         Apple M4                         free  12123 MiB, budget   3584 MiB
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
- `--max-vram <GiB>` or `--max-vram cuda0=8,cuda1=14`: per-device limits.
  The planner retains 512 MiB of headroom after applying the limit. Positive
  values cap the graph-splitting budget, negative values use auto budget
  detection, and `0` disables graph splitting.
- `--fit-print`: print the measured memory table to stdout instead of arguments
- `-p`: representative prompt (token count affects text encoder memory)
- generation inputs including init/control/reference images, LoRAs, and hires
  options flow into the measurement as they would into a real run; explicit
  `--backend` / `--params-backend` placement is rejected

## Planner order

The planner tries the fastest and most resident placements first, then falls
back to progressively lower-VRAM choices. Device budgets retain a 512 MiB
margin after both automatic and explicit `--max-vram` limits. If no GPU device
is available, the tool verifies that the workload fits available host memory
before keeping the default CPU backend.

The checks run in this order:

1. Default placement: put every module on the first GPU. This succeeds when
   the sum of all module parameters plus the peak measured compute phase fits
   that device budget. Diffusion and ControlNet buffers are added because both
   remain live during denoising; other sequential module buffers use their
   maximum. If it succeeds, the tool prints an empty line
   because no extra CLI arguments are needed.
2. Resident multi-device placement: sort modules by parameter size, largest
   first, and place each module on one GPU while keeping all parameters resident.
   For each GPU, resident parameters accumulate and sequential compute buffers
   use their maximum. Diffusion and ControlNet compute buffers are summed when
   assigned to the same GPU.
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
   the CPU runtime backend. The planner returns `SD_FIT_FAILURE` if the CPU
   parameters and compute phases exceed currently available host memory.

If the default placement already fits, the tool prints an empty line and
reports that no changes are needed. Explicit `--backend` / `--params-backend`
assignments are rejected because the tool derives placement rather than
validating an existing assignment.

See `docs/backend.md` for the placement spec syntax and the heuristic
`--auto-fit` alternative built into `sd-cli`.

## Library API

Library callers can use the same measured fitting through `sd_fit_params()` in
`stable-diffusion.h`. Start from initialized context params and workload params,
then free the result when done:

```c
sd_ctx_params_t ctx;
sd_ctx_params_init(&ctx);
ctx.diffusion_model_path = "/models/model.gguf";
ctx.max_vram = "8";

sd_fit_workload_t workload;
sd_fit_workload_init(&workload);
workload.prompt = "a cat";
workload.width = 1024;
workload.height = 1024;
workload.video_frames = 1;

sd_fit_result_t result;
enum sd_fit_status_t status = sd_fit_params(&ctx, &workload, &result);
if (status == SD_FIT_SUCCESS && result.changed) {
    printf("backend=%s\n", result.backend ? result.backend : "");
    printf("params_backend=%s\n", result.params_backend ? result.params_backend : "");
    printf("vae_tiling=%d\n", result.vae_tiling);
    printf("stream_layers=%d\n", result.stream_layers);
}
sd_fit_result_free(&result);
```

For simple text-to-image or text-to-video fitting, the scalar workload fields
are enough. `sd_fit_workload_init()` defaults to a 512x512 image workload
(`video_frames = 1`) and default VAE tiling params.

For the most accurate plan, pass a full representative request:

```c
sd_img_gen_params_t image_request;
sd_img_gen_params_init(&image_request);
image_request.prompt = "a cat";
image_request.width = 1024;
image_request.height = 1024;
image_request.batch_count = 1;
image_request.vae_tiling_params = workload.vae_tiling_params;

workload.image_gen_params = &image_request;
```

Use `workload.video_gen_params` with `sd_vid_gen_params_t` for video. Set at
most one of `image_gen_params` and `video_gen_params`; setting both returns
`SD_FIT_ERROR`. When a full request is present, it supplies conditioning,
LoRAs, hires/cache options, image/video/audio inputs, VAE tiling settings, and
other generation fields. The scalar workload fields remain as a fallback for
callers that only need a basic request. A video-only model selects the video
measurement pipeline even if an image request was supplied; shared request
fields are promoted to a video request. Supplying a video request for an
image-only model returns `SD_FIT_ERROR`.

`sd_fit_params()` returns:

- `SD_FIT_SUCCESS`: a placement was found, or the current/default placement
  already fits.
- `SD_FIT_FAILURE`: no placement was projected to fit, or `ctx.backend` /
  `ctx.params_backend` was already set by the caller.
- `SD_FIT_ERROR`: invalid inputs or a hard measurement error such as an
  unreadable model.

`sd_fit_result_t` owns `backend`, `params_backend`, and `report`; always call
`sd_fit_result_free()`. If `result.changed` is false, the current/default
placement already fits and the placement strings are null. If
`result.stream_layers` is true, preserve the caller's nonzero `ctx.max_vram`
when applying the result and also enable `--stream-layers`; the max-VRAM value
is not duplicated in the result.

## Debugging the planner

Set `SD_FIT_DEBUG_DEVICES` to plan against simulated devices instead of the
real ones, e.g. `SD_FIT_DEBUG_DEVICES="CUDA0:24,CUDA1:16"` (`name:free_gib`).
Measurement still runs on the real machine; only device enumeration is
replaced. Useful to preview placements for other hardware and to exercise
multi-device planning paths. Debug only: the emitted specs reference the
simulated device names.
