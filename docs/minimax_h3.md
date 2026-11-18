# MiniMax-H3

MiniMax-H3 jointly generates video and stereo audio with a packed diffusion
transformer. The implementation supports text-to-audio-video (T2VA), optional
first-frame conditioning (I2VA), first/last-frame conditioning (FL2VA), and
image/video/audio reference conditioning (Ref2VA).

## Model files

Pass the four MiniMax-H3 components separately:

- `--diffusion-model`: MiniMax-H3 diffusion transformer
- `--vae`: MiniMax-H3 video VAE
- `--audio-vae`: MiniMax-H3 audio VAE
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder checkpoint

The text encoder must be the MiniMax-H3 variant: Qwen3-VL-32B truncated to 50
language layers and exported without the final language-model normalization.
Its Qwen3-VL vision tower, including the three DeepStack mergers, must also be
present. If the vision tower is stored separately, pass it with `--llm_vision`.

Both the original time-embedder DiT and the smaller AdaLN curve-table variant
are detected from their weights.

### Download weights

- Download minimax_h3_fl2va/minimax_h3_ref2va
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download qwen3vl_32b_minimax_h3
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae
- Download audio vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae

## Text-to-audio-video

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten’s energetic surfing." --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/t2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

Omitting `--audio-vae` still runs the joint diffusion model but produces video without a
decoded audio track.

## First/last-frame conditioning

Add `--init-img` for I2VA, or both `--init-img` and `--end-img` for FL2VA:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "a lovely cat" -i ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/i2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

## Reference-to-audio-video conditioning

Ref2VA accepts any combination of reference images, reference videos, paired
video soundtracks, and standalone audio references:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_ref2va_pruned-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "Use the cat from <Picture 1> as the main character. Keep the cat’s appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat’s face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat’s head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat’s fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat’s identity unchanged." -r ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/r2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

`--ref-image`, `--ref-video`, and `--ref-audio` can each be repeated. A
reference video is a directory of image frames sorted lexicographically and is
treated as 24 fps. Repeated `--ref-video-audio` WAV files are paired by index
with repeated `--ref-video` inputs. WAV PCM (8/16/24/32-bit) and 32/64-bit
floating-point samples are accepted; audio is converted to stereo 32 kHz by the
pipeline.

Reference inputs are presented to Qwen3-VL in image, video, then audio order.
Videos are sampled at 2 fps for the Qwen presentation while their full 24 fps
latents condition the diffusion transformer. Paired video and audio references
share the same timeline. Ref2VA cannot be combined with `--init-img` or
`--end-img` in one request.

Reference images keep their aspect ratio and are only downscaled when their
pixel area exceeds the requested generation canvas.

The C API exposes the same inputs through `ref_images`, `ref_videos`, and
`ref_audios` in `sd_vid_gen_params_t`. Each `sd_ref_video_t` supplies its own
frame rate and optional soundtrack; non-24-fps inputs are resampled internally.

## Shape and runtime notes

- Width and height are aligned upward to a multiple of 32.
- Frame count is aligned upward to the `17k + 5` grid, with a minimum of 5.
- MiniMax-H3 runs at 24 fps; another requested value is overridden.
- The default video flow shift is 12. The audio stream is mapped internally to
  its shift of 3, so the regular samplers can operate on the packed AV latent.

## Reproducible CUDA benchmarks

`script/bench_minimax_h3.py` runs the native CLI repeatedly with a fixed prompt,
seed and workload. It requires Python 3, `ffprobe`, `ffmpeg`, a CUDA build of
`sd-cli` and all four H3 model components. For example, from the repository root:

```sh
python3 script/bench_minimax_h3.py \
  --binary build-cuda/bin/sd-cli \
  --output-dir bench-results/h3-q4-cuda0 \
  --warmup 1 --runs 3 --timeout-seconds 1800 -- \
  --diffusion-model models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf \
  --llm models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae models/vae/minimax_h3_audio_vae_fp32.safetensors \
  -p 'A small orange paper boat floating on calm water at dusk, with a cliff in the background and gentle solo piano music.' \
  --seed 424242 --rng cpu -W 864 -H 480 --video-frames 124 \
  --steps 20 --cfg-scale 1 --diffusion-fa --params-backend disk
```

Replace the relative model paths with the files being tested. Use a new, empty
output directory for every configuration; the runner preserves existing
evidence. It sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`,
`GGML_VK_VISIBLE_DEVICES=0` and `--backend cuda0`. Backend, mode, output and
batch-count overrides are rejected. GPU queries and sampling explicitly target
physical GPU 0; GPU 1 is not used. A detected compute process on GPU 0 stops the
run unless `--allow-busy-gpu` explicitly accepts and records that condition.

Each warmup and measured run starts a fresh process. `wall_s` includes startup,
model loading and output writing; the engine's `generation_s` and conditioning,
sampling, audio-decode and video-decode timings are recorded separately.
Warmup processes are excluded from medians. Keep model files, prompt, seed,
settings, toolchain and measurement conditions matched when comparing builds.

`report.json` records commands, selected environment settings, source revisions,
runner and binary SHA256, and model paths, sizes and modification times. Model
contents are not hashed by default: add `--hash-models` to record SHA256 before
timing. Record any additional experimental environment settings alongside the
report. Per-run folders retain the CLI log, native audio/video AVI and validation
results, including failed attempts.

When `nvidia-smi` is available, `gpu-before.json` records identity and idle-check
results, and `gpu.csv` samples memory use, utilization, SM clock and temperature
every 250 ms. These are whole-device samples, can include other processes and
can miss the true memory peak. Missing, incomplete or mismatched telemetry is
labelled explicitly. Sampler startup and shutdown are outside `wall_s`; its
background overhead is present during generation, so use the same sampling
setup for both builds.

The runner checks aligned dimensions, the H3 frame grid, 24 fps, stereo 32 kHz
audio, duration agreement and successful CPU decoding. These structural checks
do not establish scene fidelity, coherent motion or acceptable sound. Review
generated video and audio against the requested prompt before accepting a
performance change.

The CUDA DiT computes noninterleaved partial RoPE from strided half-head views
and broadcast positional coefficients, then packs the attention layout once.
Its regression compares the actual Attention outputs and packed flash-attention
inputs against the original graph, with real and arbitrary positional matrices,
partial/full rotation, padded strides, batch 2, and repeated inputs.

The CUDA video VAE decoder uses head-interleaved QKV views and fused SwiGLU.
Its regression checks the actual attention and feed-forward blocks, including
head mapping, gate order, flash attention modes, and repeated inputs.

Model-free tooling checks and the native H3 regressions can be run with:

```sh
python3 -m unittest discover -s script -p test_bench_minimax_h3.py
build-cuda/bin/test-h3-modulation CPU
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  build-cuda/bin/test-h3-modulation CUDA0
NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  build-cuda/bin/test-h3-modulation CUDA0 --full-scalar-oracle
build-cuda/bin/test-h3-attention CPU
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  build-cuda/bin/test-h3-attention CUDA0
build-cuda/bin/test-h3-vae-decoder CPU
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  build-cuda/bin/test-h3-vae-decoder CUDA0
```

The CUDA regressions require a CUDA build configured with `SD_BUILD_TESTS=ON`.
The default modulation command compares both production paths at default
precision; its full scalar-oracle command disables TF32 for the FinalLayer FP32
reference, with elementwise scalar and repeat checks enabled in both.
The VAE decoder command retains default CUDA precision and checks the actual
packed flash-attention inputs separately from its host arithmetic oracle.

### Opt-in native SwiGLU quantization fusion

With a matching CUDA backend, `GGML_CUDA_H3_SWIGLU_QUANT_FUSION=1` fuses packed
F32 SwiGLU and input scaling into the existing DS4 Q8 activation quantizer for
the SM120 Q4_K `14336 -> 5376` projection with at least 512 rows. Eligible graphs
can also fold the preceding 128x output scale into this kernel. Each original
SCALE and SwiGLU operation retains its rounding order.

The serialized Q4 weights, Q8 payload, quantization scales and sums, MMQ kernel
and final output scale stay the same. Observed or shared intermediates, swapped
SwiGLU, unsafe aliases and unsupported layouts keep the existing path. The
locally evaluated integration enables PR #61's packed DiT SwiGLU; it is a
dependency of this experiment and is not reimplemented here. The split
SiLU/multiply graph keeps its existing path.

The control defaults to OFF and requires the corresponding backend version; the
engine does not update its ggml dependency pin. For diagnostic attribution,
`GGML_CUDA_H3_SWIGLU_QUANT_TRACE=1` logs host fusion dispatches, while
`GGML_CUDA_GRAPH_TRACE=1` distinguishes direct execution, capture and replay.
A replay executes captured kernels without repeating their host dispatch
messages. Disable tracing for performance measurements.

### Experimental sparse attention and TeaCache

These native CUDA controls are **off by default**. They retain the existing GGUF
weights. Output-quality qualification and the final performance comparison are
in progress; enabling a control is not quality approval for a workload.

The initial workload guard covers 124-frame text-only generation at 864×480 or
1344×768, 50 Euler steps, CFG 1, the standard video/audio shifts of 12/3, and
FlashAttention. Twenty-step requests, other latent layouts, reference inputs,
other guidance/samplers, multiple devices and graph-cut execution retain the
existing diffusion path. The corresponding SM120 backend must expose the private
H3 direct-execution capability; an older backend retains dense execution.

Set `GGML_CUDA_DISABLE_GRAPHS=1` before creating a CUDA context when using these
experiments in a build with CUDA graphs. The capability query shares the backend's
cached graph policy, so changing this variable after initialization cannot bypass
active graph capture. Sparse attention and the residual cache require direct
execution. The engine does not update its ggml dependency pin.

Sparse attention computes block-centroid proxy scores, keeps protected queries
and nearby video frames dense, and corrects both the softmax numerator and mass
for skipped blocks. Kept attention retains the existing FP16 MMA path; the
centroid correction uses BF16 Tensor Core inputs with FP32 accumulation.
Conditioning/audio query blocks remain dense, and the first/last key block of each
Stream-K partial stays dense. The selected prefix/suffix layers are dense on
computed steps. These protections do not imply byte-identical final audio after
video attention or joint hidden states have changed.

With a matching experimental ggml backend, `GGML_CUDA_H3_SOL_FUSED_ROUTING=1`
selects blocks inside the attention kernel and reuses the same query–centroid
scores for the skipped-block correction. This removes the separate proxy GEMM
and global routing matrix. Centroids and threshold statistics still require
linear-size preprocessing. This path is off by default and requires its own
video/audio qualification; kernel correctness alone does not establish output
quality or an end-to-end speedup.

TeaCache evaluates a dense prefix on every step. Within the candidate reuse
window, video, audio and joint modulation probes are compared at the start of
the cached interval. Probing starts one step before the window to initialize
the immediately preceding residual. Warmup, cooldown and exhausted budgets
use a single graph without unused probes. Reuse must satisfy
both the largest relative-change limit and a separate audio limit. Only the
middle layers can reuse their cached input and output. Replay adds the change
in input to the cached output, avoiding the rounding error from subtracting and
then reconstructing a residual. The protected suffix is evaluated with dense
attention on every step, including reused steps. These
protections do not guarantee exact audio because the joint middle layers remain
approximate. The selected step window and reuse budget also apply. Cached input/output and probe buffers belong to one
sampling request. Changed conditioning, shape or non-adjacent scheduler calls
invalidate history; failed probe/copy operations disable caching and recompute
through the dense path. Selecting another cache policy disables this experiment.

`SD_H3_CACHE_POLICY=sol-tea` selects an experimental full-stack policy based on
NVIDIA SOL-Engine's MiniMax-H3 RTX 5090 TeaCache recipe. It probes the first
block's modulated input before any transformer blocks, accumulates the relative
L1 change over the complete packed sequence, and reuses the last computed
block-stack residual while the accumulated change is below the threshold.
Recomputation refreshes the residual and resets the accumulator. The final
output projection is always evaluated. `sol-control` exercises the same split
graphs and cache copies but always recomputes the block stack.

The full-stack policy defaults to threshold 0.10, five initial computed calls,
and one final computed call. By default it has no count or consecutive-reuse cap.
`SD_H3_CACHE_MAX_CONSECUTIVE` optionally limits adjacent reused calls (0 keeps
the uncapped full-stack behavior); 1 forces a fresh computation after each reuse.
The full-stack policy ignores the bounded policy's total reuse budget, audio
threshold and protected-layer settings. It retains
request isolation, finite-probe checks, supported-shape/backend restrictions,
and dense fallback on graph/copy failure. It is not quality-qualified by the
bounded policy's validation and remains opt-in; audio requires separate review.

`SD_H3_CACHE_NUM_FORWARDS` optionally sets the full-stack cooldown boundary;
zero uses the scheduler's actual call count. NVIDIA's published 50-step recipe
sets this to 49. With 50 calls in this native sampler, an explicit value of 49
makes one-based calls 6–48 eligible and keeps calls 49–50 computed. Warmup and
cooldown refer to model calls, not solver labels. The default identity rescaling
of the relative L1 statistic is used; fitted polynomial coefficients are not
implemented.

Reference: [NVIDIA SOL-Engine RTX 5090 TeaCache](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/RTX5090/teacache.py).

| Control | Default | Meaning |
| --- | --- | --- |
| `SD_H3_SOL_ATTENTION` | `off` | `sol` / `sol-bf16`: approximate attention; `control` / `bf16-control`: all-keep diagnostic |
| `SD_H3_SOL_BETA` | `-1` | Proxy threshold coefficient; increasing it can skip more blocks |
| `SD_H3_SOL_WARMUP_STEPS` | `40` | Initial computed steps without sparse attention |
| `SD_H3_SOL_COOLDOWN_STEPS` | `2` | Final computed steps without sparse attention |
| `SD_H3_SOL_STEP_INTERVAL` | `1` | Sparse-eligible step spacing, anchored at the warmup boundary; intervening steps remain dense |
| `SD_H3_SOL_DENSE_PREFIX_LAYERS` | `4` | Initial layers kept dense on computed steps |
| `SD_H3_SOL_DENSE_SUFFIX_LAYERS` | `4` | Final layers kept dense on computed steps |
| `GGML_CUDA_H3_SOL_FUSED_ROUTING` | `0` | Opt-in fused selection and centroid correction in the attention kernel |
| `SD_H3_CACHE_POLICY` | `off` | `tea` / `control`: bounded middle layers; `sol-tea` / `sol-control`: adaptive full-stack reuse / compute-only control |
| `SD_H3_CACHE_THRESHOLD` | `0.1` | Accumulated probe-change threshold |
| `SD_H3_CACHE_AUDIO_THRESHOLD` | `0.05` | Separate accumulated audio probe-change limit |
| `SD_H3_CACHE_DENSE_PREFIX_LAYERS` | `4` | Initial layers evaluated densely before each probe |
| `SD_H3_CACHE_DENSE_SUFFIX_LAYERS` | `24` | Final layers evaluated densely on every step |
| `SD_H3_CACHE_WARMUP_STEPS` | `43` | Initial steps without residual reuse |
| `SD_H3_CACHE_COOLDOWN_STEPS` | `1` | Final steps without residual reuse |
| `SD_H3_CACHE_MAX_REUSES` | `1` | Maximum actual reuses across the request |
| `SD_H3_CACHE_MAX_CONSECUTIVE` | `1` bounded / `0` full-stack | Maximum adjacent reused steps; full-stack accepts 0 for unlimited |

Step windows use zero-based indices. A probe may deny a reuse even when the
budget permits it. Malformed options disable the associated approximation.
For an experimental alternating schedule over steps 10–47 of a 50-step request,
set `SD_H3_SOL_WARMUP_STEPS=10`, `SD_H3_SOL_COOLDOWN_STEPS=2`, and
`SD_H3_SOL_STEP_INTERVAL=2`: steps 10, 12, …, 46 are sparse-eligible, with dense
attention on the intervening steps. Layer and workload guards still apply.
This schedule requires separate video/audio validation; a dense intervening
step does not guarantee correction of an earlier approximation error.
Either threshold zero or reuse budget zero disables TeaCache probing. An empty
cached layer interval falls back to normal execution. Default parameters
are conservative candidates being qualified, not a universal negligible-loss
preset. Changing them requires another output comparison.

`GGML_CUDA_H3_FATTN_LONG_CONTEXT=1` independently extends the existing dense
prefetch kernel's sequence guard from 32,768 to 65,536 tokens on supported SM120
layouts. It changes the load schedule without introducing sparse selection or
residual reuse. It is also opt-in pending full-generation exact-output checks.
The original sequence limit remains the default.

`GGML_CUDA_H3_SOL_ATTN_TRACE=1` reports actual selected/skipped block pairs and
synchronizes to read counters. Cache logs report actual compute/reuse decisions.
Disable sparse tracing for timing. Use identical weights, inputs and flags for
dense, sparse-only, cache-only and combined comparisons, and separate warm
requests from fresh-process timing.

Qualification compares every native frame at full resolution and float PCM on
the same timeline: mean SSIM ≥0.98, minimum frame SSIM ≥0.95, AlexNet/v0.1 LPIPS
≤0.05. Audio SNR is reported per channel as a waveform-fidelity diagnostic;
it does not determine perceptual acceptance by itself. Two independent reviewers
must confirm no material audio/video regression on every case, including sound
events, distortion, speech intelligibility, stereo placement and synchronization.
Applicable speech WER may increase by at most one percentage point and additional
A/V offset by at most one frame at 24 fps. Missing or uncertain reviews remain
pending. Per-case failures cannot be hidden by averaging. Qualification criteria
and candidate settings must be frozen before fresh held-out evaluation; earlier
reports retain their original criteria and outcomes. SSIM is an image-similarity
measure, not model accuracy.


### Offline visual validation of combined sparse attention and caching

The fixed regression cohort uses 24 matched prompt/seed pairs, 124 frames at
24 fps, and 50 denoising steps. The reference has caching disabled and sparse
attention on one-based steps 35–48 with beta −0.25. The candidate retains that
sparse configuration and enables full-stack TeaCache eligibility on calls 6–48
(threshold 0.10, warmup 5, cooldown 1, configured forward count 49). Eligibility
does not imply reuse on every call; generation traces record actual reuse and
sparse dispatch. Existing reference artifacts are hash-verified before reuse.

Visual screening used local `Qwen/Qwen3-VL-8B-Instruct` in BF16, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Each pair supplied all 124 native RGB frames,
spatially reduced to approximately 512×256 or 512×288 for sequence review,
plus separate native-resolution start, middle and maximum-LPIPS frame pairs.
Both reference-first and candidate-first orders were evaluated with greedy
decoding. Artifact severity and temporal consistency were assessed separately
from diagnostic SSIM, LPIPS and stereo RMS differences. Missing evidence,
invalid responses and attachment-order disagreements were retained as
inconclusive. Identical-input and synthetic alternating-white-frame controls
provided basic judge checks; they are not a substitute for human calibration.

The retained run completed 24/24 generation pairs and
24/24 model reviews. Screening results: 0 pass,
0 fail, 24 inconclusive, 0 not completed.
The judge passed the identical-input control but returned inconclusive for the
deliberately corrupted alternating-white-frame control. This failed calibration
forced all aggregate case verdicts to remain inconclusive, including cases whose
individual model responses said pass. The run therefore does not establish
reliable automatic quality acceptance or rejection; raw artifact flags require
human review.
Generation state: `all-24-pairs-ready-for-local-vision-judge`.
Judge state: `completed-visual-screening-human-and-audio-review-pending`.
These are model screening outcomes, not final quality qualification. Spatial
resizing and the model's two-frame temporal patches can conceal fine or short
defects. Human visual review, native stereo listening and audio/video
synchronization checks remain separate. Single-run timing ratios are
exploratory and do not establish a repeated, quality-qualified speedup.

Judge scripts, model weights, Python dependencies, prompts and generated reports
are external validation artifacts and are not part of the engine code, build,
tests or PR changes. Evidence identifier: `qwen3vl-20261116`.

Follow-up diagnostics verified that the alternating white frames reached the
model's preprocessed video tensors. The same model recognized blank stills and
fully blank videos but missed alternating one-frame white flashes in the tested
video representations, including a synchronized side-by-side comparison.
Presenting consecutive matched frames as separate images detected both the
white-frame and heavy-blur controls and passed identical-image controls in both
attachment orders (six checks). This supports further calibration of frame-pair
screening; it does not qualify all 24 cases. Pilot frame-pair responses also
treated material/texture differences as severe defects, requiring human
adjudication of acceptable generative variation.

The initial response-format checker additionally rejected empty explanations
for categories marked pass with no defect. A separate corrected format audit
allows those empty fields while retaining evidence and frame-index requirements
for reported defects. It recovered 30 of 32 rejected responses out of 96 views;
two remained invalid. Original reports are preserved, and this format correction
does not override the failed video-judge calibration or confer quality approval.

### Focused cache-window comparison: calls 18–48

After human review identified video and audio degradation with cache eligibility
6–48, a focused three-case experiment changed only
`SD_H3_CACHE_WARMUP_STEPS` from 5 to 17. Sparse attention remained on steps
35–48 with beta −0.25; the cache threshold, cooldown, configured forward count,
binary, prompts, seeds and other generation settings were unchanged. All runs
used physical GPU 0. References remained sparse/cache-OFF outputs, and the
previous cache 6–48 outputs were retained for three-way playback.

Values below compare each cache configuration with its matching reference;
arrows show cache 6–48 → cache 18–48. RMS level differences are diagnostics,
not perceptual audio scores.

| Case / seed | Mean SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- |
| Scissors / 2836001 | 0.6695 → 0.8936 | 0.3282 → 0.0576 | 1.8768 → 2.2910 | 2.0004 → 2.3363 |
| Speech / 2836005 | 0.4548 → 0.8301 | 0.4679 → 0.1071 | 4.4108 → 1.7636 | 4.5984 → 2.1276 |
| Basketball / 2836017 | 0.4964 → 0.8692 | 0.4663 → 0.0995 | 13.9101 → 5.9975 | 13.8822 → 5.9469 |

Generation traces verified 25 actual reused calls in each new run, all within
18–48, versus 35 with the earlier window. Sparse dispatch remained within the
intended window on computed calls. New generation times were 162.79, 163.23 and
164.95 seconds, versus 302.55, 303.63 and 305.70 seconds for the retained
references: approximately 1.85–1.86× in these single-run observations.

Video similarity improved in all three cases; audio level differences improved
in two and increased slightly in the scissors case. Basketball audio remained
approximately 6 dB lower per channel than its reference. This experiment does
not establish repaired audio quality or a full-cohort quality pass. Manual
video, listening and synchronization review remain pending. The 18–48 profile
has not yet been evaluated across all 24 cases or accepted as a default.
Validation tooling and outputs remain external; evidence identifier:
`cache18-48-20261116`.

### Focused sparse-window comparison: steps 32–48, caching OFF

A focused three-case experiment expanded sparse attention from steps 35–48
to 32–48, changing only `SD_H3_SOL_WARMUP_STEPS` from 34 to 31.
Caching remained OFF; beta −0.25, the binary, prompts, seeds, 50 steps,
resolution and other settings matched the retained sparse 35–48 references.
Generation used physical GPU 0. Trace checks required 374 sparse layer
dispatches (17 steps × 22 layers) and no cache reuse.

Native RGB comparisons measured all 124 frames. Audio columns report
absolute whole-clip RMS level differences for each native float PCM channel;
these are diagnostics and do not establish perceptual audio quality.

| Case / seed | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Max LPIPS | Left stereo diff (dB) | Right stereo diff (dB) | Reference / candidate generation (s) | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / 2836001 | 0.9935 | 0.9831 | 0.0039 | 0.0119 | 0.0226 | 0.0071 | 302.55 / 297.49 | 1.0170× |
| speech-1 / 2836005 | 0.9949 | 0.9929 | 0.0027 | 0.0040 | 0.0062 | 0.0415 | 303.63 / 301.47 | 1.0072× |
| motion_sync-1 / 2836017 | 0.9953 | 0.9818 | 0.0028 | 0.0117 | 0.0573 | 0.0757 | 305.70 / 303.98 | 1.0057× |

Timing ratios compare single new runs with retained historical references;
small differences need matched repeated measurements before claiming a gain.
Manual visual, audio and synchronization review remains pending. This is
a focused diagnostic experiment, not a full 24-case qualification or an
automatic quality pass. No Qwen approval is used. External playback files
include both configurations and lossless audio. Validation scripts and
outputs stay outside the engine repository; evidence identifier:
`sparse32-48-cacheoff-20261116`.

#### Qwen frame-pair screening of the sparse 32–48 experiment

An external local Qwen3-VL-8B-Instruct review compared the same three pairs
using overlapping four-frame groups (stride three), covering all 124 native
RGB frames in both attachment orders. Settings and numerical quality scores
were withheld from the judge. Identical, alternating-white and heavy-blur
controls were repeated, with additional per-case identical controls.

Model revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Global controls passed: True.

| Case | Visual screening | Pass / fail / inconclusive windows | Order disagreements |
| --- | --- | --- | --- |
| sharp_transients-1 | pass | 41 / 0 / 0 | 0 |
| speech-1 | pass | 41 / 0 / 0 | 0 |
| motion_sync-1 | pass | 41 / 0 / 0 | 0 |

A window passes only when both orders pass; conflicting orders are
inconclusive. Concordant failures are flags for human adjudication, not
proof of a defect. These short frame groups do not replace full-speed
playback, audio listening or synchronization review. No final audiovisual
acceptance is conferred. Raw responses, controls and cited frames remain
external; no judge code or model weights are part of the repository.

### Focused sparse-window comparison: steps 28–48, caching OFF

A focused three-case experiment expanded sparse attention from steps 35–48
to 28–48, changing only `SD_H3_SOL_WARMUP_STEPS` from 34 to 27.
Caching remained OFF; beta −0.25, the binary, prompts, seeds, 50 steps,
resolution and other settings matched the retained sparse 35–48 references.
Generation used physical GPU 0. Trace checks required 462 sparse layer
dispatches (21 steps × 22 layers) and no cache reuse.

Native RGB comparisons measured all 124 frames. Audio columns report
absolute whole-clip RMS level differences for each native float PCM channel;
these are diagnostics and do not establish perceptual audio quality.

| Case / seed | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Max LPIPS | Left stereo diff (dB) | Right stereo diff (dB) | Reference / candidate generation (s) | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / 2836001 | 0.9911 | 0.9741 | 0.0054 | 0.0168 | 0.0067 | 0.0351 | 302.55 / 296.15 | 1.0216× |
| speech-1 / 2836005 | 0.9881 | 0.9754 | 0.0076 | 0.0131 | 0.0800 | 0.0688 | 303.63 / 299.58 | 1.0135× |
| motion_sync-1 / 2836017 | 0.9944 | 0.9826 | 0.0037 | 0.0115 | 0.0589 | 0.0570 | 305.70 / 301.50 | 1.0139× |

Timing ratios compare single new runs with retained historical references;
small differences need matched repeated measurements before claiming a gain.
Manual visual, audio and synchronization review remains pending. This is
a focused diagnostic experiment, not a full 24-case qualification or an
automatic quality pass. No Qwen approval is used. External playback files
include both configurations and lossless audio. Validation scripts and
outputs stay outside the engine repository; evidence identifier:
`sparse28-48-cacheoff-20261116`.

#### Qwen frame-pair screening of the sparse 28–48 experiment

An external local Qwen3-VL-8B-Instruct review compared the same three pairs
using overlapping four-frame groups (stride three), covering all 124 native
RGB frames in both attachment orders. Settings and numerical quality scores
were withheld from the judge. Identical, alternating-white and heavy-blur
controls were repeated, with additional per-case identical controls.

Model revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Global controls passed: True.

| Case | Visual screening | Pass / fail / inconclusive windows | Order disagreements |
| --- | --- | --- | --- |
| sharp_transients-1 | pass | 41 / 0 / 0 | 0 |
| speech-1 | pass | 41 / 0 / 0 | 0 |
| motion_sync-1 | pass | 41 / 0 / 0 | 0 |

A window passes only when both orders pass; conflicting orders are
inconclusive. Concordant failures are flags for human adjudication, not
proof of a defect. These short frame groups do not replace full-speed
playback, audio listening or synchronization review. No final audiovisual
acceptance is conferred. Raw responses, controls and cited frames remain
external; no judge code or model weights are part of the repository.

### Focused sparse-window comparison: steps 24–48, caching OFF

A focused three-case experiment expanded sparse attention from steps 35–48
to 24–48, changing only `SD_H3_SOL_WARMUP_STEPS` from 34 to 23.
Caching remained OFF; beta −0.25, the binary, prompts, seeds, 50 steps,
resolution and other settings matched the retained sparse 35–48 references.
Generation used physical GPU 0. Trace checks required 550 sparse layer
dispatches (25 steps × 22 layers) and no cache reuse.

Native RGB comparisons measured all 124 frames. Audio columns report
absolute whole-clip RMS level differences for each native float PCM channel;
these are diagnostics and do not establish perceptual audio quality.

| Case / seed | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Max LPIPS | Left stereo diff (dB) | Right stereo diff (dB) | Reference / candidate generation (s) | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / 2836001 | 0.9886 | 0.9643 | 0.0066 | 0.0183 | 0.0369 | 0.0293 | 302.55 / 297.28 | 1.0177× |
| speech-1 / 2836005 | 0.9799 | 0.9662 | 0.0129 | 0.0220 | 0.1441 | 0.1575 | 303.63 / 298.84 | 1.0160× |
| motion_sync-1 / 2836017 | 0.9887 | 0.9743 | 0.0083 | 0.0183 | 0.3553 | 0.2946 | 305.70 / 300.19 | 1.0184× |

Timing ratios compare single new runs with retained historical references;
small differences need matched repeated measurements before claiming a gain.
Manual visual, audio and synchronization review remains pending. This is
a focused diagnostic experiment, not a full 24-case qualification or an
automatic quality pass. No Qwen approval is used. External playback files
include both configurations and lossless audio. Validation scripts and
outputs stay outside the engine repository; evidence identifier:
`sparse24-48-cacheoff-20261116`.

#### Qwen frame-pair screening of the sparse 24–48 experiment

An external local Qwen3-VL-8B-Instruct review compared the same three pairs
using overlapping four-frame groups (stride three), covering all 124 native
RGB frames in both attachment orders. Settings and numerical quality scores
were withheld from the judge. Identical, alternating-white and heavy-blur
controls were repeated, with additional per-case identical controls.

Model revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Global controls passed: True.

| Case | Visual screening | Pass / fail / inconclusive windows | Order disagreements |
| --- | --- | --- | --- |
| sharp_transients-1 | pass | 41 / 0 / 0 | 0 |
| speech-1 | inconclusive | 36 / 0 / 5 | 5 |
| motion_sync-1 | pass | 41 / 0 / 0 | 0 |

A window passes only when both orders pass; conflicting orders are
inconclusive. Concordant failures are flags for human adjudication, not
proof of a defect. These short frame groups do not replace full-speed
playback, audio listening or synchronization review. No final audiovisual
acceptance is conferred. Raw responses, controls and cited frames remain
external; no judge code or model weights are part of the repository.

#### Matched repeated timing of sparse 24–48 versus 35–48

The scissors case (seed 2836001, 864×480, 124 frames, 50 steps) was rerun
on GPU 0 using the same binary, caching OFF and beta −0.25. Native capture
and sparse trace logging were disabled equally. Each run used a fresh
process. One warm-up per configuration was excluded, followed by three
paired measurements in A/B, B/A, A/B order (A: 35–48; B: 24–48).
Decoded video and audio matched the respective quality outputs in all runs.

| Measure | Reference median (s) | Candidate median (s) | Speedup | Time reduction |
| --- | --- | --- | --- | --- |
| Generation | 303.42 | 298.23 | 1.0174× | 1.71% |
| Process wall | 304.56 | 299.40 | 1.0172× | 1.69% |

Generation samples (seconds): reference 303.83, 303.33, 303.42;
candidate 298.29, 298.23, 298.02.

These are three measurements on one case, not a full-cohort performance
qualification, persistent-context throughput result or significance test.
External evidence: `sparse24-48-cacheoff-20261116/matched-performance`.

### Focused sparse 24–48 comparison against dense references

Retained dense outputs were available for the same three prompts and seeds.
Reference receipts, native RGB/PCM hashes, execution traces, model arguments,
binary identity and output metadata were checked before reuse. Both profiles
used caching OFF. The reference had sparsity OFF; the candidate used sparse
steps 24–48 with beta −0.25. No new generations were needed for this comparison.
Candidate traces verified 550 sparse layer dispatches; dense traces had none.

Metrics compare all 124 native RGB frames. Audio columns are absolute
whole-clip RMS differences per channel, not perceptual audio quality scores.

| Case / seed | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Max LPIPS | Left stereo diff (dB) | Right stereo diff (dB) | Dense / sparse generation (s) | Historical timing ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / 2836001 | 0.9871 | 0.9656 | 0.0075 | 0.0186 | 0.0007 | 0.0473 | 305.64 / 297.28 | 1.0281× |
| speech-1 / 2836005 | 0.9797 | 0.9675 | 0.0129 | 0.0208 | 0.1118 | 0.1369 | 309.30 / 298.84 | 1.0350× |
| motion_sync-1 / 2836017 | 0.9888 | 0.9749 | 0.0082 | 0.0181 | 0.2326 | 0.1554 | 312.04 / 300.19 | 1.0395× |

Timing ratios compare retained single runs with historical dense references;
these are not matched repeated dense-versus-sparse timings. The separate matched benchmark compares sparse 35–48 with 24–48.
Manual visual, audio and synchronization review remains pending. This is
a focused diagnostic experiment, not a full 24-case qualification or an
automatic quality pass. Qwen results are reported separately. External playback files
include both configurations and lossless audio. Validation scripts and
outputs stay outside the engine repository; evidence identifier:
`sparse24-vs-dense-20261116`.

#### Qwen frame-pair screening of sparse 24–48 versus dense

An external local Qwen3-VL-8B-Instruct review compared the same three pairs
using overlapping four-frame groups (stride three), covering all 124 native
RGB frames in both attachment orders. Settings and numerical quality scores
were withheld from the judge. Identical, alternating-white and heavy-blur
controls were repeated, with additional per-case identical controls.

Model revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Global controls passed: True.

| Case | Visual screening | Pass / fail / inconclusive windows | Order disagreements |
| --- | --- | --- | --- |
| sharp_transients-1 | pass | 41 / 0 / 0 | 0 |
| speech-1 | inconclusive | 36 / 0 / 5 | 5 |
| motion_sync-1 | pass | 41 / 0 / 0 | 0 |

A window passes only when both orders pass; conflicting orders are
inconclusive. Concordant failures are flags for human adjudication, not
proof of a defect. These short frame groups do not replace full-speed
playback, audio listening or synchronization review. No final audiovisual
acceptance is conferred. Raw responses, controls and cited frames remain
external; no judge code or model weights are part of the repository.

Assistant visual inspection of speech frame 51 (2.125 s) in the encoded
preview pair found a golden bread-roll-like item in dense replaced by a dark
phone-like object in sparse 24–48. The drinking glass remained visible in both,
so the exact Qwen glass-replacement description was not corroborated. This
is a confirmed appearance difference in one frame, not a human quality verdict;
full playback and listening review remain pending.

#### Neutral-label sensitivity check of the speech result

After the user reported that the outputs looked the same, an external
diagnostic Qwen recheck used neutral A/B labels, independently swapping
label mapping and attachment order. The prompt explicitly separated
coherent appearance variation from perceptual quality defects. Human
feedback, earlier findings and metric values were withheld from the model.
The original inconclusive result and raw evidence remain unchanged.

Status: calibration-failed. Calibration passed: False.
Diagnostic result: `inconclusive`.
Window counts: `{}`.

This is a prompt/presentation sensitivity check on the same local model,
not an independent judge or a replacement qualification. No audio or
full-speed synchronization acceptance is conferred. Evidence is external:
`sparse24-vs-dense-20261116/qwen-neutral-speech-v1`.

### Selective cache pilot against retained dense and sparse 24–48 references

Only C was generated. A (dense/cache OFF) and B (sparse 24–48/cache OFF)
were retained outputs with matching prompts, seeds and settings, verified
against receipts and traces. C used sparse 24–48, beta −0.25, SOL cache
eligibility 18–48, threshold 0.05 and max consecutive reuse 1. Its binary
adds enforcement of the consecutive-reuse option to the SOL controller;
the retained SOL build differs only in that controller and its unit test.
Controller tests and address/undefined-behavior sanitizer checks passed.
No A/B regeneration or new cache-OFF output equivalence run was performed.

| Comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / C-vs-A | 0.9706 | 0.9356 | 0.0161 | 0.4837 | 0.4808 |
| sharp_transients-1 / C-vs-B | 0.9759 | 0.9424 | 0.0130 | 0.4829 | 0.5281 |

Actual reused calls (one-based): `[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]`.
Generation times: A 305.64s; B 297.28s; C 221.42s. These are exploratory single-run comparisons.

Qwen calibration passed: True.
sharp_transients-1-C-vs-A: pass; order disagreements: 0.
sharp_transients-1-C-vs-B: pass; order disagreements: 0.

Qwen reviewed all frames in four-frame overlapping windows, both orders.
RMS level differences do not measure perceived audio quality. Human listening,
full-speed playback and synchronization review remain pending. This pilot
does not qualify the broader cohort. External evidence identifier:
`selective-cache05-sparse24-20261116`; no judge scripts are in the codebase.

### Focused three-case selective cache validation

The completed scissors C pilot was reused. Only speech and basketball C
were newly generated. A (dense/cache OFF) and B (sparse 24–48/cache OFF)
were retained outputs with matching prompts, seeds and settings, verified
against receipts and traces. C used sparse 24–48, beta −0.25, SOL cache
eligibility 18–48, threshold 0.05 and max consecutive reuse 1. Its binary
adds enforcement of the consecutive-reuse option to the SOL controller;
the retained SOL build differs only in that controller and its unit test.
Controller tests and address/undefined-behavior sanitizer checks passed.
No A/B regeneration or new cache-OFF output equivalence run was performed.

| Comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / C-vs-A | 0.9706 | 0.9356 | 0.0161 | 0.4837 | 0.4808 |
| sharp_transients-1 / C-vs-B | 0.9759 | 0.9424 | 0.0130 | 0.4829 | 0.5281 |

Actual reused calls (one-based): `[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]`.
Generation times: A 305.64s; B 297.28s; C 221.42s. These are exploratory single-run comparisons.

| Comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| speech-1 / C-vs-A | 0.9078 | 0.8534 | 0.0576 | 0.2351 | 0.2712 |
| speech-1 / C-vs-B | 0.9169 | 0.8602 | 0.0528 | 0.1233 | 0.1343 |

Actual reused calls (one-based): `[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]`.
Generation times: A 309.30s; B 298.84s; C 220.55s. These are exploratory single-run comparisons.

| Comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| motion_sync-1 / C-vs-A | 0.9235 | 0.7448 | 0.0590 | 1.0318 | 1.0533 |
| motion_sync-1 / C-vs-B | 0.9287 | 0.7598 | 0.0560 | 0.7991 | 0.8978 |

Actual reused calls (one-based): `[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]`.
Generation times: A 312.04s; B 300.19s; C 225.50s. These are exploratory single-run comparisons.

Qwen calibration passed: True.
sharp_transients-1-C-vs-A: pass; order disagreements: 0.
sharp_transients-1-C-vs-B: pass; order disagreements: 0.
speech-1-C-vs-A: fail; order disagreements: 2.
speech-1-C-vs-B: pass; order disagreements: 0.
motion_sync-1-C-vs-A: pass; order disagreements: 0.
motion_sync-1-C-vs-B: pass; order disagreements: 0.

Qwen reviewed all frames in four-frame overlapping windows, both orders.
RMS level differences do not measure perceived audio quality. Human listening,
full-speed playback and synchronization review remain pending. This three-case experiment
does not qualify the broader cohort. External evidence identifier:
`selective-cache05-sparse24-20261116`; no judge scripts are in the codebase.

### Selective cache eligibility 6–48 compared with retained C, B and A

Only D was generated for the three focused cases. A is retained dense/cache
OFF; B is retained sparse 24–48/cache OFF; C is retained selective cache
18–48 plus sparse 24–48. D uses the same binary, prompts, seeds and settings
as C, changing only cache warmup from 17 to 5 calls (eligibility 6–48).
Both C and D use threshold 0.05 and maximum consecutive reuse 1; sparse
attention remains 24–48 with beta −0.25. Reference receipts and hashes
are checked before reuse. Logs verify the cache window, threshold and
fresh computation after each reuse. Generation and Qwen use GPU 0.

| Case / comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / D-vs-C | 0.7854 | 0.5653 | 0.1652 | 0.2625 | 0.2303 |
| sharp_transients-1 / D-vs-B | 0.7860 | 0.5620 | 0.1644 | 0.2204 | 0.2979 |
| sharp_transients-1 / D-vs-A | 0.7871 | 0.5617 | 0.1640 | 0.2212 | 0.2506 |

sharp_transients-1 actual D reused calls (one-based): `[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]`.
Generation times: A 305.64s; B 297.28s; C 221.42s; D 185.72s. These are exploratory single-run comparisons.

| Case / comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| speech-1 / D-vs-C | 0.7005 | 0.6028 | 0.2000 | 0.4160 | 0.5702 |
| speech-1 / D-vs-B | 0.7056 | 0.6266 | 0.1917 | 0.2927 | 0.4359 |
| speech-1 / D-vs-A | 0.7047 | 0.6288 | 0.1916 | 0.1809 | 0.2990 |

speech-1 actual D reused calls (one-based): `[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]`.
Generation times: A 309.30s; B 298.84s; C 220.55s; D 189.59s. These are exploratory single-run comparisons.

| Case / comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| motion_sync-1 / D-vs-C | 0.6482 | 0.4468 | 0.2666 | 1.5012 | 1.4609 |
| motion_sync-1 / D-vs-B | 0.6459 | 0.4458 | 0.2689 | 2.3003 | 2.3588 |
| motion_sync-1 / D-vs-A | 0.6440 | 0.4400 | 0.2694 | 2.5329 | 2.5142 |

motion_sync-1 actual D reused calls (one-based): `[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]`.
Generation times: A 312.04s; B 300.19s; C 225.50s; D 190.47s. These are exploratory single-run comparisons.

Qwen calibration passed: True.
sharp_transients-1-D-vs-C: fail; order disagreements: 8.
sharp_transients-1-D-vs-B: fail; order disagreements: 10.
sharp_transients-1-D-vs-A: fail; order disagreements: 8.
speech-1-D-vs-C: fail; order disagreements: 9.
speech-1-D-vs-B: fail; order disagreements: 10.
speech-1-D-vs-A: fail; order disagreements: 7.
motion_sync-1-D-vs-C: fail; order disagreements: 20.
motion_sync-1-D-vs-B: fail; order disagreements: 17.
motion_sync-1-D-vs-A: fail; order disagreements: 17.

Qwen screened all 124 frames in overlapping four-frame windows, both
attachment orders. RMS level differences do not establish perceptual audio
quality. Manual listening, full-speed playback and synchronization review
remain pending. Three focused cases do not qualify the full 24-case cohort.
Validation code and evidence remain external:
`selective-cache6-05-sparse24-20261116`.

### Cache 18–48 at threshold 0.05 without a consecutive-reuse cap

Only E was generated for the three focused cases. A is retained dense/cache
OFF; B is retained sparse 24–48/cache OFF; C is retained selective cache
18–48 plus sparse 24–48. E uses the same binary, prompts, seeds and settings
as C, changing only maximum consecutive reuse from 1 to 0 (uncapped).
Both retain cache eligibility 18–48 and threshold 0.05, with sparse
attention 24–48 and beta −0.25. Consecutive reuse is permitted while
accumulated change remains below the threshold; reaching it triggers fresh
computation. Trace checks reconstruct the accumulated joint-L1 change,
verify threshold-triggered refresh and record the longest reuse streak.
Reference receipts and hashes are checked before reuse. GPU work uses GPU 0.

| Case / comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| sharp_transients-1 / E-vs-C | 0.9467 | 0.8631 | 0.0353 | 0.4447 | 0.4719 |
| sharp_transients-1 / E-vs-B | 0.9309 | 0.8430 | 0.0443 | 0.9277 | 1.0000 |
| sharp_transients-1 / E-vs-A | 0.9289 | 0.8335 | 0.0458 | 0.9284 | 0.9527 |

sharp_transients-1 actual E reused calls (one-based): `[18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 37, 38, 40, 41, 43, 45]`.
Longest consecutive reuse streak: 3.
Generation times: A 305.64s; B 297.28s; C 221.42s; E 190.19s. These are exploratory single-run comparisons.

| Case / comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| speech-1 / E-vs-C | 0.9369 | 0.9007 | 0.0344 | 1.1037 | 1.1406 |
| speech-1 / E-vs-B | 0.8835 | 0.8090 | 0.0717 | 1.2270 | 1.2749 |
| speech-1 / E-vs-A | 0.8760 | 0.8059 | 0.0752 | 1.3387 | 1.4117 |

speech-1 actual E reused calls (one-based): `[18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 37, 38, 40, 41, 43, 45]`.
Longest consecutive reuse streak: 3.
Generation times: A 309.30s; B 298.84s; C 220.55s; E 191.92s. These are exploratory single-run comparisons.

| Case / comparison | Mean SSIM | Worst-frame SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| motion_sync-1 / E-vs-C | 0.9580 | 0.8277 | 0.0327 | 1.7061 | 1.6964 |
| motion_sync-1 / E-vs-B | 0.9103 | 0.7256 | 0.0640 | 2.5052 | 2.5943 |
| motion_sync-1 / E-vs-A | 0.9082 | 0.7151 | 0.0658 | 2.7379 | 2.7497 |

motion_sync-1 actual E reused calls (one-based): `[18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 37, 38, 40, 41, 43, 45]`.
Longest consecutive reuse streak: 3.
Generation times: A 312.04s; B 300.19s; C 225.50s; E 193.07s. These are exploratory single-run comparisons.

Qwen calibration passed: True.
sharp_transients-1-E-vs-C: pass; order disagreements: 0.
sharp_transients-1-E-vs-B: pass; order disagreements: 0.
sharp_transients-1-E-vs-A: pass; order disagreements: 0.
speech-1-E-vs-C: pass; order disagreements: 0.
speech-1-E-vs-B: inconclusive; order disagreements: 1.
speech-1-E-vs-A: fail; order disagreements: 7.
motion_sync-1-E-vs-C: pass; order disagreements: 0.
motion_sync-1-E-vs-B: pass; order disagreements: 0.
motion_sync-1-E-vs-A: pass; order disagreements: 0.

Qwen screened all 124 frames in overlapping four-frame windows, both
attachment orders. RMS level differences do not establish perceptual audio
quality. Manual listening, full-speed playback and synchronization review
remain pending. Three focused cases do not qualify the full 24-case cohort.
Validation code and evidence remain external:
`selective-cache18-05-unbounded-sparse24-20261116`.

### Broad validation of uncapped cache 18–48 with sparse attention 24–48

The 24-case cohort uses matching prompts and seeds, 50 denoising steps and
124 frames at 24 fps: 18 cases at 864×480 and six at 1344×768. E enables
SOL cache eligibility 18–48 at accumulated-change threshold 0.05 without
a consecutive-reuse cap, with sparse attention 24–48 and beta −0.25.
A is dense with caching disabled; B uses sparse attention 24–48 with
caching disabled. All dense references and the three focused E/B cases
were retained after receipt/hash checks; the remaining 21 E and 21 B
outputs were generated. Generation traces verify reuse eligibility,
threshold accumulation and sparse calls. This is the original development
cohort, not held-out qualification.

| Comparison (24 cases) | Mean SSIM range | Mean LPIPS range | Maximum left stereo diff (dB) | Maximum right stereo diff (dB) |
| --- | --- | --- | --- | --- |
| E versus A | 0.8760–0.9825 | 0.0121–0.0752 | 8.0424 | 8.0208 |
| E versus B | 0.8835–0.9828 | 0.0104–0.0730 | 7.9555 | 7.9215 |

External local Qwen3-VL-8B-Instruct screening uses all native RGB frames
in overlapping four-frame windows, stride three, with both attachment
orders. Controls include identical frames, alternating white frames and
heavy blur. Numerical similarity and visual verdicts are diagnostic;
the model does not assess audio or certify full-speed temporal quality.

Qwen execution status: `completed-visual-screening-human-audio-review-pending`.
E versus A visual screening: pass 21, fail 2, inconclusive 1.
E versus B visual screening: pass 22, fail 1, inconclusive 1.
Controls matching the expected outcome: 100/100; invalid responses: 0.

Repeated performance status: `completed`.
One representative 864x480 case, same executable for A/B/E, fresh process per run. One excluded warmup per configuration and three measured rounds with rotated order. Native capture and sparse tracing disabled equally. Cache decision logging remains inherent to E. Decode checks verify output equals corresponding quality output. Not full-cohort matched performance or persistent-context throughput.
Median generation time: A 312.17s, B 299.83s, E 194.16s.
E speedup: 1.608× versus dense A and 1.544× versus sparse-only B.

Broad-cohort generation-time medians, separated by resolution, are shown below.
These aggregate one run per input, including retained references; they are
exploratory and distinct from the repeated matched benchmark above.

| Resolution | Cases | A median (s) | B median (s) | E median (s) | A/E median ratio | B/E median ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 864×480 | 18 | 312.48 | 302.74 | 191.98 | 1.628× | 1.577× |
| 1344×768 | 6 | 1144.04 | 1112.92 | 704.55 | 1.624× | 1.580× |

Left/right differences measure whole-clip RMS level change, not perceptual
audio quality. Manual visual review accepted the flagged speech-1, speech-2
and ambience_detail_camera-4 outputs. Manual review rejected music_stereo-3's
E audio; this prevented overall acceptance of the original-seed cohort. The
new-seed review and subsequent configuration decision are recorded below;
they do not change the original result. External validation
scripts, judge code and generated artifacts are excluded from the engine
repository. External evidence identifier: `selective-E-broad-20261116`.

### Consecutive-reuse limit: music_stereo-3 audio follow-up

The failing wind-chime input (seed 2836011, 864×480, 124 frames, 50 steps)
was regenerated as C using E's executable and settings, changing only
`SD_H3_CACHE_MAX_CONSECUTIVE` from 0 (unlimited) to 1. Both use cache
eligibility 18–48, threshold 0.05, sparse attention 24–48 and beta −0.25.
Existing A/B/E outputs were retained after hash and trace verification.
GPU work used GPU 0. C reused 14 isolated steps, 18, 20, …, 44; E reused
20 steps with a maximum streak of three. This experiment isolates the
reuse-cap setting, which changes both the reuse count and refresh schedule.

| Comparison | Mean SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- |
| C versus A | 0.9891 | 0.0083 | 1.3044 | 1.2469 |
| C versus B | 0.9848 | 0.0123 | 1.2176 | 1.1476 |
| E versus C | 0.9778 | 0.0155 | 6.7380 | 6.7739 |
| E versus A | 0.9716 | 0.0193 | 8.0424 | 8.0208 |

C remains quieter than A, but the level loss is substantially smaller than E's.
Single-run generation times were A 313.90s, B 303.49s, C 221.80s and E 192.32s;
C took 15.3% longer than E. These are exploratory timings, not a repeated
matched benchmark. No 1344×768 C result was generated in this follow-up.

The same external Qwen3-VL method screened all 124 native frames in overlapping
four-frame windows, stride three, both presentation orders. C versus A,
C versus B and E versus C all passed visual screening; 10/10 controls matched
the expected outcome and none of the 256 responses was invalid. This judge
does not assess audio. C requires manual listening review and is not a
qualified fix for the entire cohort. Native WAV and preview videos were
exported without loudness normalization. Validation scripts and raw evidence
remain outside the repository: `music3-cache-cap1-20261117`.

### New-seed review and preferred E configuration

The same music_stereo-3 wind-chime prompt was regenerated with seed 2837011
at the same 864×480 resolution, 124 frames and 50 steps. A and E used the
same new seed and executable; A disabled both caching and sparse attention.
E retained sparse attention 24–48, beta −0.25, cache eligibility 18–48,
threshold 0.05 and no consecutive-reuse cap. Its 20 reused calls matched
the original case's schedule.

| Seed | E versus A mean SSIM | Mean LPIPS | Left stereo diff (dB) | Right stereo diff (dB) | Manual review |
| --- | --- | --- | --- | --- | --- |
| 2836011 | 0.9716 | 0.0193 | 8.0424 | 8.0208 | Audio rejected |
| 2837011 | 0.9757 | 0.0146 | 1.2630 | 1.4108 | Video and audio accepted |

The new-seed single-run generation times were A 313.88s and E 187.95s.
Qwen visual screening remained inconclusive: 36 of 41 windows passed and
five had presentation-order disagreements, with all six calibration controls
passing. Manual acceptance is separate from this model verdict. Stereo RMS
differences remain level diagnostics, not perceptual acceptance thresholds.

Following this review, E was selected as the preferred configuration for
continued performance work, with output review and C available as a fallback.
This is a user-reviewed deployment choice, not a universal quality guarantee
or a change to the runtime's opt-in defaults. The original-seed regression
remains documented; changing the seed does not fix that particular output.
External evidence identifier: `music3-E-seed2837011-20261117`. Validation and
judge code remain outside the engine and backend repositories.

### Output-preserving attention optimization prototype

All timing comparisons below use the integrated E benchmark stack, including
the previously documented PR #61 packed SwiGLU and QKV-view dependency. Its
external control helpers are not added to this engine PR. Existing baseline
fusions are held constant across configurations and are not counted as new
improvements. These timings do not describe an uncombined engine checkout.

An external CUDA prototype evaluates 128 query rows per dense-attention tile,
then recomputes affected Stream-K boundaries with the original 64-row schedule.
This preserves the original partial reductions. Sparse attention and E's cache
settings remain unchanged. A smaller backend implementation is now prepared
behind `GGML_CUDA_H3_FATTN_WIDE=1`; it remains opt-in and passed separate
integration validation. Its completed integration results follow the external
prototype results below.

All 27 microbenchmark variants (nine shapes/layouts, baseline, candidate OFF
and candidate ON) produced byte-identical results for both input seeds.
Uneven sequence lengths, padded layouts, masked attention and alternate head
counts were included. CUDA memcheck and racecheck passed. The measured
attention speedups were 1.242× at 15,429 tokens and 1.176× at 37,740 tokens.

Full-generation checks compared every native RGB and float-PCM byte with
retained E output. Both cases below matched exactly, including their cache
reuse schedules. Consequently, their existing dense-reference metrics and
review outcomes carry over; this is not a new Qwen or human approval.

| Case / seed | Resolution | Matched E (s) | Corrected prototype (s) | Time reduction | Native video/audio versus E |
| --- | --- | --- | --- | --- | --- |
| music_stereo-3 / 2837011 | 864×480 | 191.74 | 175.41 | 8.52% | Byte-identical |
| speech-4 / 2836008 | 1344×768 | 699.34 | 648.62 | 7.25% | Byte-identical |

The low-resolution figures are medians of three fresh-process measurements per
configuration in rotated order; E ranged from 188.94 to 191.78 seconds and the
candidate from 175.38 to 175.81 seconds. The high-resolution figures are one
matched pair, not repeated estimates. Both configurations used the same
executable with the optimization off/on, fixed input and seed, 50 steps,
124 frames at 24 fps, and the same native-capture/tracing settings. Every
timed output's decoded video and audio hashes matched retained E. These
figures supersede the earlier single-run observations for this prototype;
they do not establish full-cohort speedup. Earlier uncorrected 128×64 and
128×128 candidates missed the frozen incremental SSIM screen against E;
Qwen rejected the former and remained inconclusive for the latter. Their
results remain recorded rather than being treated as accepted optimizations.
Experimental harnesses, benchmark artifacts and judge code stay outside the PRs. Evidence
identifiers: `attention-tiles-probe-20261117` and
`attention-seam-probe-20261117`.

The smaller backend implementation keeps the existing H3 shape, layout and
precision checks, dispatches sparse attention first, and requires compatible
64/128-row configurations and a 2:1 occupancy relationship. Unsupported cases
retain the existing path; builds with CUDA graph support also require graphs
to be disabled before selecting the wider kernel. Native tests cover dispatch,
uneven lengths, repeated inputs and the prefetch-disabled fallback. The combined
working implementation compiled and passed all 30 operator benchmark variants,
CUDA memcheck/racecheck, and native wide-enabled, wide-disabled and
prefetch-disabled checks. Both full examples reproduced E's native RGB and
float PCM exactly, with identical cache-reuse schedules. Matched timing for
this smaller implementation is complete, as reported below. External evidence identifier:
`chain-attention-integration-20261117`.

The next incremental candidate uses 128 rather than 256 threads in the existing
128-channel fused normalization kernel. It is evaluated on top of the faster
attention prototype, preserving E's sparse/cache policy. Both native example
outputs matched E, and the normalization microbenchmark improved approximately
1.51× at each resolution. The complete cumulative comparison used one executable
with each optimization enabled in turn:

| Resolution | E (s) | + Corrected attention (s) | + Compact normalization (s) | Normalization incremental time reduction | Combined time reduction versus E |
| --- | --- | --- | --- | --- | --- |
| 864×480 | 191.34 | 175.15 | 173.75 | 0.80% | 9.19% |
| 1344×768 | 699.79 | 648.33 | 645.92 | 0.37% | 7.70% |

The low-resolution values are medians of three rotated measurements per
configuration; the high-resolution values are one matched triple. Every timed
output's decoded video and audio matched E. The kernel gain is not an end-to-end
speedup claim. Evidence identifier: `chain-norm-compact-20261117`.

The matching compact kernel is included in the backend working tree behind
`GGML_CUDA_H3_NORM_ROPE_COMPACT=1`, restricted to 128-column SM120 inputs.
Other inputs retain the existing kernel. The combined integration build passed
native and full-generation identity checks. Its matched timing and complete
graph-enabled CUDA build passed. All 12 selected CTest cases passed, including
compact selection, fusion-disabled fallback, wide-attention fallback and actual
graph replay. External evidence identifier: `chain-attention-fullbuild-20261117`.

The final integrated source was measured separately from the prototype above,
using one matched pair per resolution, with both new optimizations enabled
together on E. Inputs, seeds, 50 steps and 124 frames at 24 fps were held fixed.

| Resolution | E (s) | E + attention + compact normalization (s) | Time reduction |
| --- | --- | --- | --- |
| 864×480 | 191.26 | 174.10 | 8.97% |
| 1344×768 | 701.66 | 645.61 | 7.99% |

Every timed decoded video and audio hash matched E. These single matched pairs
verify the combined source integration; they are separate measurements from
the prototype medians and do not establish full-cohort performance.

### Video-VAE FlashAttention candidate

`SD_H3_VAE_FLASH_ATTN=1` opts CUDA video decoding into the existing
FlashAttention helper. It preserves the global flash-attention setting and
leaves encoding, the text encoder, the denoiser and E's sparse/cache policy
unchanged. The helper retains its existing fallback when an attention operation
is unsupported. The local source toggle passed automated pilot validation.
New human playback review is pending; this is not a full-cohort qualification.

The instrumented prototype used the validated attention-plus-normalization
backend. Both fixed prompt/seed examples passed the incremental visual screen
against E, whose output is byte-identical to that backend on these examples:

| Resolution | Mean SSIM vs E | Worst-frame SSIM vs E | LPIPS vs E | Left stereo diff (dB) | Right stereo diff (dB) |
| --- | --- | --- | --- | --- | --- |
| 864×480 | 0.999843 | 0.999817 | 0.0000323 | 0.000000 | 0.000000 |
| 1344×768 | 0.999816 | 0.999768 | 0.0000491 | 0.000000 | 0.000000 |

All native float PCM samples and cache-reuse schedules matched E. Video changes
are confined to VAE attention rounding. The experiment-specific screen requires
mean SSIM at least 0.995 and LPIPS at most 0.005 against E; these are not universal
perceptual acceptance thresholds. One matched low-resolution triple measured
E at 190.62 s, E plus attention/normalization at 174.20 s, and the VAE candidate
at 169.33 s: 2.80% incremental and 11.17% cumulative time reduction. All three
decoded outputs matched their quality runs. These are single observations from
one executable, separate from earlier integration benchmarks. The completed
high-resolution triple measured 702.04 → 645.30 → 633.45 s, a 1.84%
incremental reduction and 9.77% cumulative reduction against E. All six timed
outputs matched their corresponding quality runs. Local Qwen3-VL screening
passed both comparisons against E, covering all 124 native frames in overlapping
four-frame windows and both attachment orders; its controls passed. Against dense
A, speech passed and the wind-chime comparison was inconclusive (five window
order disagreements). The judge does not assess audio. Existing decoder-block
regressions passed on CPU and CUDA, including flash/manual attention and batched
layouts. Minimal-source OFF/ON generation checks passed on the fixed 864×480
example: OFF reproduced E and ON reproduced the instrumented prototype, including
every native RGB/PCM byte and the cache schedule. The 1344×768 metrics and timing
above are from the instrumented prototype. New human playback review is pending.
Experimental graph-count
assertions and all judge/benchmark code stay outside the repositories.
External evidence identifiers: `chain-vae-flash-20261117` and
`chain-vae-flash-source-20261117`.
