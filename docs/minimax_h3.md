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
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten’s energetic surfing." --cfg-scale 1.0 -v -W 864 -H 480 --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/t2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

Omitting `--audio-vae` still runs the joint diffusion model but produces video without a
decoded audio track.

## First/last-frame conditioning

Add `--init-img` for I2VA, or both `--init-img` and `--end-img` for FL2VA:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "a lovely cat" -i ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/i2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

## Reference-to-audio-video conditioning

Ref2VA accepts any combination of reference images, reference videos, paired
video soundtracks, and standalone audio references:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_ref2va_pruned-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "Use the cat from <Picture 1> as the main character. Keep the cat’s appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat’s face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat’s head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat’s fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat’s identity unchanged." -r ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --offload-to-cpu --rng cpu --fps 24 --video-frames 56
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

Reference images keep their aspect ratio and are only downscaled when they
exceed the configured limit; neither mode scales a reference up. The limit is
selected by `ref_image_size`. The default, `match`, uses the pixel area of the
requested generation canvas. Pass `--ref-image-args "ref_image_size=max"` (or
`ref_image_size` in `sd_vid_gen_params_t.ref_image_args`) to use a 2048 px short
edge instead, which preserves more identity detail at a higher sampling cost:
reference latents are attended at every sampling step.

Whichever limit applies, each axis is then snapped to the nearest 32-pixel
multiple. An axis of at least 16 pixels can therefore grow by up to 16 pixels
past the scaled size (a 48-pixel axis becomes 64), and references are never
narrower than 32 pixels.
`resize_before_vae=false` is not supported here; MiniMax-H3 keeps references on
the 32-pixel grid and reports a warning.

The C API exposes the same inputs through `ref_images`, `ref_videos`, and
`ref_audios` in `sd_vid_gen_params_t`. Each `sd_ref_video_t` supplies its own
frame rate and optional soundtrack; non-24-fps inputs are resampled internally.

## Shape and runtime notes

- Width and height are aligned upward to a multiple of 32.
- Frame count is aligned upward to the `17k + 5` grid, with a minimum of 5.
- MiniMax-H3 runs at 24 fps; another requested value is overridden.
- The default video flow shift is 12. The audio stream is mapped internally to
  its shift of 3, so the regular samplers can operate on the packed AV latent.

## Strix ROCm (gfx1151)

Build with the [ROCm instructions](build.md#build-with-hipblas) and target `gfx1151` for
Strix Halo. Use the GGML revision pinned by this repository: its HIP backend
contains the Strix attention precision correction and scale/residual fusion.
An externally supplied GGML build must contain the same changes.

The attention correction uses FP32 for weighted-value accumulation, rescaling
and combination on the affected unmasked, head-128 path when F32 precision is
requested. It preserves the existing Q/K and probability arithmetic. The
scale/residual fusion preserves the separate FP32 multiply and add roundings
for the supported H3 layout. Both paths select gfx1151 at runtime and leave
unsupported configurations on the existing path.

### TAE-H3 and EasyCache

TAE-H3 and EasyCache are optional, existing features. TAE-H3 replaces the video
decoder with a lightweight approximation; it does not replace the diffusion
GGUF or the audio VAE. Use the H3 decoder weights, not the SD, Wan or Hunyuan
TAE weights.

The tested TAE-H3 checkpoint is
[`taeh3.safetensors` at revision `62f7591`](https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors)
(22,709,752 bytes; SHA-256
`4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13`).
Save it as `models/taeh3.safetensors` for the example below. `--tae` is an alias
of `--taesd`; do not use `--taesd-preview-only` when replacing final decoding.

EasyCache reuses model results within the sampling schedule. The tested policy
uses `threshold=0.20,start=0.20,end=0.80`. A 20-step request still has 20 nominal
sampling steps, but it may execute fewer model evaluations. The number of
reuses depends on the input; it is not a fixed-step distillation schedule.

```sh
./build/bin/sd-cli -M vid_gen \
  --diffusion-model models/minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --llm models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae models/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae models/minimax_h3_audio_vae_fp32.safetensors \
  --tae models/taeh3.safetensors \
  --backend ROCm0 --params-backend ROCm0 --flash-attn on \
  --cache-mode easycache \
  --cache-option "threshold=0.20,start=0.20,end=0.80" \
  --sampling-method euler --steps 20 --cfg-scale 1 --guidance 7 \
  --seed 424242 -W 864 -H 480 --video-frames 124 --fps 24 -b 1 \
  -p "A small orange paper boat floating slowly on calm water at dusk, with a cliff in the background; gentle somber solo piano music, no speech." \
  -o h3-strix.avi
```

Use the device name reported by your build if it differs from `ROCm0`.
For full-VAE decoding, omit `--tae` and retain `--vae`. To disable EasyCache,
omit `--cache-mode` and `--cache-option`. These options remain opt-in; neither
approximation is enabled merely by selecting Strix.

### Recorded comparison

One 864x480, 124-frame example on Radeon 8060S/gfx1151 with the prompt and
seed above produced the following initialization-plus-generation times:

| Configuration | Time | Reduction versus preceding row |
|---|---:|---:|
| Corrected attention, full VAE, no cache | 26m57.3s | Reference |
| Above + TAE-H3 | 26m09.0s | 48.3s (2.99%) |
| Above + EasyCache | 17m16.4s | 8m52.6s (33.94%) |
| Above + scale/residual fusion | 16m52.4s | 24.1s (2.32%) |

The total observed reduction is 37.4%. These are historical single clean runs,
not repeated paired speedup estimates or a promise for other inputs. The
earlier approximately 28-minute result used incorrect FP16 weighted-value
accumulation and is excluded from this comparison.

Measurements used a separate quiet harness with profiling, progress logging
and telemetry disabled; native output export, encoding and validation followed
the timer. The CLI command above reproduces the generation configuration, not
that timing harness. The exact model-forward count was not instrumented in the
corrected clean cached run.

TAE-H3 changes video detail, and EasyCache changes both video and stereo audio.
For this example, final output versus the corrected full-VAE reference had mean
frame SSIM 0.8234, LPIPS 0.1519, and left/right audio RMSE 0.01861/0.01908.
The scale/residual fusion alone preserved all 124 native RGB frames and both
audio channels byte-for-byte versus TAE-H3 + EasyCache. Compare final video and
listen to both channels for the intended workload before choosing the faster
configuration; these measurements do not qualify every prompt or seed.

## Apple Silicon (Metal)

EasyCache is a measured accelerator on Apple M-series GPUs for the same
full-VAE Ref2VA workload. TAE-H3 is an orthogonal option (not used in the
comparisons here); its lightweight video decoder trades additional quality
for additional speed. The Metal partial-RoPE fast path for H3 ships in the
engine; it is backend-gated and leaves other backends unchanged.

The measured EasyCache policy on Metal is
`threshold=0.30,start=0.15,end=0.85`. The more conservative Strix policy
`threshold=0.20,start=0.20,end=0.80` also works on Metal and preserves
more of the no-cache output; the recorded comparison below lists both.
As with Strix, a 20-step request keeps 20 nominal sampling steps but may
execute fewer model evaluations.

```sh
./build/bin/sd-cli -M vid_gen \
  --diffusion-model models/minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --llm models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae models/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae models/minimax_h3_audio_vae_fp32.safetensors \
  --backend MTL0 --params-backend MTL0 --flash-attn on \
  --cache-mode easycache \
  --cache-option "threshold=0.30,start=0.15,end=0.85" \
  --sampling-method euler --steps 20 --cfg-scale 1 --guidance 7 \
  --seed 424242 -W 864 -H 480 --video-frames 124 --fps 24 -b 1 \
  -p "A small orange paper boat floating slowly on calm water at dusk, with a cliff in the background; gentle somber solo piano music, no speech." \
  -o h3-mac.mp4
```

Use the device name reported by your build if it differs from `MTL0`.
To run without EasyCache, omit `--cache-mode` and `--cache-option`;
EasyCache remains opt-in.

### Recorded comparison

One 864x480, 124-frame example on Apple M3 Ultra (60 GPU cores, 96 GB) with
the prompt and seed above, full video and audio VAE decoding, no TAE,
produced the following generation times:

| Configuration | Generation | Reduction versus baseline |
|---|---:|---:|
| Baseline (no cache) | 23m05.2s | Reference |
| EasyCache `0.20,0.20,0.80` | 14m53.3s | 8m11.9s (35.52%) |
| EasyCache `0.30,0.15,0.85` | 12m45.1s | 10m20.1s (44.77%) |

These are historical single clean runs on one host, not repeated paired
speedup estimates or a guarantee for other inputs. Timers exclude output
export, encoding and validation; profiling, progress logging and telemetry
were disabled. The exact model-forward count was not instrumented.

EasyCache changes both video and stereo audio. For this example, final
output versus the no-cache baseline had:

| Policy | mean frame SSIM | mean LPIPS (AlexNet) | L/R audio RMSE |
|---|---:|---:|---|
| `0.20,0.20,0.80` | 0.9942 | 0.0326 | 0.01555 / 0.01577 |
| `0.30,0.15,0.85` | 0.9900 | 0.0607 | 0.02277 / 0.02327 |

Per-frame LPIPS rises toward the end of the clip under both policies and
more noticeably under `0.30,0.15,0.85`; isolated audio samples deviate up
to about 0.22 under that policy against a peak absolute signal near 0.37,
so headphones are recommended when evaluating audio before shipping. The
more conservative policy halves the audio deviation. Compare final video
and listen to both channels for the intended workload before choosing;
these measurements do not qualify every prompt or seed.
