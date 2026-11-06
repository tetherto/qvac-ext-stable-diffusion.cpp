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
