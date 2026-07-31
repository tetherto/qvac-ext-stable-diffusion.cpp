# ABot-World

[ABot-World-0-5B-LF](https://huggingface.co/acvlab/ABot-World-0-5B-LF) (AMAP CV
Lab, Apache-2.0) is a **causal, interactive** derivative of Wan2.2-TI2V-5B: it
generates an explorable world block-by-block, driven by keyboard actions, using
a rolling attention window and a 4-step distilled sampler.

## Status in this repo

**Supported:**

- Model detection, architecture instantiation, and full weight loading
  (F16 / Q8_0 GGUF or safetensors), including the extra `act_control_adapter.*`
  action-conditioning tensors. The model reuses the Wan2.2-TI2V latent space
  (48ch, 16x) and VAE.
- **Interactive walk** via the session C API
  (`sd_abot_session_{params_init,new,step,frames_free,free}`): a standalone
  session owns the DiT + taehv decoder + a scene pack; each
  `step(action_mask)` (bits 0..7 = W,A,S,D,I,J,K,L held) generates one latent
  block and returns its decoded RGB frames. Opt-in per-layer KV cache
  (`ABOT_KV_CACHE=1`) captures history K/V once per finalized block instead of
  recomputing it every denoise step; `ABOT_PROF=1` prints per-stage timings.
- **Native scene creation** (`sd_abot_scene_create`): builds the scene pack a
  session walks in from a prompt + first-frame image, on-device — umT5-XXL
  encodes the prompt, the Wan2.2 VAE encodes the image — replacing the
  reference implementation's offline PyTorch extraction. Packs without an
  image (text-only, `first_frame_mask = 0`) are supported at the format level
  but gated: the distilled checkpoint cannot bootstrap a coherent first frame
  from noise, so front-ends should require an image until a T2V-capable
  checkpoint ships.

**Not supported:** the batch `generate_image()`/`generate_video()` paths —
those are one-shot, whereas ABot needs the stateful causal session. Both batch
entrypoints reject ABot models with an error pointing at the session API, and
the capability queries (`sd_ctx_supports_image_generation` /
`sd_ctx_supports_video_generation`) report `false`, so front-ends pre-screen
it consistently.

## Detection

A GGUF/safetensors checkpoint is classified `VERSION_ABOT_WORLD` when it is a
Wan model that also contains `model.diffusion_model.act_control_adapter.conv.weight`.

## Models

| role | file | notes |
|---|---|---|
| DiT (walk) | ABot-World DiT GGUF | F16 or Q8_0 |
| pixel decoder (walk) | taew2_2 GGUF | streaming taehv decoder; the walk uses the decoder half only ("unknown tensor tae.encoder.*" load warnings are expected) |
| prompt encoder (scene creation) | umT5-XXL GGUF/safetensors | F16 or Q8_0 |
| first-frame encoder (scene creation) | Wan2.2 VAE GGUF/safetensors | F16 (a Q8_0 conversion is a no-op: conv weights stay F16) |

## GGUF conversion

The DiT checkpoint converts to GGUF with standard tooling
(`sd-cli -M convert`). Notes:
- `patch_embedding.weight` is a Conv3d `[3072,48,1,2,2]`; stored 4D as
  `[147456,1,2,2]` (ggml/ComfyUI-GGUF Conv3d convention).
- The 6 `act_control_adapter.*` tensors are kept F16 (convs); everything else
  follows the usual F16/Q8_0 split.
- Q8_0 (5.86 GB) vs F16 (10.55 GB): validated near-lossless — per-block latent
  cosine >= 0.9998 vs the bf16 reference over golden rollouts.

## Validation

- `script/validate_abot_world.sh` loads an ABot GGUF, asserts detection +
  clean tensor load + the guarded batch rejection, and (optionally, with
  `WAN_*` env vars) confirms a stock Wan model is unaffected.
- The `examples/abot-{parity,walk,session}` harnesses (static builds only)
  validate against the PyTorch reference: per-step parity across all attention
  regimes, full-chain golden-replay walks, taehv decode parity, and
  `--mode create-scene` native scene packs gated against reference
  extractions.
