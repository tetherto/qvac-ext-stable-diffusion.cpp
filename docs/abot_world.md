# ABot-World

[ABot-World-0-5B-LF](https://huggingface.co/acvlab/ABot-World-0-5B-LF) (AMAP CV
Lab, Apache-2.0) is a **causal, interactive** derivative of Wan2.2-TI2V-5B: it
generates an explorable world block-by-block, driven by keyboard actions, using
a rolling KV cache and a 4-step distilled sampler.

## Status in this repo

**Supported:** model detection, architecture instantiation, and full weight
loading (F16 / Q8_0 GGUF or safetensors), including the extra
`act_control_adapter.*` action-conditioning tensors. The model reuses the
Wan2.2-TI2V latent space (48ch, 16x) and VAE.

**Not yet supported:** actual generation. ABot-World cannot run through the
batch paths — those are bidirectional/one-shot, whereas ABot needs a stateful
causal session (KV cache, per-block action injection, its distilled 4-step
schedule). Both batch entrypoints reject ABot models with a clear error at the
shared `GenerationRequest` stage, and the capability queries
(`sd_ctx_supports_image_generation` / `sd_ctx_supports_video_generation`)
report `false`, so front-ends pre-screen it consistently. The interactive
session API is a planned follow-up.

## Detection

A GGUF/safetensors checkpoint is classified `VERSION_ABOT_WORLD` when it is a
Wan model that also contains `model.diffusion_model.act_control_adapter.conv.weight`.

## GGUF conversion

The DiT checkpoint converts to GGUF with standard tooling. Notes:
- `patch_embedding.weight` is a Conv3d `[3072,48,1,2,2]`; stored 4D as
  `[147456,1,2,2]` (ggml/ComfyUI-GGUF Conv3d convention).
- The 6 `act_control_adapter.*` tensors are kept F16 (convs); everything else
  follows the usual F16/Q8_0 split.
- Q8_0 (5.86 GB) vs F16 (10.55 GB): validated near-lossless — per-block latent
  cosine >= 0.9998 vs the bf16 reference over golden rollouts.

## Validation

`script/validate_abot_world.sh` loads an ABot GGUF, asserts detection + clean
tensor load + the guarded rejection, and (optionally, with `WAN_*` env vars)
confirms a stock Wan model is unaffected.
