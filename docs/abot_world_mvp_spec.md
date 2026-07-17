# ABot-World walk MVP — implementation spec

Goal: a native, Python-free minimal "walk" — load the DiT (GGUF) + taehv +
a precomputed **scene pack**, drive it with a short scripted action sequence,
and write an mp4 — validated in CI against a committed reference.

This spec is written so another dev can implement the denoise core with a
concrete parity target. Status: **E1 (load/detect) merged; denoise core below is
the remaining work.**

## Inputs (no T5 / no VAE encoder on device)

The scene pack (produced offline by `ABot-World/tools/extract_scene.py`, a
safetensors file `sd.cpp` already reads) contains:

| tensor | shape | role |
|---|---|---|
| `prompt_embeds` | [1, 512, 4096] | umT5-XXL output; feeds cross-attn context |
| `first_frame_latents` | [1, 1, 48, Hl, Wl] | Wan2.2-VAE latent of the start photo |
| `ref_latents` | [1, 5, 48, 1, 32, 32] | anchor slots (zeros in the MVP) |
| `ref_mask` | [1, 5] | zeros in the MVP |

The **golden pack** (`ABot-World/tools/dump_golden_walk.py`) adds, per block:
`noise_block{b}.npy` (the exact input noise) and `latent_block{b}.npy` (the DiT
output) plus decoded `frame_*.png` and `walk.json` (action bitmasks, seed,
shapes). The C++ walk **loads the golden noise** rather than using an RNG, so
parity does not depend on matching torch's random stream.

## Per-block algorithm (MVP: short walk, no KV eviction)

Keep the walk short enough that total frames ≤ `local_attn_size` (8): then no
eviction / RoPE re-basing is needed and the causal forward equals a
full-sequence recompute with a causal mask + absolute RoPE (the KV cache is only
an optimization of this same math). This is the key simplification.

For block `b` (each block = `num_frame_per_block`=3 latent frames):

1. `x = noise_block[b]`  (load from golden, [1,3,48,Hl,Wl])
2. If `b == 0`: replace the first latent frame with `first_frame_latents`
   (`x[:, 0:1] = first_frame_latents`).
3. **4-step DMD denoise** over `denoising_step_list = [1000,750,500,250]`
   warped by `timestep_shift=5.0` (see `configs/long_forcing_dmd.yaml`):
   for each step t, run the DiT forward on the *causal* concatenation of all
   previously-generated latent frames + the current block, and take the model's
   sample prediction (this model is a distilled few-step generator — follow
   `WanDiffusionWrapper.inference` / `causal_inference.generate_next_block` in
   the reference).
4. DiT forward specifics (extend `WAN::Wan::forward_orig`):
   - after `patch_embedding`, add the action-control features:
     `x = x + ActControlAdapter(act_planes) * act_context_scale` with
     **`act_context_scale = 1.0`** (reference `wan_wrapper.py:698`).
     `act_planes`: broadcast the 8-key bitmask → 8 channels → repeat_interleave
     4 → 32 channels over the frame's spatial grid, pixel-unshuffled by 16 (see
     `ActControlAdapter` doc in `wan.hpp` and `SimpleAdapter` in the reference).
   - self-attention uses a **causal mask** across frames and absolute-frame RoPE
     (reference `causal_rope_apply`, start_frame = block's first frame index).
   - cross-attention uses `prompt_embeds` (already supported).
5. Append the block's output latents to the running sequence (the "KV" of the
   MVP is just this kept latent list, re-fed each block).
6. Decode the block's latents with **taehv** (streaming; `TAEHV` in `tae.hpp`,
   already wired for the TI2V family) → RGB frames → mp4.

## Parity gates (validate against golden)

- Per-block DiT-output latent cosine vs `latent_block{b}.npy` ≥ **0.999** (F16).
- Decoded-frame PSNR vs golden `frame_*.png` ≥ **30 dB** (Q8 lane looser).
- Implement a `--dump-latents` flag on the walk CLI and a small compare script;
  wire it as the CI assertion.

## Deliverables

- `examples/abot-walk/` CLI: args `--diffusion-model --taehv --scene
  --golden --actions --out`; loads via the existing model loader; block loop
  above; writes mp4 (reuse `examples/common/media_io` for encoding).
- `script/validate_abot_world.sh` already checks load + guard; extend with the
  walk + latent/frame parity once the core lands.
- CI: `.github/workflows/abot-walk-mvp.yml` (scaffolded) — pulls model files
  from corp S3, builds, runs the walk, asserts parity + records the mp4 as an
  artifact.
