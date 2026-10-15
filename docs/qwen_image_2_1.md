# Qwen-Image-2.1

Qwen-Image-2.1 supports text-to-image generation and ordered reference-image
editing. The engine detects this architecture from the weights and selects its
flow schedule automatically. Use the dedicated 2.1 VAE and a Qwen3-VL-8B
encoder; earlier Qwen Image model components are not interchangeable.

## Validated model files

| File | Source revision |
| --- | --- |
| `qwen_image_2.1-Q4_K.gguf` | [leejet/Qwen-Image-2.1-GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/tree/cc11433936a06e9765f7c0c0b1f0436cfd2b9856) |
| `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/f982a07559d4a2f6c8744d840bf6fccab30eea96) |
| `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | Same encoder repository and revision; required for editing with the separate GGUF encoder |
| `qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/a8342359c61979b06a8dafb1a47991fe1f30e994/vae) |

Weights retain their own licenses. See the [official model license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE)
before using or distributing them.

## Native CLI

With a CUDA-enabled build and the model files under `models/`:

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/sd-cli \
  --diffusion-model models/qwen_image_2.1-Q4_K.gguf \
  --llm models/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --vae models/qwen_image_2.1_vae_bf16.safetensors \
  --backend cuda0 --params-backend cuda0 --diffusion-fa \
  --steps 30 --cfg-scale 6 --sampling-method euler \
  -W 512 -H 512 --seed 42 \
  -p 'A red ceramic teapot on a wooden table, soft daylight.' \
  -o teapot.png
```

Dimensions must be multiples of 32. Dynamic-backend builds must make their CUDA
module discoverable; `GGML_BACKEND_PATH` can point to the backend library file.
The CLI loads backend modules before context creation.

## C API reference editing and RGBA

Set `sd_ctx_params_t.llm_vision_path` to the projector file when using the split
GGUF encoder. Leave prediction and scheduler automatic. Supply references in
`sd_img_gen_params_t.ref_images`, with their count in `ref_images_count`.
References are interpreted in array order, and target generation starts from
noise. FLUX `@imageN` prompt tags are not required.

`sd_ctx_uses_reference_images()` identifies contexts whose reference images
should use that path. `sd_ctx_get_image_channels()` returns the decoded channel
count expected by the context (four for Qwen-Image-2.1, zero for an invalid
context). Decode transparent inputs as RGBA and preserve the channel count when
saving the returned image. The CLI's existing RGB input loader does not preserve
reference alpha; use the C API for transparent references.

The VAE produces RGBA output. Describe an RGBA image on a transparent background
to request transparent content. Alpha values are preserved without thresholding;
model-generated RGB beneath transparent pixels need not be neutral.

## Validation and limits

Validated with the model files above on RTX 5090 device 0 using CUDA 13.3.
Text-to-image, single/two-reference edits, and transparent generation/editing
were exercised through the C API via a separately modified QVAC addon. The
addon and registry changes are maintained separately from this engine change.
Cancellation, subsequent generation, repeated unload, and missing-projector
failure/recovery were also checked. Other hardware, Turbo, prefix KV caching,
and broad LoRA compatibility have not been validated.

The Qwen VAE skips temporal upsampling on its first frame; applying the existing
Wan video path here corrupts the decoded image. `test-qwen-image-21` checks this
behavior while retaining Wan's temporal convolution, as well as reference-token
ordering and the model's flow schedule:

```sh
ctest --test-dir build --output-on-failure \
  -R 'qwen-image-21|vae-routing|vae-name-conversion|fit-params|lora-validation'
```
