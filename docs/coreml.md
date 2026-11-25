## Running the diffusion model and the tiny decoder on the Apple Neural Engine (CoreML sidecars)

On Apple devices, two parts of the pipeline can run as CoreML models on the Neural Engine (ANE): the diffusion model (`--coreml-diffusion`), which is the UNet of SD 1.x/2.x and SDXL models or the [FLUX.2-klein 4B](#flux2-klein-4b) denoiser, and the tiny autoencoder's decoder used by SDXS and `--taesd` (`--coreml-tae`), including TAEF2 for FLUX.2. The text encoder, tokenizer and scheduler stay on ggml (Metal/CPU). The ggml diffusion weights are not loaded when the diffusion model runs on CoreML. `--coreml-unet` is an alias of `--coreml-diffusion`.

SDXS-512-DreamShaper on iPhone 16 (512x512, 1 step): the UNet takes 640-680 ms with ggml Metal and 48-53 ms on the ANE, and the images match (36-38 dB PSNR). On an M2, running the tiny decoder on the ANE as well takes a whole image from 1,728 ms (all ggml Metal) to about 95 ms. [examples/ios-bench](../examples/ios-bench/README.md) reproduces this comparison on macOS and iOS.

### Build

```bash
cmake -B build -DSD_METAL=ON -DSD_COREML=ON
cmake --build build --config Release
```

`SD_COREML` needs an Apple platform. It adds the `--coreml-diffusion` and `--coreml-tae` options to `sd-cli` (`coreml_diffusion_path` and `coreml_tae_decoder_path` in `sd_ctx_params_t`; `coreml_unet_path` is a deprecated alias of `coreml_diffusion_path`, used when that is empty). The UNet runs through `CoreMLUNetRunner` (`src/model/diffusion/coreml_unet_runner.h`), the FLUX.2 denoiser through `FluxCoreMLRunner` (`src/model/diffusion/flux_coreml_runner.h`) and the decoder through `CoreMLTinyImageAutoEncoder` (`src/model/vae/coreml_tae_runner.h`).

### Regenerate the artifacts

`script/coreml/export_artifacts.py` rebuilds every model file used in the measurements here from the original weights at pinned Hugging Face revisions, running the converters below one step at a time:

```bash
pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.3 coremltools==8.3.0 numpy==1.26.4 scikit-learn==1.5.2 huggingface_hub
python script/coreml/export_artifacts.py --out artifacts --sd-cli build/bin/sd-cli            # everything
python script/coreml/export_artifacts.py --out artifacts --models sd21base --dry-run          # print the commands
python script/coreml/export_artifacts.py --out artifacts --skip-existing                     # only what is missing
```

| Model | Artifact | Used as |
| --- | --- | --- |
| SDXS-512-DreamShaper | `sdxs-f16.gguf` (sd-cli -M convert) | `-m` |
| | `sdxs_unet_fp16.mlmodelc` | `--coreml-diffusion` |
| | `sdxs_tae_decoder.mlmodelc` | `--coreml-tae` |
| SD 2.1-base | `sd21base-Q8_0.gguf` (sd-cli -M convert; the ggml UNet is only used without `--coreml-diffusion`) | `-m` |
| | `sd21base_unet_p6.mlmodelc` (6-bit palette, batch 2, split-einsum attention) | `--coreml-diffusion`, with `--prediction eps` |
| | `taesd.safetensors`, `taesd_decoder.mlmodelc` | `--taesd`, `--coreml-tae` |
| SDXL 1.0 base (`--models sdxl`, not in the default set) | `sd_xl_base_1.0.safetensors` (the checkpoint itself; its UNet is not loaded) | `-m` |
| | `sdxl_unet_mixed/` (two chunks, 4/6-bit mixed palette, batch 2, split-einsum attention, 768x768) | `--coreml-diffusion` |
| | `taesdxl.safetensors`, `taesdxl_decoder_768.mlmodelc` | `--taesd`, `--coreml-tae` |

On this branch, `sd-cli -M convert` reads only `text_encoder/` from a diffusers folder, so the GGUF files come out without a UNet or VAE. Pass `--sd-cli` an `sd-cli` built from `master` until that is fixed.

The SD 2.1-base UNet takes about 5 minutes and 8 GB of RAM; nothing else should run alongside it on a 16 GB Mac. k-means palettes are not bit-for-bit reproducible, so a regenerated UNet can differ slightly from an earlier one in fine image detail.

### Export the UNet

```bash
pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4
python script/coreml/convert_unet.py --unet sdxs-512-dreamshaper/unet --out sdxs_unet_fp16.mlpackage --compile
```

`--unet` is the `unet/` folder of a diffusers pipeline (pipelines that only ship the `fp16` weight variant work too). The script checks the exported model against PyTorch fp32 on the CPU+ANE and CPU+GPU, and `--compile` also writes the `.mlmodelc` to ship. It records the UNet's prediction type (epsilon or v-prediction, from the pipeline's scheduler config or `--prediction`) in the model; sd.cpp reads it for SD 2.x, because its usual check runs the UNet on an 8x8 latent, which a fixed-shape CoreML UNet cannot take.

#### Weight precision

| `--precision` | Weights | Use for |
| --- | --- | --- |
| `fp16` (default) | fp16 | Small UNets such as SDXS (603 MB). Matches ggml Metal closely. |
| `w8b` | int8, one scale per block of 32 weights (iOS 18+) | SD 2.x and other multi-step models. |
| `w8` | int8, one scale per output channel | Older iOS only; drifts over many steps. |
| `p6` | 6-bit k-means palette per layer | Large UNets on iOS (SD 2.x): the ANE reads palettes directly, so the compiled model is cached. |
| `p4` | 4-bit k-means palette per layer | Not recommended: SDXL images lose fine detail (see [SDXL](#sdxl)). |
| `mixed` | 4-bit palettes, with 6 bits for the layers most sensitive to 4 bits, averaging `--mixed-bits` (default 4.5) | SDXL on iOS. |

The SD 2.1 UNet is 1.6 GB in fp16, too big to compile for the ANE: CoreML silently runs it on the CPU (about 1.1 s per call on an M2). int8 weights halve it and it runs on the ANE. How the int8 scales are shared matters for a 20-step model, because weight errors compound from step to step. Measured on an M2 with SD 2.1-768 at 512x512, 20 Euler steps, CFG 7, the same seed, against the ggml Metal (Q8_0) image:

| SD 2.1 UNet | Size | UNet per call | Image vs ggml Metal |
| --- | --- | --- | --- |
| ggml Metal, Q8_0 | 2.3 GB GGUF (whole model) | 1,320 ms | reference |
| CoreML fp16, on the GPU | 1.6 GB | about 480 ms | 23.9 dB |
| CoreML `w8`, on the ANE | 868 MB | 242 ms | 19.9 dB: valid image, but the composition shifts |
| CoreML `w8b`, on the ANE | 891 MB | 258 ms | 24.8 dB: same composition as Metal |

With one scale per channel, a single large weight coarsens the rounding of every other weight in that channel; with one scale per 32 weights it only affects its 31 neighbours. `w8b` needs the iOS 18 op set (the script selects it). Asymmetric int8 (with a zero point) moves the whole UNet off the ANE onto the CPU, so the script only uses symmetric int8.

### Export the tiny decoder

```bash
python script/coreml/convert_tae_decoder.py --vae sdxs-512-dreamshaper/vae --out sdxs_tae_decoder.mlpackage --compile
```

`--vae` is a diffusers `AutoencoderTiny` folder: SDXS's own `vae/`, or [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) for models used with `--taesd`. The export is the same graph as sd.cpp's `TinyDecoder` and returns the image in [0, 1]. It leaves out the `x * 2 - 1` that diffusers' `DecoderTiny.forward` ends with; including it gives dark images, because sd.cpp clamps the output to [0, 1]. The decoder has no attention and no SiLU, so it runs fully on the ANE as exported (53.7 dB against PyTorch fp32, 55 dB against the ggml decoder).

The ggml decoder is unusually slow on Metal for so small a model (1.2M parameters, about 1.1 s at 512x512), so the decoder sidecar saves more time per image than the UNet one: it takes about 10-20 ms on an M2.

For FLUX.2, pass [madebyollin/taef2](https://huggingface.co/madebyollin/taef2)'s `taef2.safetensors` itself; `--vae` also takes a `.safetensors` file in the original taesd key layout:

```bash
python script/coreml/convert_tae_decoder.py --vae taef2.safetensors --size 512x512 --out taef2_decoder_512.mlpackage --compile
```

TAEF2 decodes 32-channel latents, while FLUX.2 hands the decoder its 128-channel latent with 2x2 patches folded into channels, so the exported graph starts with the same unpatchify as sd.cpp's decoder and returns 16 image pixels per latent pixel. On an M2 it decodes a 512x512 image in 11 ms on the ANE (67 dB against PyTorch fp32, 58 dB against the ggml TAEF2); the full FLUX.2 VAE takes 4.6 s on Metal.

### Run

```bash
sd-cli -m sdxs-f16.gguf --coreml-diffusion sdxs_unet_fp16.mlmodelc --coreml-tae sdxs_tae_decoder.mlmodelc \
       -p "portrait of a lovely cat" --cfg-scale 1 --steps 1
```

The two options are independent. `-m` still has to point at the full model: the text encoder and VAE come from it, and the UNet tensors in it are skipped. `--coreml-tae` needs the image tiny autoencoder (SDXS, or `--taesd` with an SD, SDXL or FLUX.2 model); with `--vae-tiling` the decoder falls back to ggml, because the CoreML decoder has one fixed latent shape. `SD_COREML_COMPUTE_UNITS` picks where CoreML runs: `cpu_ne` (default), `all`, `cpu_gpu` or `cpu`.

The sidecars run fixed graphs, so some options are rejected instead of being silently ignored. With `--coreml-diffusion`, a generation with circular padding (`--circular`, `--circularx`, `--circulary`) or any LoRA fails with an error; the FLUX.2 denoiser also rejects reference images. With `--coreml-tae`, circular padding, tiling and encoding use the ggml tiny decoder, which stays loaded.

CFG with a scale above 1 uses the pipeline's batched CFG (`--batched-cfg`, on by default): a UNet exported with `--batch 2` takes both passes in one call, and a batch-1 UNet runs them one after the other.

The first load on a device compiles each model for the ANE (about 10 s for the SDXS UNet on iPhone 16). CoreML caches the result, so later loads take well under a second. Ship a `.mlmodelc`; an `.mlpackage` is compiled on every load.

### Apple's pre-converted UNets

The UNets Apple publishes on Hugging Face ([apple/coreml-stable-diffusion-2-1-base-palettized](https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized) and the other `apple/coreml-stable-diffusion-*` repos) work as `--coreml-diffusion`. Use the `split_einsum_v2/compiled/Unet.mlmodelc` variant, which is made for the ANE. These models take float16 inputs, the text context as `[N, D, 1, T]`, and a batch of 2: CFG's conditional and unconditional passes run in one call. sd.cpp handles all three; the batch of 2 needs a CFG scale above 1 and no ControlNet, image CFG, skip layers or step caching. They do not record their prediction type, so pass `--prediction eps` for SD 2.1-base. The text encoder still comes from the `-m` model; SD 2.1 and SD 2.1-base share the same OpenCLIP text encoder.

Apple's 6-bit palettized SD 2.1-base UNet (637 MB) is the one to use on iPhone. The ANE reads palettized weights directly, so the compiled model is cached after the first load. The per-block int8 export (`w8b`) is expanded to fp16 on the CPU on every load instead: on an iPhone 16 that took about 105 s and 5.3 GB each time, and the app ran out of memory. On an M2, 512x512, 20 Euler steps, CFG 7, with TAESD on CoreML: 6.3 s per image, 0.25 s cached load, 0.9 GB peak footprint.

`convert_unet.py --precision p6 --batch 2` follows the same recipe from the original diffusers weights (SD 2.1-base: 621 MB, about 5 minutes and 8 GB of RAM on an M2 with 2 k-means workers; needs scikit-learn). Its images match Apple's file in composition and subject (fine details differ after 20 steps) and it caches the same way. Apple also rewrites attention for the ANE; `--attention` reimplements that (our own code, same math, 2e-6 max difference in fp32):

| `--attention` (SD 2.1-base, p6, batch 2, M2) | One UNet call on the ANE | 20-step image |
| --- | --- | --- |
| `standard` | 537 ms | 9.8 s |
| `split_einsum_v2`: one small attention per head, queries in chunks of 512 | 368-381 ms | 7.5 s |
| `ane`: the same, with the whole transformer block channels-first (Linear as 1x1 conv) | 332 ms | not measured |
| Apple's pre-converted file | 304 ms | 6.3 s |

On an iPhone 16 (512x512, 20 Euler steps, CFG 7, TAESD on CoreML, mean of 3 warm images), the `split_einsum_v2` export takes 7.1 s per image against 6.1 s for Apple's file; both load from the cache in about 0.2 s after a one-time compile (64 s, 0.65 GB peak) and peak at 1.6 GB while generating.

Against ggml Metal (`sd21base-Q8_0.gguf`, TAESD as the decoder in every run, 512x512, 20 Euler steps, CFG 7, mean of 3 warm images):

| SD 2.1-base | iPhone 16 | M2 | Peak memory (iPhone 16) |
| --- | --- | --- | --- |
| All ggml Metal | 71.8 s | 47.3 s | 2.0 GB |
| UNet on CoreML, TAESD on Metal | 9.6 s | 8.8 s | 1.3 GB |
| UNet and TAESD on CoreML | 7.7 s (9.4x faster) | 7.6 s (6.2x faster) | 1.3 GB |

The UNet is 97% of an all-Metal image (20 steps, two passes each for CFG): 70.2 s on Metal against 7.5 s on the ANE on the iPhone. The iPhone runs followed each other without a cool-down, so the phone throttled as it warmed and times grow from image to image (all Metal 68.7, 72.6, 74.1 s); on a cool phone the CoreML pipeline took 7.1-7.3 s. CoreML's compute plan places 4,394 of the UNet's 4,399 operations and all 84 TAESD operations on the ANE; the rest is the timestep's sin/cos on the CPU.

The artifacts below use `split_einsum_v2`. For SDXS the attention form makes no measurable difference, so it keeps `standard`.

### SDXL

The SDXL UNet also takes `y`, the 2816-value vector conditioning (pooled CLIP-G embedding followed by the embedded image size and crop) that sd.cpp's SDXL conditioner builds. `convert_unet.py` exports SDXL UNets with a `y` input, so the model applies `add_embedding` to exactly what the ggml UNet would get; the export checks this against diffusers given the raw size and crop ids (3e-7 max relative error in fp32). Apple's SDXL models, which take `text_embeds` and the raw `time_ids` instead, also work: the runtime recovers the ids from `y`.

SDXL's UNet is 5.1 GB in fp16, so for iOS it is palettized and split in two. `--chunks 2` splits the converted model with coremltools' `bisect_model` into a folder of `*_chunk1`/`*_chunk2` models; `--coreml-diffusion` takes that folder (or any folder of `*chunk<N>.mlmodelc` files, such as Apple's `UnetChunk1/2.mlmodelc`) and runs the chunks in order, each taking its inputs from the UNet inputs and earlier chunks' outputs.

```bash
python script/coreml/convert_unet.py --unet sdxl-base/unet --out sdxl_unet_mixed.mlpackage --precision mixed \
       --batch 2 --attention split_einsum_v2 --sizes 768x768 --chunks 2 --compile
sd-cli -m sd_xl_base_1.0.safetensors --coreml-diffusion sdxl_unet_mixed --taesd taesdxl.safetensors \
       --coreml-tae taesdxl_decoder_768.mlmodelc -W 768 -H 768 --steps 20 --cfg-scale 7 -p "a red apple on a wooden table"
```

The export traces the UNet in fp32 (10 GB of weights) and peaks at about 30 GB of RAM and 15 GB of temporary disk, so run it on a Mac with 32 GB or more. `--precision mixed` first measures, for every weight, how much a 4-bit palette for that weight alone changes the UNet output (about 800 UNet calls, 20 minutes on an M3 Ultra, cached in `--sensitivity`), then gives 6 bits to the most sensitive weights until the average reaches `--mixed-bits`. For SDXL, 190 of 794 weights get 6 bits; the most sensitive are the downsampling convolutions, `conv_in`, the time embedding and the shortcut convolutions.

One batch-2 UNet call at 768x768 against PyTorch fp32:

| `--precision` | Size (2 chunks) | Cosine | Max abs error |
| --- | --- | --- | --- |
| `p4` | 1.24 GB | 0.998816 | 0.33 |
| `mixed` (4.5 bits) | 1.39 GB | 0.999974 | 0.043 |
| `p6` | 1.85 GB | 0.999976 | 0.037 |

On an M2 MacBook Air (16 GB), 768x768, 20 Euler steps, CFG 7, TAESDXL as the decoder, two prompts:

| SDXL UNet | Per image | Peak memory | Image |
| --- | --- | --- | --- |
| ggml Metal, Q8_0 | 162-232 s | 4.8 GB | reference |
| CoreML `p4` | 27.5 s | 2.3 GB | right subject, smeared texture (9-11 dB against Metal) |
| CoreML `mixed` | 25-28 s | 2.3 GB | same content as Metal (16-19 dB) |
| CoreML `p6` | 27.9 s | 2.3 GB | same content as Metal (17-18 dB) |

The UNet is 23-24 s of the CoreML image, so the speed is the same at every bit width; the first load compiles the chunks for the ANE (5.5-6 minutes on the M2), and later loads take about a second. The PSNR against Metal is lower than SD 2.1's because the trajectories of 20 SDXL steps diverge in fine detail, not because of artifacts: subject, composition and lighting match. Plain 4-bit palettes keep the subject but lose fine texture; spending 6 bits on a quarter of the weights recovers it at 75% of the 6-bit size. The faint cross-hatching TAESDXL leaves in dark areas appears in the Metal images as well.

Apple's [coreml-stable-diffusion-xl-base-ios](https://huggingface.co/apple/coreml-stable-diffusion-xl-base-ios) chunks (4.04-bit) also load and run at the same speed (18 s sampling), but with sd.cpp's conditioning they do not follow the prompt (an apple prompt gives a red railing); prefer the `mixed` export.

The Metal numbers above come from an sd-cli built from the 2026-03-01 branch, measured before batched CFG gave flat images on Metal (fixed since); this branch's Metal image matches it at 38.6 dB.

### FLUX.2-klein 4B

The FLUX.2-klein 4B denoiser runs on the ANE as five chunks of 8-bit palettized weights (3.9 GB). On a MacBook Air with an M2 and 16 GB (512x512, 4 steps, CFG 1, the mean of 3 images after a warm-up), with the TAEF2 decoder sidecar:

| | ggml Metal (Q8_0, full VAE) | CoreML denoiser + TAEF2 sidecar |
| --- | --- | --- |
| Per image | 45.4 s | 12.8 s |
| Denoising, 4 steps | 37.6 s | 7.5 s |
| Per step, steps 2-4 | 7.4 s (5.2 s on a cool machine) | 1.9 s |
| Decode | 5.5 s | 0.04 s |
| Peak memory | 7.6 GB | 2.6 GB |

The images show the same picture as Metal's (22 dB PSNR), and repeated runs give identical images. Metal slows down as the fanless Air heats up; the ANE does not. The Qwen3 text encoder stays on ggml in both.

Export each chunk in its own process from the official checkpoint ([black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)) and the official [flux2](https://github.com/black-forest-labs/flux2) source. The fixture is one captured ggml call (`SDCPP_FLUX_CAPTURE_DIR`, see [performance](./performance.md)); it fixes the image size and the number of text tokens (512 at 512x512):

```bash
pip install -r script/requirements-flux2-coreml.txt
for k in 1 2 3 4 5; do
  python script/export_flux2_klein_coreml.py chunk --source flux2-official --weights flux-2-klein-4b.safetensors \
      --fixture fixture-512/call-1 --chunk $k --output klein4b_512
done
python script/compare_flux_coreml_fixture.py --model klein4b_512 --compute-units cpu_ne \
    --fixture fixture-512/call-1 --output parity.json --max-nrmse 0.1
sd-cli --diffusion-model flux-2-klein-4b-Q8_0.gguf --vae flux2-vae.safetensors --llm Qwen3-4B-Q4_K_M.gguf \
       --coreml-diffusion klein4b_512 --taesd taef2.safetensors --coreml-tae taef2_decoder_512.mlmodelc \
       -W 512 -H 512 --steps 4 --cfg-scale 1 -p "a lovely cat sitting on a wooden bench in a sunny garden"
```

Each chunk loads only the weights it uses; converting one peaks at 9-12 GB of memory and takes about 3 minutes on an M2. `--diffusion-model` is still needed for the model's metadata, but its weights are not loaded. The first load on a device compiles the chunks for the ANE (one to two minutes on an M2); later loads take about 2 s. Text-to-image only: reference images, LoRAs and other sizes fail with an error.

What the export does, and why:

- **FP16 overflow.** Double block 4's text MLP writes values of about 100,000 into the text tokens' residual stream, past FP16's 65,504, so FP16 images break from the second step on. Every read of a residual stream goes through a LayerNorm without affine parameters, so dividing what writes the streams by 8 (`--fp16-rescale 8`, the default) changes nothing in FP32 and keeps FP16 finite.
- **Program size.** The ANE runs a program of at most about 1 GB of stored weights on an M2 (about 2 GB on an M3); a larger one runs entirely on the CPU, with no error. Palettized weights count at their compressed size. The default cuts (`--cuts 3,6,13,20`) give chunks of 0.63-0.85 GB at 8 bits.
- **Attention on the M2.** One attention op over all 24 heads of the 1,536-token sequence returns wrong values on an M2's ANE (and differs from run to run); the M3 computes it correctly. The export computes attention per head, which is correct on both and faster.
- **Precision.** 8-bit palettes keep the picture; 6-bit ones (`--palette-bits 6`, 2.8 GB) change details over the 4 steps, and fp16 weights (`--palette-bits 0`) exceed the M2's program size at these cuts. The sinusoidal timestep embedding stays FP32 on the CPU.

`SDCPP_FLUX2_COREML_MODEL` still selects the denoiser when `--coreml-diffusion` is not given. Set `SD_COREML_COMPUTE_UNITS=all` for the older FP32 export described in [performance](./performance.md).

### Supported sizes

Every input shape of the CoreML model is fixed at export time (batch 1, 77 prompt tokens, one image size). Prompts longer than 77 tokens, ControlNet and models that take `c_concat` (inpainting, Instruct-Pix2Pix) fail with an error.

| Model | Sizes to export | Notes |
| --- | --- | --- |
| SD 2.1 (768-v) | 512x512 tested | Native size is 768x768. At 512x512 the images are good; 768x768 is not tested yet. |
| SDXL 1.0 base | 768x768 tested | Native size is 1024x1024; 768x768 keeps the iOS model and activations small, as Apple's iOS export does. |
| SDXS-512-DreamShaper | 512x512 only | Distilled at 512 only. At 512x768, 768x512 and 768x768 it duplicates features (extra eyes and noses) on ggml Metal and on CoreML alike. `convert_unet.py` refuses other sizes for SDXS unless `--allow-untrained-sizes` is given. |

For other models, `convert_unet.py` warns when a size differs from the one in the UNet config. Only list sizes the model was trained at.

#### Several sizes in one file

`--sizes 512x512,512x768` exports a multifunction model (iOS 18 / macOS 15 or later): one fixed-shape function per size, all sharing one copy of the weights, so four sizes take the same space as one. The runtime picks the function that matches the requested size and loads it on first use, which costs one ANE compile per size.

Flexible shapes do not work on the ANE. With `EnumeratedShapes` or `RangeDim` inputs, the traced UNet keeps runtime reshapes the ANE cannot run, and CoreML places the whole graph on the CPU (about 12x slower). A multifunction model built with the iOS 18 op set fails to load with ANE compute units, so the script defaults to the iOS 17 op set, which runs at the same speed.

Known issue: loading several functions of a multifunction model in one process has shown a higher peak memory footprint (2.0 GB against 1.2 GB on iPhone 16). Prefer a single-size export until this is understood.

### Tests

`test-coreml` (built with `-DSD_COREML=ON -DSD_BUILD_TESTS=ON` and static libraries) runs tiny CoreML models through the sidecar runtime: fp32 and fp16 inputs, channels-first contexts, shape rejection, prediction metadata, batched CFG against separate passes, SDXL's `y` and Apple's `text_embeds`/`time_ids` inputs, chunked models, decoder output checks at 8x and 16x (TAEF2), and the FLUX.2 sidecar as one model and as two chunks passing an fp16 tensor. With `SD_TEST_SDXS_DIR` set to the SDXS artifacts it also runs the pipeline checks: circular padding and LoRAs are rejected with the CoreML UNet, and `--vae-tiling` falls back to the ggml decoder with the same image. `test_convert_unet.py` checks the attention, SiLU and SDXL time id rewrites against the original diffusers code on small random UNets, and the mixed-bit selection.

```bash
python tests/coreml/make_fixtures.py build/coreml-fixtures
build/bin/test-coreml build/coreml-fixtures
SD_TEST_SDXS_DIR=artifacts build/bin/test-coreml build/coreml-fixtures
python -m unittest script/coreml/test_convert_unet.py
(cd script && python -m unittest test_export_flux2_klein_coreml test_compare_flux_coreml_fixture)
```

### Notes on accuracy

The ANE's fp16 `silu` op is not accurate enough for a one-step model: SDXS images visibly change (17 dB PSNR against Metal). `convert_unet.py` writes SiLU as `0.5 * x * (1 + tanh(x / 2))`, which is the same function, stays on the ANE at the same speed, and restores 37-39 dB. Writing it as `x * sigmoid(x)` does not help, because coremltools fuses that back into `silu`. Pass `--keep-silu` only to reproduce the problem.

On Metal, keep `--diffusion-conv-direct` and `--vae-conv-direct` off: direct convolution makes the SDXS decoder about 3x slower.
