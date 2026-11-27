## Running the UNet and the tiny decoder on the Apple Neural Engine (CoreML sidecars)

On Apple devices, two parts of the pipeline can run as CoreML models on the Neural Engine (ANE): the UNet of SD 1.x/2.x (including SD-Turbo) and SDXL models (`--coreml-unet`), and the tiny autoencoder's decoder used by SDXS and `--taesd` (`--coreml-tae`). The text encoder, tokenizer and scheduler stay on ggml (Metal/CPU). The ggml UNet weights are not loaded when the UNet runs on CoreML.

SDXS-512-DreamShaper on iPhone 16 (512x512, 1 step): the UNet takes 640-680 ms with ggml Metal and 48-53 ms on the ANE, and the images match (36-38 dB PSNR). On an M2, running the tiny decoder on the ANE as well takes a whole image from 1,728 ms (all ggml Metal) to about 95 ms. [examples/ios-bench](../examples/ios-bench/README.md) reproduces this comparison on macOS and iOS.

### Build

```bash
cmake -B build -DSD_METAL=ON -DSD_COREML=ON
cmake --build build --config Release
```

`SD_COREML` needs an Apple platform. It is the same option that builds the Core ML FLUX.2 denoiser (see [performance](./performance.md)); for SD 1.x/2.x and SDXL it adds the `--coreml-unet` and `--coreml-tae` options to `sd-cli` (`coreml_unet_path` and `coreml_tae_decoder_path` in `sd_ctx_params_t`). The UNet runs through `CoreMLUNetRunner` (`src/model/diffusion/coreml_unet_runner.h`) and the decoder through `CoreMLTinyImageAutoEncoder` (`src/model/vae/coreml_tae_runner.h`).

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
| | `sdxs_unet_fp16.mlmodelc` | `--coreml-unet` |
| | `sdxs_tae_decoder.mlmodelc` | `--coreml-tae` |
| SD 2.1-base | `sd21base-Q8_0.gguf` (sd-cli -M convert; the ggml UNet is only used without `--coreml-unet`) | `-m` |
| | `sd21base_unet_p6.mlmodelc` (6-bit palette, batch 2, split-einsum attention) | `--coreml-unet`, with `--prediction eps` |
| | `taesd.safetensors`, `taesd_decoder.mlmodelc` | `--taesd`, `--coreml-tae` |
| SD-Turbo (`--models sdturbo`, not in the default set) | `sdturbo-Q8_0.gguf` (sd-cli -M convert; the ggml UNet is only used without `--coreml-unet`) | `-m` |
| | `sdturbo_unet_p8.mlmodelc` (8-bit palette, batch 1, standard attention), for the ANE | `--coreml-unet` |
| | `sdturbo_unet_fp16.mlmodelc` (fp16, batch 1, standard attention), for the GPU | `--coreml-unet`, with `SD_COREML_COMPUTE_UNITS=cpu_gpu` |
| | `taesd.safetensors`, `taesd_decoder.mlmodelc` | `--taesd`, `--coreml-tae` |
| SDXL 1.0 base (`--models sdxl`, not in the default set) | `sd_xl_base_1.0.safetensors` (the checkpoint itself; its UNet is not loaded) | `-m` |
| | `sdxl_unet_mixed/` (two chunks, 4/6-bit mixed palette, batch 2, split-einsum attention, 768x768) | `--coreml-unet` |
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
| `p8` | 8-bit k-means palette per layer | Distilled few-step UNets (SD-Turbo): as close to fp16 as a Q8_0 GGUF is (see [SD-Turbo](#sd-turbo)). |
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

### Run

```bash
sd-cli -m sdxs-f16.gguf --coreml-unet sdxs_unet_fp16.mlmodelc --coreml-tae sdxs_tae_decoder.mlmodelc \
       -p "portrait of a lovely cat" --cfg-scale 1 --steps 1
```

The two options are independent. `-m` still has to point at the full model: the text encoder and VAE come from it, and the UNet tensors in it are skipped. `--coreml-tae` needs the image tiny autoencoder (SDXS, or `--taesd` with an SD or SDXL model); with `--vae-tiling` the decoder falls back to ggml, because the CoreML decoder has one fixed latent shape. `SD_COREML_COMPUTE_UNITS` picks where CoreML runs: `cpu_ne` (default), `all`, `cpu_gpu` or `cpu`.

The sidecars run fixed graphs, so some options are rejected instead of being silently ignored. With `--coreml-unet`, a generation with circular padding (`--circular`, `--circularx`, `--circulary`) or any LoRA fails with an error, as with the Core ML FLUX.2 denoiser. With `--coreml-tae`, circular padding, tiling and encoding use the ggml tiny decoder, which stays loaded.

CFG with a scale above 1 uses the pipeline's batched CFG (`--batched-cfg`, on by default): a UNet exported with `--batch 2` takes both passes in one call, and a batch-1 UNet runs them one after the other.

The first load on a device compiles each model for the ANE (about 10 s for the SDXS UNet on iPhone 16). CoreML caches the result, so later loads take well under a second. Ship a `.mlmodelc`; an `.mlpackage` is compiled on every load.

### Apple's pre-converted UNets

The UNets Apple publishes on Hugging Face ([apple/coreml-stable-diffusion-2-1-base-palettized](https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized) and the other `apple/coreml-stable-diffusion-*` repos) work as `--coreml-unet`. Use the `split_einsum_v2/compiled/Unet.mlmodelc` variant, which is made for the ANE. These models take float16 inputs, the text context as `[N, D, 1, T]`, and a batch of 2: CFG's conditional and unconditional passes run in one call. sd.cpp handles all three; the batch of 2 needs a CFG scale above 1 and no ControlNet, image CFG, skip layers or step caching. They do not record their prediction type, so pass `--prediction eps` for SD 2.1-base. The text encoder still comes from the `-m` model; SD 2.1 and SD 2.1-base share the same OpenCLIP text encoder.

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

SDXL's UNet is 5.1 GB in fp16, so for iOS it is palettized and split in two. `--chunks 2` splits the converted model with coremltools' `bisect_model` into a folder of `*_chunk1`/`*_chunk2` models; `--coreml-unet` takes that folder (or any folder of `*chunk<N>.mlmodelc` files, such as Apple's `UnetChunk1/2.mlmodelc`) and runs the chunks in order, each taking its inputs from the UNet inputs and earlier chunks' outputs.

```bash
python script/coreml/convert_unet.py --unet sdxl-base/unet --out sdxl_unet_mixed.mlpackage --precision mixed \
       --batch 2 --attention split_einsum_v2 --sizes 768x768 --chunks 2 --compile
sd-cli -m sd_xl_base_1.0.safetensors --coreml-unet sdxl_unet_mixed --taesd taesdxl.safetensors \
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

### SD-Turbo

[SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) is SD 2.1 distilled to 1-4 steps without CFG, at 512x512. Its UNet has the SD 2.1 architecture and predicts epsilon, which the export records, so no `--prediction` is needed. Without CFG the UNet is exported for batch 1.

```bash
python script/coreml/export_artifacts.py --out artifacts --models sdturbo --sd-cli build/bin/sd-cli
sd-cli -m sdturbo-Q8_0.gguf --coreml-unet sdturbo_unet_p8.mlmodelc --taesd taesd.safetensors \
       --coreml-tae taesd_decoder.mlmodelc --cfg-scale 1 --steps 4 -p "portrait of a lovely cat"
```

Measured with `sd-bench` on a Mac mini (M4, 16 GB, macOS 15.3) and a Mac Studio (M3 Ultra, 60-core GPU, macOS 26.7): 512x512, CFG 1, TAESD as the decoder, mean of 3 warm images. It has not been measured on an iPhone yet.

| SD-Turbo | M4, 1 step | M4, 4 steps | M3 Ultra, 1 step | M3 Ultra, 4 steps |
| --- | --- | --- | --- | --- |
| All ggml Metal (Q8_0) | 1,116 ms | 2,873 ms | 461 ms | 1,138 ms |
| `sdturbo_unet_p8` on the ANE, TAESD on CoreML | 231 ms (4.8x faster) | 775 ms (3.7x faster) | 241 ms | 851 ms |
| `sdturbo_unet_fp16` on the GPU (`cpu_gpu`), TAESD on CoreML | 328 ms | 1,084 ms | 91 ms (5.1x faster) | 272 ms (4.2x faster) |

One UNet call:

| SD-Turbo UNet | M4 | M3 Ultra |
| --- | --- | --- |
| ggml Metal, Q8_0 | 590 ms | 228 ms |
| `sdturbo_unet_p8` on the ANE | 178 ms | 205 ms |
| `sdturbo_unet_fp16` on the ANE | 192 ms | 215 ms |
| `sdturbo_unet_fp16` on the GPU | 252 ms | 60 ms |
| `sdturbo_unet_p8` on the GPU | 740 ms | 150 ms |

The ANE costs about the same on both Macs, while the GPU follows its core count. So the ANE is the place for the UNet on an M4 (and, by the SD 2.1-base numbers above, on an iPhone), and the GPU on a Max or Ultra: use `sdturbo_unet_p8` with the default compute units on the first, and `sdturbo_unet_fp16` with `SD_COREML_COMPUTE_UNITS=cpu_gpu` on the second. Keep the palettized UNet off the GPU, where it is 2.5-3x slower than fp16. The tiny decoder takes 480 ms (M4) or 210 ms (M3 Ultra) on Metal and 10 ms on CoreML.

The first load compiles the UNet for the ANE (12-24 s); later loads take 0.3 s, for the fp16 export as well. On the GPU the first load takes 1.5-6 s and later ones 0.3-0.6 s.

#### Weights

A distilled model has few steps to average weight error out, so SD-Turbo needs more bits than SD 2.1-base. PSNR of the final image against ggml Metal with F16 weights on the same Mac, mean of the three `sd-bench` prompts:

| SD-Turbo UNet | Size | 1 step | 4 steps |
| --- | --- | --- | --- |
| ggml Metal, Q8_0 GGUF | 2.3 GB (whole model) | 29.5 dB | 28.5 dB |
| CoreML fp16, on the GPU | 1.6 GB | 51.4-51.6 dB | 48.9-49.1 dB |
| CoreML fp16, on the ANE | 1.6 GB | 39.8-40.4 dB | 35.3-36.2 dB |
| CoreML `p8`, on the ANE | 827 MB | 30.7-30.8 dB | 28.7-28.8 dB |
| CoreML `p6`, on the ANE (M3 Ultra only) | 622 MB | 21.1 dB | 21.1 dB |

`p8` differs from fp16 as much as the Q8_0 GGUF does and is half the size of fp16, so it is the ANE artifact. `p6`, the SD 2.1-base choice, still gives a valid image with the same composition, but details change (the text on a billboard comes out different); use it only when the 200 MB matter. `w8b` (851 MB) did not compile for the ANE on the M3 Ultra: each load took 60-137 s and then ran on the CPU. The fp16 export runs on the ANE of both Macs; whether a 1.6 GB UNet compiles for the ANE on an iPhone is not tested.

#### Attention

One fp16 UNet call:

| `--attention` | M4, ANE | M4, GPU | M3 Ultra, ANE | M3 Ultra, GPU |
| --- | --- | --- | --- | --- |
| `standard` | 192 ms | 252 ms | 215 ms | 60 ms |
| `split_einsum_v2` | 203 ms | 269 ms | 190 ms | NaN output |
| `ane` | 150 ms, wrong image | 299 ms | 143 ms | 110 ms |

Only `standard` gives the right image on both Macs and both devices, and at batch 1 it costs at most 13% against `split_einsum_v2`, so both SD-Turbo artifacts use it. `ane` is the fastest on the ANE, but on the M4 (macOS 15.3) its images are blurred or scrambled (11-14 dB against Metal), while the same export is correct on the M3 Ultra (macOS 26.7); the export's own check shows it only as a cosine of 0.9991 against PyTorch instead of 0.999999. On the M3 Ultra a `split_einsum_v2` export returns NaN on the GPU when the queries are split into chunks (the default `--query-chunk 512` at 512x512; `--query-chunk 4096` avoids it), and the image is noise; the M4's GPU runs it correctly. `convert_unet.py` checks every export on the ANE and the GPU and prints a warning naming the compute units to avoid when the output is not finite. The SD 2.1-base artifact has the same UNet architecture and a `split_einsum_v2` export, so run it with the default `cpu_ne`.

### Supported sizes

Every input shape of the CoreML model is fixed at export time (batch 1, 77 prompt tokens, one image size). Prompts longer than 77 tokens, ControlNet and models that take `c_concat` (inpainting, Instruct-Pix2Pix) fail with an error.

| Model | Sizes to export | Notes |
| --- | --- | --- |
| SD 2.1 (768-v) | 512x512 tested | Native size is 768x768. At 512x512 the images are good; 768x768 is not tested yet. |
| SDXL 1.0 base | 768x768 tested | Native size is 1024x1024; 768x768 keeps the iOS model and activations small, as Apple's iOS export does. |
| SD-Turbo | 512x512 tested | Distilled at 512x512. |
| SDXS-512-DreamShaper | 512x512 only | Distilled at 512 only. At 512x768, 768x512 and 768x768 it duplicates features (extra eyes and noses) on ggml Metal and on CoreML alike. `convert_unet.py` refuses other sizes for SDXS unless `--allow-untrained-sizes` is given. |

For other models, `convert_unet.py` warns when a size differs from the one in the UNet config. Only list sizes the model was trained at.

#### Several sizes in one file

`--sizes 512x512,512x768` exports a multifunction model (iOS 18 / macOS 15 or later): one fixed-shape function per size, all sharing one copy of the weights, so four sizes take the same space as one. The runtime picks the function that matches the requested size and loads it on first use, which costs one ANE compile per size.

Flexible shapes do not work on the ANE. With `EnumeratedShapes` or `RangeDim` inputs, the traced UNet keeps runtime reshapes the ANE cannot run, and CoreML places the whole graph on the CPU (about 12x slower). A multifunction model built with the iOS 18 op set fails to load with ANE compute units, so the script defaults to the iOS 17 op set, which runs at the same speed.

Known issue: loading several functions of a multifunction model in one process has shown a higher peak memory footprint (2.0 GB against 1.2 GB on iPhone 16). Prefer a single-size export until this is understood.

### Tests

`test-coreml` (built with `-DSD_COREML=ON -DSD_BUILD_TESTS=ON` and static libraries) runs tiny CoreML models through the sidecar runtime: fp32 and fp16 inputs, channels-first contexts, shape rejection, prediction metadata, batched CFG against separate passes, SDXL's `y` and Apple's `text_embeds`/`time_ids` inputs, chunked models, and decoder output checks. With `SD_TEST_SDXS_DIR` set to the SDXS artifacts it also runs the pipeline checks: circular padding and LoRAs are rejected with the CoreML UNet, and `--vae-tiling` falls back to the ggml decoder with the same image. `test_convert_unet.py` checks the attention, SiLU and SDXL time id rewrites against the original diffusers code on small random UNets, the mixed-bit selection, and the warning for exports whose output is not finite. `test_export_artifacts.py` checks the commands `export_artifacts.py` plans for SD-Turbo, without downloading anything.

```bash
python tests/coreml/make_fixtures.py build/coreml-fixtures
build/bin/test-coreml build/coreml-fixtures
SD_TEST_SDXS_DIR=artifacts build/bin/test-coreml build/coreml-fixtures
python -m unittest script/coreml/test_convert_unet.py script/coreml/test_export_artifacts.py
```

### Notes on accuracy

The ANE's fp16 `silu` op is not accurate enough for a one-step model: SDXS images visibly change (17 dB PSNR against Metal). `convert_unet.py` writes SiLU as `0.5 * x * (1 + tanh(x / 2))`, which is the same function, stays on the ANE at the same speed, and restores 37-39 dB. Writing it as `x * sigmoid(x)` does not help, because coremltools fuses that back into `silu`. Pass `--keep-silu` only to reproduce the problem.

On Metal, keep `--diffusion-conv-direct` and `--vae-conv-direct` off: direct convolution makes the SDXS decoder about 3x slower.
