## Running the UNet and the tiny decoder on the Apple Neural Engine (CoreML sidecars)

On Apple devices, two parts of the pipeline can run as CoreML models on the Neural Engine (ANE): the UNet of SD 1.x/2.x models (`--coreml-unet`), and the tiny autoencoder's decoder used by SDXS and `--taesd` (`--coreml-tae`). The text encoder, tokenizer and scheduler stay on ggml (Metal/CPU). The ggml UNet weights are not loaded when the UNet runs on CoreML.

SDXS-512-DreamShaper on iPhone 16 (512x512, 1 step): the UNet takes 640-680 ms with ggml Metal and 48-53 ms on the ANE, and the images match (36-38 dB PSNR). On an M2, running the tiny decoder on the ANE as well takes a whole image from 1,728 ms (all ggml Metal) to about 95 ms. [examples/ios-bench](../examples/ios-bench/README.md) reproduces this comparison on macOS and iOS.

### Build

```bash
cmake -B build -DSD_METAL=ON -DSD_COREML=ON
cmake --build build --config Release
```

`SD_COREML` needs an Apple platform. It links CoreML and adds the `--coreml-unet` and `--coreml-tae` options to `sd-cli` (`coreml_unet_path` and `coreml_tae_decoder_path` in `sd_ctx_params_t`).

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

### Supported sizes

Every input shape of the CoreML model is fixed at export time (batch 1, 77 prompt tokens, one image size). Prompts longer than 77 tokens, ControlNet and models that take `c_concat` or `y` fail with an error.

| Model | Sizes to export | Notes |
| --- | --- | --- |
| SD 2.1 (768-v) | 512x512 tested | Native size is 768x768. At 512x512 the images are good; 768x768 is not tested yet. |
| SDXS-512-DreamShaper | 512x512 only | Distilled at 512 only. At 512x768, 768x512 and 768x768 it duplicates features (extra eyes and noses) on ggml Metal and on CoreML alike. `convert_unet.py` refuses other sizes for SDXS unless `--allow-untrained-sizes` is given. |

For other models, `convert_unet.py` warns when a size differs from the one in the UNet config. Only list sizes the model was trained at.

#### Several sizes in one file

`--sizes 512x512,512x768` exports a multifunction model (iOS 18 / macOS 15 or later): one fixed-shape function per size, all sharing one copy of the weights, so four sizes take the same space as one. The runtime picks the function that matches the requested size and loads it on first use, which costs one ANE compile per size.

Flexible shapes do not work on the ANE. With `EnumeratedShapes` or `RangeDim` inputs, the traced UNet keeps runtime reshapes the ANE cannot run, and CoreML places the whole graph on the CPU (about 12x slower). A multifunction model built with the iOS 18 op set fails to load with ANE compute units, so the script defaults to the iOS 17 op set, which runs at the same speed.

Known issue: loading several functions of a multifunction model in one process has shown a higher peak memory footprint (2.0 GB against 1.2 GB on iPhone 16). Prefer a single-size export until this is understood.

### Notes on accuracy

The ANE's fp16 `silu` op is not accurate enough for a one-step model: SDXS images visibly change (17 dB PSNR against Metal). `convert_unet.py` writes SiLU as `0.5 * x * (1 + tanh(x / 2))`, which is the same function, stays on the ANE at the same speed, and restores 37-39 dB. Writing it as `x * sigmoid(x)` does not help, because coremltools fuses that back into `silu`. Pass `--keep-silu` only to reproduce the problem.

On Metal, keep `--diffusion-conv-direct` and `--vae-conv-direct` off: direct convolution makes the SDXS decoder about 3x slower.
