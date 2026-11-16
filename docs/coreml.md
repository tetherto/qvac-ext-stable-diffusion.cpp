## Running the UNet and the tiny decoder on the Apple Neural Engine (CoreML sidecars)

On Apple devices, two parts of the pipeline can run as CoreML models on the Neural Engine (ANE): the UNet of SD 1.x/2.x models (`--coreml-unet`), and the tiny autoencoder's decoder used by SDXS and `--taesd` (`--coreml-tae`). The text encoder, tokenizer and scheduler stay on ggml (Metal/CPU). The ggml UNet weights are not loaded when the UNet runs on CoreML.

SDXS-512-DreamShaper on iPhone 16 (512x512, 1 step): the UNet takes 640-680 ms with ggml Metal and 48-53 ms on the ANE, and the images match (36-38 dB PSNR). On an M2, running the tiny decoder on the ANE as well takes a whole image from 1,728 ms (all ggml Metal) to about 95 ms. [examples/ios-bench](../examples/ios-bench/README.md) reproduces this comparison on macOS and iOS.

### Build

```bash
cmake -B build -DSD_METAL=ON -DSD_COREML=ON
cmake --build build --config Release
```

`SD_COREML` needs an Apple platform. It links CoreML and adds the `--coreml-unet` and `--coreml-tae` options to `sd-cli` (`coreml_unet_path` and `coreml_tae_decoder_path` in `sd_ctx_params_t`).

### Export the UNet

```bash
pip install torch==2.5.1 diffusers==0.31.0 coremltools==8.3.0 numpy==1.26.4
python script/coreml/convert_unet.py --unet sdxs-512-dreamshaper/unet --out sdxs_unet_fp16.mlpackage --compile
```

`--unet` is the `unet/` folder of a diffusers pipeline. The script checks the exported model against PyTorch fp32 on the CPU+ANE and CPU+GPU, and `--compile` also writes the `.mlmodelc` to ship. Use fp16 (the default): `--precision w8` halves the file but runs no faster on the ANE and drifts more.

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

### Supported sizes

Every input shape of the CoreML model is fixed at export time (batch 1, 77 prompt tokens, one image size). Prompts longer than 77 tokens, ControlNet and models that take `c_concat` or `y` fail with an error.

| Model | Sizes to export | Notes |
| --- | --- | --- |
| SDXS-512-DreamShaper | 512x512 only | Distilled at 512 only. At 512x768, 768x512 and 768x768 it duplicates features (extra eyes and noses) on ggml Metal and on CoreML alike. `convert_unet.py` refuses other sizes for SDXS unless `--allow-untrained-sizes` is given. |

For other models, `convert_unet.py` warns when a size differs from the one in the UNet config. Only list sizes the model was trained at.

#### Several sizes in one file

`--sizes 512x512,512x768` exports a multifunction model (iOS 18 / macOS 15 or later): one fixed-shape function per size, all sharing one copy of the weights, so four sizes take the same space as one. The runtime picks the function that matches the requested size and loads it on first use, which costs one ANE compile per size.

Flexible shapes do not work on the ANE. With `EnumeratedShapes` or `RangeDim` inputs, the traced UNet keeps runtime reshapes the ANE cannot run, and CoreML places the whole graph on the CPU (about 12x slower). A multifunction model built with the iOS 18 op set fails to load with ANE compute units, so the script defaults to the iOS 17 op set, which runs at the same speed.

Known issue: loading several functions of a multifunction model in one process has shown a higher peak memory footprint (2.0 GB against 1.2 GB on iPhone 16). Prefer a single-size export until this is understood.

### Notes on accuracy

The ANE's fp16 `silu` op is not accurate enough for a one-step model: SDXS images visibly change (17 dB PSNR against Metal). `convert_unet.py` writes SiLU as `0.5 * x * (1 + tanh(x / 2))`, which is the same function, stays on the ANE at the same speed, and restores 37-39 dB. Writing it as `x * sigmoid(x)` does not help, because coremltools fuses that back into `silu`. Pass `--keep-silu` only to reproduce the problem.

On Metal, keep `--diffusion-conv-direct` and `--vae-conv-direct` off: direct convolution makes the SDXS decoder about 3x slower.
