## sd-bench: Metal vs CoreML benchmark (macOS and iOS)

Compares ggml Metal with the CoreML UNet and tiny-decoder sidecars ([docs/coreml.md](../../docs/coreml.md)) on SDXS-512-DreamShaper. It loads the model once, generates 1 warm-up and 3 timed images (512x512, 1 step, CFG 1, seed 42), and prints the text encoder, UNet, decoder and total time for each. The prompts are a cat portrait, a cabin at sunset and a billboard with the text "COFFEE".

```
sd-bench bench <model.gguf> <unet.mlmodelc|-> <out_prefix> [tae_decoder.mlmodelc]
```

Pass `-` to run the UNet with ggml Metal, or a `.mlmodelc` to run it on CoreML (`SD_COREML_COMPUTE_UNITS` picks the device, `cpu_ne` by default). The optional last argument runs the tiny decoder on CoreML too. Images are written as `<out_prefix>_<prompt>.png`.

Options after the positional arguments: `--steps N`, `--cfg F`, `--sampler NAME`, `--taesd FILE` (the ggml TAESD weights, for models without a built-in tiny decoder; needed with a CoreML decoder too) and `--images N` (1-3 timed images). The defaults fit SDXS. For SD 2.1 with the `w8b` UNet and TAESD on CoreML:

```
sd-bench bench stable-diffusion-v2-1-Q8_0.gguf sd21_unet_w8b.mlmodelc sd21_ane taesd_decoder.mlmodelc \
         --taesd taesd.safetensors --steps 20 --cfg 7 --sampler euler
```

For FLUX.2-klein 4B, whose model comes in separate files, pass `-` as the model and the files as options. The CoreML denoiser is a folder of chunks ([docs/coreml.md](../../docs/coreml.md#flux2-klein-4b)); `--diffusion-model` still has to be given for the model's metadata, but its weights are not loaded when the denoiser runs on CoreML:

```
sd-bench bench - klein4b_512 flux_ane taef2_decoder_512.mlmodelc --diffusion-model flux-2-klein-4b-Q4_0.gguf \
         --vae flux2-vae.safetensors --llm Qwen3-4B-Q4_K_M.gguf --taesd taef2.safetensors --steps 4 --cfg 1 --sampler euler
sd-bench bench - - flux_metal --diffusion-model flux-2-klein-4b-Q4_0.gguf --vae flux2-vae.safetensors \
         --llm Qwen3-4B-Q4_K_M.gguf --steps 4 --cfg 1 --sampler euler --mmap
```

### macOS

```bash
cmake -S examples/ios-bench -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench
build-bench/sd-bench bench sdxs-f16.gguf - mac_metal
build-bench/sd-bench bench sdxs-f16.gguf sdxs_unet_fp16.mlmodelc mac_ane
build-bench/sd-bench bench sdxs-f16.gguf sdxs_unet_fp16.mlmodelc mac_ane_tae sdxs_tae_decoder.mlmodelc
```

### iOS

The app runs `sd-bench` when the first launch argument is `bench`, and `sd-cli` otherwise, from its Documents directory, and prints the peak memory footprint at the end.

```bash
cmake -S examples/ios-bench -B build-ios -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 -DSDBENCH_TEAM=<your team id>
xcodebuild -project build-ios/sdbench.xcodeproj -target sdbench -configuration Release -sdk iphoneos \
           -allowProvisioningUpdates build

DEVICE=<device id from: xcrun devicectl list devices>
xcrun devicectl device install app --device $DEVICE build-ios/Release-iphoneos/sdbench.app
for f in sdxs-f16.gguf sdxs_unet_fp16.mlmodelc; do
  xcrun devicectl device copy to --device $DEVICE --domain-type appDataContainer \
        --domain-identifier io.tether.sdbench --source $f --destination Documents/$f
done
xcrun devicectl device process launch --device $DEVICE --console --terminate-existing io.tether.sdbench -- \
      bench sdxs-f16.gguf - ios_metal
xcrun devicectl device process launch --device $DEVICE --console --terminate-existing io.tether.sdbench -- \
      bench sdxs-f16.gguf sdxs_unet_fp16.mlmodelc ios_ane
xcrun devicectl device process launch --device $DEVICE --console --terminate-existing io.tether.sdbench -- \
      bench sdxs-f16.gguf sdxs_unet_fp16.mlmodelc ios_ane_tae sdxs_tae_decoder.mlmodelc
```

Copy `sdxs_tae_decoder.mlmodelc` to Documents as well for the last run.

Copy the images back with `xcrun devicectl device copy from ... --source Documents/ios_ane_cat.png --destination .`. The first CoreML run on a device includes the ANE compile (about 10 s on iPhone 16); later runs load from the cache.

The app asks for the increased-memory-limit entitlement (`sdbench.entitlements`).

The app finishes launching first and runs on a background thread, because iOS kills an app that takes more than 20 s to launch, and a first-time Neural Engine compile can take minutes. It also copies its output to `Documents/sdbench.log`, which you can fetch with `xcrun devicectl device copy from ... --source Documents/sdbench.log`. Set `SDBENCH_MEMLOG=1` (`--environment-variables '{"SDBENCH_MEMLOG":"1"}'`) to log the app's memory every second, and `SDBENCH_VERBOSE=1` for every sd.cpp log line. `compile <model.mlmodelc>` loads only a CoreML model and exits, which shows its load time and peak memory on its own.

SD 2.1's `w8b` UNet does not fit on an iPhone 16. CoreML expands the per-block int8 weights to fp16 on the CPU on every load (it is not cached): about 105 s and a 5.3 GB peak against the app's 6.1 GB limit, and the rest of the pipeline then exceeds it.
