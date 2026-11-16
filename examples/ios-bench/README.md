## sd-bench: Metal vs CoreML benchmark (macOS and iOS)

Compares ggml Metal with the CoreML UNet and tiny-decoder sidecars ([docs/coreml.md](../../docs/coreml.md)) on SDXS-512-DreamShaper. It loads the model once, generates 1 warm-up and 3 timed images (512x512, 1 step, CFG 1, seed 42), and prints the text encoder, UNet, decoder and total time for each. The prompts are a cat portrait, a cabin at sunset and a billboard with the text "COFFEE".

```
sd-bench bench <model.gguf> <unet.mlmodelc|-> <out_prefix> [tae_decoder.mlmodelc]
```

Pass `-` to run the UNet with ggml Metal, or a `.mlmodelc` to run it on CoreML (`SD_COREML_COMPUTE_UNITS` picks the device, `cpu_ne` by default). The optional last argument runs the tiny decoder on CoreML too. Images are written as `<out_prefix>_<prompt>.png`.

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
