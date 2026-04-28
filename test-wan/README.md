# Wan Video Generation Tests

This directory contains test configurations for running Wan video generation models with the `stable-diffusion.cpp` engine.

## Prerequisites

1. Build the project with Metal support:
   ```bash
   cd ~/Documents/qvac-ext-stable-diffusion.cpp
   make build-metal
   ```

2. Download model files to the `models/` directory

## Model Downloads

The Wan models are available from Hugging Face. Download and place them in `models/` directory:

### Core Models

#### Wan2.1 - Text-to-Video (T2V)

- **1.3B (Small, Fast)**
  - Safetensors: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models
  - File: `wan2.1_t2v_1.3B_fp16.safetensors`

- **14B (Large, Higher Quality)**
  - GGUF: https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/tree/main
  - File: `wan2.1_t2v_14b_Q8_0.gguf`

#### Wan2.1 - Image-to-Video (I2V)

- **14B 480P**
  - GGUF: https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/tree/main
  - File: `wan2.1_i2v_14b_480p_Q8_0.gguf`

#### Wan2.2 - Tiny Image-to-Video (TI2V)

- **5B (Tiny, Mobile-optimized)**
  - Safetensors: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models
  - File: `wan2.2_ti2v_5B_fp16.safetensors`

### Required Components

#### VAE (Video Auto Encoder)

- **Wan2.1 VAE** (for all Wan2.1 models)
  - Safetensors: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors
  - File: `wan_2.1_vae.safetensors`

- **Wan2.2 VAE** (for Wan2.2 TI2V only)
  - Safetensors: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/vae/wan2.2_vae.safetensors
  - File: `wan2.2_vae.safetensors`

#### Text Encoder (T5-XXL)

- **Safetensors**: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/text_encoders/umt5_xxl_fp16.safetensors
  - File: `umt5_xxl_fp16.safetensors` (for safetensors diffusion models)

- **GGUF**: https://huggingface.co/city96/umt5-xxl-encoder-gguf/tree/main
  - File: `umt5_xxl_Q8_0.gguf` (for gguf diffusion models)

## Running Tests

### Test All Models (if available)

```bash
./test-wan-models.sh all
```

### Test Specific Model

```bash
# Wan2.1 T2V 1.3B (fastest, good for testing)
./test-wan-models.sh wan2-1-t2v-small

# Wan2.1 T2V 14B (highest quality, requires more VRAM)
./test-wan-models.sh wan2-1-t2v-large

# Wan2.2 TI2V 5B (balanced)
./test-wan-models.sh wan2-2-t2v

# Wan2.1 I2V 14B (image-to-video)
./test-wan-models.sh wan2-1-i2v
```

## Output

Generated videos will be saved in the `output/` directory:
- `wan2-1-t2v-1.3b-test.mp4`
- `wan2-1-t2v-14b-test.mp4`
- `wan2-2-ti2v-5b-test.mp4`
- `wan2-1-i2v-14b-test.mp4`

## Performance Notes

### Memory Requirements (VRAM)

- **Wan2.1 1.3B**: ~4-6 GB
- **Wan2.1 14B**: ~8-16 GB
- **Wan2.2 5B**: ~6-8 GB

### Tips for Optimization

1. **Reduce Video Frames**: Lower `--video-frames` (default: 33)
   ```bash
   --video-frames 16  # Faster, shorter videos
   ```

2. **Offload to CPU**: Keep models in RAM, load to VRAM as needed
   ```bash
   --offload-to-cpu  # Already enabled in test script
   ```

3. **Use GGUF Quantized Models**: Smaller file sizes, less VRAM
   - `Q8_0` quantization: ~8-bit precision (recommended)
   - `Q5_K_M` quantization: ~5-bit precision (even smaller)

4. **Reduce Resolution**: Lower output dimensions
   ```bash
   -W 640 -H 360  # Instead of 832x480
   ```

## Troubleshooting

### Model not found
Ensure all required files are downloaded to the `models/` directory with exact filenames.

### Out of memory
- Reduce `--video-frames`
- Enable `--offload-to-cpu`
- Use lower resolution
- Use quantized models (GGUF format)

### Slow generation
- Metal GPU acceleration requires macOS with Apple Silicon
- Verify Metal is being used: `system_profiler SPDisplaysDataType`
- Consider using smaller models (1.3B or 5B instead of 14B)

## Command Reference

### Basic Text-to-Video
```bash
./build/bin/sd-cli -M vid_gen \
  --diffusion-model model.gguf \
  --vae vae.safetensors \
  --t5xxl t5.gguf \
  -p "a cat jumping" \
  --video-frames 33 \
  -W 832 -H 480 \
  -o output.mp4
```

### With Advanced Options
```bash
./build/bin/sd-cli -M vid_gen \
  --diffusion-model model.gguf \
  --vae vae.safetensors \
  --t5xxl t5.gguf \
  -p "a cat jumping" \
  --cfg-scale 6.0 \
  --sampling-method euler \
  --steps 20 \
  --video-frames 33 \
  --flow-shift 3.0 \
  --diffusion-fa \
  --offload-to-cpu \
  -W 832 -H 480 \
  -o output.mp4
```

## References

- [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [stable-diffusion.cpp Wan Docs](https://github.com/tetherto/qvac-ext-stable-diffusion.cpp/blob/master/docs/wan.md)
- [ComfyUI Wan Implementation](https://github.com/Comfy-Org)
