# Getting Started with Wan Video Generation

## Current Status

✅ **Setup Complete**
- Repository cloned and built with Metal GPU support
- Test infrastructure ready
- Binaries verified and working
- Ready to download models and run tests

## What You Have

### Built Binaries
- **sd-cli**: Located at `~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli`
- **sd-server**: Located at `~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-server`

### Test Infrastructure  
- **test-wan-models.sh**: Automated test runner
- **wan-functions.sh**: Shell helper functions
- **verify-setup.sh**: Verification script (run anytime to check setup)
- **README.md**: Complete model documentation
- **QUICKSTART.txt**: Quick reference card

### Directories
- **models/**: Place downloaded model files here
- **output/**: Generated videos are saved here
- **cache/**: Model cache directory

## Step-by-Step Guide

### Step 1: Verify Everything Works

```bash
cd ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan
./verify-setup.sh
```

You should see all checkmarks (✓).

### Step 2: Download Models

Choose the model you want to start with and download from Hugging Face.

**For Testing (Fastest, ~10 minutes setup):**
Download these 3 files to `test-wan/models/`:
1. `wan2.1_t2v_1.3B_fp16.safetensors` (2.5 GB) from [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models)
2. `wan_2.1_vae.safetensors` (1.2 GB) from [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors)
3. `umt5_xxl_fp16.safetensors` (4.6 GB) from [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/text_encoders/umt5_xxl_fp16.safetensors)

**Total: ~8.3 GB** | **Time to download: 30-60 minutes depending on speed**

See `README.md` for links to other models (14B high-quality, Wan2.2, etc.)

### Step 3: Verify Models Were Downloaded

```bash
ls -lh ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/
```

You should see the three model files listed.

### Step 4: Run Your First Test

```bash
cd ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan
./test-wan-models.sh wan2-1-t2v-small
```

**What to expect:**
- First run initializes everything (may take a minute)
- The model gets loaded into GPU memory
- Video generation starts
- Output saved to `output/wan2-1-t2v-1.3b-test.mp4`

**Estimated time: 30-60 seconds on Apple Silicon M-series**

### Step 5: Check the Generated Video

```bash
open ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/output/wan2-1-t2v-1.3b-test.mp4
```

## Using Helper Functions

Add these to your shell for easy video generation:

```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
source ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/wan-functions.sh
```

Then use:

```bash
# Generate a quick video
wan_t2v_small "a cat playing with a ball" ~/my_video.mp4

# Generate high-quality video (slower)
wan_t2v_large "a sunset over mountains" ~/my_video.mp4

# Animate an image
wan_i2v ~/image.png "the person dances" ~/my_video.mp4

# Show help
wan_help

# Run benchmark
wan_benchmark small
```

## Customizing Video Generation

### Simple Command

```bash
~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli -M vid_gen \
  --diffusion-model ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan2.1_t2v_1.3B_fp16.safetensors \
  --vae ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/wan_2.1_vae.safetensors \
  --t5xxl ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/umt5_xxl_fp16.safetensors \
  -p "your prompt here" \
  -o output.mp4
```

### Full Command with Options

```bash
~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli -M vid_gen \
  --diffusion-model models/wan2.1_t2v_1.3B_fp16.safetensors \
  --vae models/wan_2.1_vae.safetensors \
  --t5xxl models/umt5_xxl_fp16.safetensors \
  -p "a cat jumping over a fence" \
  --cfg-scale 6.0 \          # Prompt strength (higher = stricter adherence)
  --sampling-method euler \   # Sampling algorithm
  --steps 20 \               # Diffusion steps (higher = better quality)
  --video-frames 33 \        # Number of frames (33 = ~1.3 seconds at 24fps)
  --flow-shift 3.0 \         # Motion dynamics
  -W 832 -H 480 \            # Resolution
  --diffusion-fa \           # Flash attention (faster, less memory)
  --offload-to-cpu \         # Store models in RAM, load to GPU when needed
  -o output.mp4
```

## Troubleshooting

### Models Not Found
Make sure file names match exactly (case-sensitive):
- ✓ `wan2.1_t2v_1.3B_fp16.safetensors`
- ✗ `Wan2.1_t2v_1.3b.safetensors` (wrong case/name)

### Out of Memory
```bash
# Reduce frames for shorter videos
--video-frames 16

# Reduce resolution
-W 640 -H 360

# Enable CPU offloading (slightly slower)
--offload-to-cpu
```

### Very Slow Generation
- Verify Metal GPU is being used: `system_profiler SPDisplaysDataType | grep Metal`
- Try smaller model (1.3B instead of 14B)
- Check disk speed (models may be on slow storage)

### No GPU Acceleration
The project was built with Metal support:
```bash
# Verify build configuration
file ~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli
# Should show: "Mach-O 64-bit executable arm64"
```

## Model Comparison

| Model | Size | Speed | Quality | VRAM | Best For |
|-------|------|-------|---------|------|----------|
| Wan2.1 1.3B | 2.5 GB | 30-60s | Good | 4 GB | Testing, quick videos |
| Wan2.1 14B | 7.5 GB | 3-5 min | Excellent | 8-16 GB | High quality results |
| Wan2.2 TI2V 5B | 5.0 GB | 1-2 min | Good | 6 GB | Mobile-friendly |
| Wan2.1 I2V 14B | 7.5 GB | 3-5 min | Excellent | 8-16 GB | Animating images |

## Next Steps

1. ✅ **Verify setup** → Run `verify-setup.sh`
2. ⬇️ **Download models** → See Step 2 above
3. ▶️ **Generate first video** → Run test command
4. 🎨 **Customize prompts** → Try your own descriptions
5. 📊 **Benchmark performance** → Run `wan_benchmark small`
6. 📚 **Explore options** → See README.md for advanced features

## Performance Tips

### For Speed
```bash
--cfg-scale 3.5      # Lower = faster, less precise
--steps 8            # Fewer steps = faster but lower quality
--video-frames 16    # Shorter videos
-W 640 -H 360        # Lower resolution
```

### For Quality
```bash
--cfg-scale 7.5      # Higher = stricter adherence to prompt
--steps 30           # More steps = better quality
--video-frames 48    # More frames = smoother motion
-W 832 -H 480        # Higher resolution
```

### For Memory
```bash
--offload-to-cpu     # Load models from RAM on demand
--vae-tiling         # Process VAE in tiles
```

## Getting Help

- **Quick reference**: `cat ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/QUICKSTART.txt`
- **Detailed docs**: `cat ~/Documents/qvac-ext-stable-diffusion.cpp/BUILDING_WAN.md`
- **Model info**: `cat ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/README.md`
- **CLI help**: `~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli --help`

## Links

- [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [Model Zoo](https://huggingface.co/Comfy-Org)

---

**Ready to generate videos!** 🎬

Start with: `cd ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan && ./verify-setup.sh`
