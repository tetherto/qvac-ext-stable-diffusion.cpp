# Building and Testing Wan Video Generation

This guide walks through building the stable-diffusion.cpp extension with Wan support and running tests.

## Summary

The repository has been successfully cloned, built, and configured with test infrastructure for Wan video generation models (Wan2.1 and Wan2.2).

### What Was Done

1. **Repository Setup**
   - Cloned: `https://github.com/tetherto/qvac-ext-stable-diffusion.cpp.git`
   - Location: `~/Documents/qvac-ext-stable-diffusion.cpp`
   - Initialized git submodules (ggml backend)

2. **Build System**
   - Created `Makefile` wrapper for CMake
   - Supports multiple build configurations:
     - `make build` - CPU only
     - `make build-metal` - Apple Silicon GPU (Metal)
     - `make build-cuda` - NVIDIA GPU
     - `make build-vulkan` - Vulkan GPU
   - Successfully built with Metal GPU support

3. **Binary Locations**
   - CLI: `~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli`
   - Server: `~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-server`

4. **Test Infrastructure**
   - Test directory: `~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/`
   - Test script: `test-wan/test-wan-models.sh`
   - Helper functions: `test-wan/wan-functions.sh`
   - Documentation: `test-wan/README.md`
   - Model storage: `test-wan/models/`
   - Output storage: `test-wan/output/`

## Quick Start

### 1. Verify Build

```bash
~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli --help
```

### 2. Download Models

Download Wan model files to `~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/models/`

**Recommended Starting Models:**
- `wan2.1_t2v_1.3B_fp16.safetensors` (fast test)
- `wan_2.1_vae.safetensors` (required)
- `umt5_xxl_fp16.safetensors` (required)

See `test-wan/README.md` for complete download links.

### 3. Run Tests

```bash
cd ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan

# Test all available models
./test-wan-models.sh all

# Test specific model
./test-wan-models.sh wan2-1-t2v-small
```

### 4. Use Helper Functions

```bash
# Load functions into shell
source ~/Documents/qvac-ext-stable-diffusion.cpp/test-wan/wan-functions.sh

# Generate video
wan_t2v_small "a cat jumping" ~/video.mp4

# Show help
wan_help
```

## Directory Structure

```
qvac-ext-stable-diffusion.cpp/
├── Makefile                    # New: Build wrapper for CMake
├── build/                      # Build artifacts
│   └── bin/
│       ├── sd-cli              # Video/image generation CLI
│       └── sd-server           # REST API server
├── test-wan/                   # New: Test infrastructure
│   ├── README.md               # Detailed test documentation
│   ├── test-wan-models.sh      # Test runner script
│   ├── wan-functions.sh        # Helper functions
│   ├── models/                 # Model storage
│   ├── output/                 # Generated videos
│   └── cache/                  # Model cache
├── src/                        # Source code
├── docs/                       # Documentation
│   ├── wan.md                  # Wan model documentation
│   ├── build.md                # Build instructions
│   └── ...
└── ...
```

## Available Make Targets

```bash
cd ~/Documents/qvac-ext-stable-diffusion.cpp

# Build for Apple Silicon (Metal GPU)
make build-metal

# Build CPU only
make build

# Build with CUDA (if NVIDIA GPU available)
make build-cuda

# Build with Vulkan
make build-vulkan

# Clean build artifacts
make clean

# Show help
make help
```

## Testing Wan Models

### Wan2.1 Text-to-Video (T2V)

**1.3B Model (Fast):**
```bash
./build/bin/sd-cli -M vid_gen \
  --diffusion-model test-wan/models/wan2.1_t2v_1.3B_fp16.safetensors \
  --vae test-wan/models/wan_2.1_vae.safetensors \
  --t5xxl test-wan/models/umt5_xxl_fp16.safetensors \
  -p "a lovely cat" \
  --video-frames 33 \
  -W 832 -H 480 \
  --offload-to-cpu \
  -o test-wan/output/test.mp4
```

**14B Model (High Quality):**
```bash
./build/bin/sd-cli -M vid_gen \
  --diffusion-model test-wan/models/wan2.1_t2v_14b_Q8_0.gguf \
  --vae test-wan/models/wan_2.1_vae.safetensors \
  --t5xxl test-wan/models/umt5_xxl_Q8_0.gguf \
  -p "a serene landscape" \
  --video-frames 33 \
  -W 832 -H 480 \
  --offload-to-cpu \
  -o test-wan/output/test.mp4
```

### Wan2.1 Image-to-Video (I2V)

```bash
./build/bin/sd-cli -M vid_gen \
  --diffusion-model test-wan/models/wan2.1_i2v_14b_480p_Q8_0.gguf \
  --vae test-wan/models/wan_2.1_vae.safetensors \
  --t5xxl test-wan/models/umt5_xxl_Q8_0.gguf \
  --clip_vision test-wan/models/clip_vision_h.safetensors \
  -i input_image.png \
  -p "the subject moves" \
  --video-frames 33 \
  -W 480 -H 832 \
  --offload-to-cpu \
  -o test-wan/output/test.mp4
```

## Performance Tips

### Memory Optimization
- Use `--offload-to-cpu` to keep models in RAM
- Reduce `--video-frames` for shorter videos
- Use quantized GGUF models (Q8_0, Q5_K_M)

### Speed Optimization
- Use smaller models (1.3B for testing)
- Reduce video resolution (`-W 640 -H 360`)
- Use Metal GPU: Built with `-DSD_METAL=ON`

### Quality Settings
```bash
# Fast generation
--cfg-scale 3.5 --steps 8 --video-frames 16

# Balanced
--cfg-scale 6.0 --steps 20 --video-frames 33

# High quality
--cfg-scale 7.5 --steps 30 --video-frames 48
```

## Model Formats

- **Safetensors** (.safetensors): Higher quality, larger files
- **GGUF** (.gguf): Quantized, smaller files, faster loading

Quantization levels:
- `Q8_0`: 8-bit (recommended, good balance)
- `Q5_K_M`: 5-bit (smaller, acceptable quality)
- `F16`: 16-bit float (highest quality, largest size)

## Troubleshooting

### Build Issues

**CMake not found:**
```bash
brew install cmake
```

**Metal not working:**
Verify Metal GPU support:
```bash
system_profiler SPDisplaysDataType | grep "Metal"
```

### Runtime Issues

**Models not found:**
- Check file names match exactly (case-sensitive)
- Verify paths in commands
- Use absolute paths to avoid issues

**Out of memory:**
- Enable `--offload-to-cpu`
- Reduce `--video-frames`
- Use smaller models

**Slow generation:**
- Verify Metal is being used: `system_profiler SPDisplaysDataType`
- Try smaller models first (1.3B)
- Check disk I/O: models may be on slow storage

## Integration with qvac Project

The stable-diffusion.cpp extension can be integrated into the main qvac project:

```bash
# From qvac root
cd ~/Documents/qvac
# Link or copy the built binaries
ln -s ~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-cli ./bin/sd-cli
ln -s ~/Documents/qvac-ext-stable-diffusion.cpp/build/bin/sd-server ./bin/sd-server
```

## References

- [stable-diffusion.cpp Repository](https://github.com/leejet/stable-diffusion.cpp)
- [Wan2.1 Paper & Code](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2 Paper & Code](https://github.com/Wan-Video/Wan2.2)
- [ComfyUI Wan Implementation](https://github.com/Comfy-Org)

## Next Steps

1. Download the first set of models (see `test-wan/README.md`)
2. Run the test script: `./test-wan/test-wan-models.sh wan2-1-t2v-small`
3. Evaluate quality and speed
4. Adjust parameters as needed for your use case
5. Integrate with qvac pipeline

---

**Built:** April 28, 2026
**Configuration:** Metal GPU (Apple Silicon)
**Status:** Ready for testing
