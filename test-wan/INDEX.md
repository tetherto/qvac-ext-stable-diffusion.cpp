# Wan Video Generation - Complete Documentation Index

## Quick Navigation

### 🚀 Just Getting Started?
→ Start here: **[GETTING_STARTED.md](GETTING_STARTED.md)**

### ⚡ Need Quick Reference?
→ Check: **[QUICKSTART.txt](QUICKSTART.txt)**

### 🎯 Want to Run Tests?
→ Use: **[test-wan-models.sh](test-wan-models.sh)** (executable script)

### 📖 Need Complete Details?
→ Read: **[README.md](README.md)**

### 🔧 Want Helper Functions?
→ Source: **[wan-functions.sh](wan-functions.sh)**

### ✅ Need to Verify Setup?
→ Run: **[verify-setup.sh](verify-setup.sh)** (executable script)

---

## Document Guide

### [GETTING_STARTED.md](GETTING_STARTED.md) - START HERE
**Purpose**: Step-by-step guide for first-time users
**Contains**:
- Current setup status
- What you have installed
- Step-by-step download and setup instructions
- First test walkthrough
- Troubleshooting guide

**Best for**: New users who just built the project

---

### [README.md](README.md)
**Purpose**: Comprehensive documentation
**Contains**:
- Model information and download links
- Running tests with detailed options
- Performance tips and optimization
- Command reference
- Detailed troubleshooting

**Best for**: Understanding models and fine-tuning parameters

---

### [QUICKSTART.txt](QUICKSTART.txt)
**Purpose**: Quick reference card
**Contains**:
- Project overview
- Available binaries
- Build commands
- Supported models
- Performance estimates
- Quick command template

**Best for**: Quick lookup while working

---

### [test-wan-models.sh](test-wan-models.sh)
**Purpose**: Automated test runner
**Usage**: `./test-wan-models.sh [test_name]`
**Tests available**:
- `wan2-1-t2v-small` - Fast 1.3B model
- `wan2-1-t2v-large` - High-quality 14B model
- `wan2-2-t2v` - Balanced 5B model
- `wan2-1-i2v` - Image-to-video 14B model
- `all` - Run all available tests

**Best for**: Systematic testing of different models

---

### [wan-functions.sh](wan-functions.sh)
**Purpose**: Shell helper functions for easy video generation
**Functions**:
- `wan_t2v_small` - Fast text-to-video
- `wan_t2v_large` - High-quality text-to-video
- `wan_i2v` - Image-to-video
- `wan_ti2v` - Tiny text-to-video
- `wan_benchmark` - Performance testing
- `wan_help` - Show help

**Usage**: `source wan-functions.sh` then use functions

**Best for**: Quick video generation from command line

---

### [verify-setup.sh](verify-setup.sh)
**Purpose**: Verify project setup is correct
**Checks**:
- Project directory exists
- Binaries are built
- Test infrastructure in place
- Directory structure correct
- Metal GPU availability
- Model files present

**Usage**: `./verify-setup.sh`

**Best for**: Troubleshooting setup issues

---

## File Organization

```
test-wan/
├── INDEX.md                    ← You are here
├── GETTING_STARTED.md          ← Start here for new users
├── README.md                   ← Detailed documentation
├── QUICKSTART.txt              ← Quick reference
├── test-wan-models.sh          ← Automated test runner
├── wan-functions.sh            ← Shell helper functions
├── verify-setup.sh             ← Setup verification
├── models/                     ← Download models here
├── output/                     ← Generated videos saved here
└── cache/                      ← Model cache directory
```

## Parent Project Documentation

Also available at root of project:

- **[BUILDING_WAN.md](../BUILDING_WAN.md)** - Build system and integration guide
- **[Makefile](../Makefile)** - Build system with easy targets
- **[README.md](../README.md)** - Original stable-diffusion.cpp documentation

## Quick Commands

### Verify Everything Works
```bash
./verify-setup.sh
```

### Download Models
See [README.md](README.md) for download links

### Run Tests
```bash
./test-wan-models.sh wan2-1-t2v-small
```

### Generate Videos with Functions
```bash
source wan-functions.sh
wan_t2v_small "your prompt here" output.mp4
```

### Build Project
```bash
cd .. && make build-metal
```

## Learning Path

1. **Understand the Project**
   - Read: [QUICKSTART.txt](QUICKSTART.txt)
   - Verify: `./verify-setup.sh`

2. **Set Up Models**
   - Follow: [GETTING_STARTED.md](GETTING_STARTED.md) Step 2
   - Check: `./verify-setup.sh` (should show models)

3. **Run Your First Test**
   - Follow: [GETTING_STARTED.md](GETTING_STARTED.md) Step 4

4. **Explore and Customize**
   - Load functions: `source wan-functions.sh`
   - Try examples: See [README.md](README.md)

5. **Optimize Performance**
   - Read: Performance Tips in [README.md](README.md)
   - Run: `wan_benchmark small` and `wan_benchmark large`

6. **Integrate with qvac**
   - See: [BUILDING_WAN.md](../BUILDING_WAN.md)

## Common Tasks

### Generate a Video from Text
```bash
source wan-functions.sh
wan_t2v_small "a cat playing" ~/video.mp4
```

### Animate an Image
```bash
source wan-functions.sh
wan_i2v ~/image.png "the person dances" ~/video.mp4
```

### Run Performance Test
```bash
source wan-functions.sh
wan_benchmark small
```

### Test High-Quality Model
```bash
./test-wan-models.sh wan2-1-t2v-large
```

### Check Available Models
```bash
ls -lh models/
```

### View Generated Videos
```bash
open output/
```

## Troubleshooting

### Can't find something?
→ Read: [GETTING_STARTED.md](GETTING_STARTED.md) - Troubleshooting section

### Setup issues?
→ Run: `./verify-setup.sh`

### Performance issues?
→ Read: [README.md](README.md) - Performance section

### Command not working?
→ Try: `./verify-setup.sh` to diagnose

### Need help with models?
→ See: [README.md](README.md) - Model section

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `./models/` | Place downloaded model files here |
| `./output/` | Generated videos are saved here |
| `./cache/` | Model cache directory |
| `./` | Executable scripts and documentation |
| `../build/bin/` | Built binaries (sd-cli, sd-server) |

## Key Files

| File | Type | Purpose |
|------|------|---------|
| GETTING_STARTED.md | Markdown | Step-by-step guide for new users |
| README.md | Markdown | Complete documentation |
| QUICKSTART.txt | Text | Quick reference card |
| test-wan-models.sh | Bash | Automated test runner |
| wan-functions.sh | Bash | Shell helper functions |
| verify-setup.sh | Bash | Setup verification tool |

## Recommended Reading Order

1. **First Time?** → [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Need Details?** → [README.md](README.md)
3. **Quick Lookup?** → [QUICKSTART.txt](QUICKSTART.txt)
4. **Troubleshoot?** → [GETTING_STARTED.md](GETTING_STARTED.md) Troubleshooting
5. **Run Tests?** → Use [test-wan-models.sh](test-wan-models.sh)
6. **Easy Commands?** → Source [wan-functions.sh](wan-functions.sh)

---

**Version**: April 28, 2026
**Status**: Ready for testing
**Configuration**: Metal GPU (Apple Silicon)

**Let's generate videos!** 🎬
