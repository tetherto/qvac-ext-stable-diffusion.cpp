# Metal IM2COL_3D Support - Patch Instructions

## Status

Wan video generation models require the `IM2COL_3D` operation for 3D convolutions. This operation is **not yet implemented in the stable GGML Metal backend**, which causes the error:

```
[ERROR] ggml_metal_op_encode_impl: error: unsupported op 'IM2COL_3D'
```

## Solution

A patch implementing Metal IM2COL_3D support is available in CLDawes' branch at:
https://github.com/CLDawes/ggml/tree/patch-qwen-image

This includes:
- Metal support for `IM2COL_3D` operation
- Metal support for `DIAG_MASK_INF` operation  
- Fix for `PAD` operation

## How to Apply the Patch

### Option 1: Apply to This Branch (Recommended)

```bash
cd ~/Documents/qvac-ext-stable-diffusion.cpp/ggml

# Add CLDawes remote
git remote add cldawes https://github.com/CLDawes/ggml.git

# Fetch the patch branch
git fetch cldawes patch-qwen-image

# Merge with strategy to accept theirs (patches' version)
git merge -X theirs cldawes/patch-qwen-image

# Build with Metal support
cd ..
make clean
make build-metal
```

### Option 2: Manual Patch Application

The key commit is `3455d939 metal: add support for opt_im2col_3d` from CLDawes' repository.

You can view the full patch at:
```bash
git show 3455d939 > im2col_3d.patch
```

Then apply it with:
```bash
git apply im2col_3d.patch
```

### Option 3: Use Latest GGML Master

A newer version of GGML may have this support merged. Try updating the submodule:

```bash
cd ~/Documents/qvac-ext-stable-diffusion.cpp/ggml
git fetch origin master
git merge origin/master
# Then rebuild
```

## Current Workaround

Until Metal IM2COL_3D support is applied, use CPU-only build:

```bash
cd ~/Documents/qvac-ext-stable-diffusion.cpp
make clean
make build
```

Your M3 Ultra will still provide good performance even with CPU compute, though slower than GPU acceleration.

## Performance Comparison

| Configuration | Speed | Notes |
|---------------|-------|-------|
| Metal IM2COL_3D (Patched) | 15-30 sec (1.3B) | Recommended - GPU accelerated |
| CPU-only | 30-90 sec (1.3B) | Current workaround - still usable |
| Metal (Current) | ❌ Crashes | IM2COL_3D not supported |

## PR Status

- **ggml-org/llama.cpp PR #16669**: "metal: add ops DIAG_MASK_INF, IM2COL_3D, fix op PAD"
  - Status: Under review
  - Submitted by: CLDawes
  - Timeline: TBD for merge

## References

- Issue: [leejet/stable-diffusion.cpp #850](https://github.com/leejet/stable-diffusion.cpp/issues/850)
- Patch: [CLDawes/ggml@patch-qwen-image](https://github.com/CLDawes/ggml/tree/patch-qwen-image)
- PR: [ggml-org/llama.cpp #16669](https://github.com/ggml-org/llama.cpp/pull/16669)

## Troubleshooting

### Merge Conflicts

If you encounter merge conflicts when applying the patch:

```bash
# Use theirs (CLDawes' version)
git merge -X theirs cldawes/patch-qwen-image

# Or manually resolve conflicts and continue
git status  # See conflicted files
# Edit files...
git add .
git commit
```

### Build Errors After Patching

If you get compilation errors after applying the patch, it may be due to API changes between ggml versions. Try:

1. Check if the patch needs ggml submodule updated to a specific commit
2. Review the full CLDawes branch history for context
3. Fall back to CPU-only build

### Still Getting IM2COL_3D Error

If the error persists after patching:

1. Verify the patch was applied: `grep -r "im2col_3d" ggml/src/ggml-metal/`
2. Clean and rebuild: `make clean && make build-metal`
3. Check that Metal backend is being used in compilation output

## Testing the Patch

Once patched and built, test with:

```bash
cd test-wan
./test-wan-models.sh wan2-1-t2v-small
```

You should see no more `unsupported op 'IM2COL_3D'` errors.

---

**Last Updated**: April 28, 2026  
**Status**: Awaiting upstream merge  
**Workaround**: CPU-only build available
