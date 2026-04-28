#!/bin/bash

# Setup script for Wan video generation tests
# Run this once to verify everything is set up correctly

set -e

PROJECT_DIR="$HOME/Documents/qvac-ext-stable-diffusion.cpp"
TEST_DIR="$PROJECT_DIR/test-wan"
CLI="$PROJECT_DIR/build/bin/sd-cli"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          WAN VIDEO GENERATION - VERIFICATION SCRIPT           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if project exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ ERROR: Project not found at $PROJECT_DIR"
    echo "   Please ensure you cloned the repository"
    exit 1
fi
echo "✓ Project directory found"

# Check if CLI was built
if [ ! -f "$CLI" ]; then
    echo "❌ ERROR: sd-cli not found at $CLI"
    echo "   Please build the project first:"
    echo "   cd $PROJECT_DIR && make build-metal"
    exit 1
fi
echo "✓ sd-cli binary found"

# Check if test infrastructure exists
if [ ! -f "$TEST_DIR/test-wan-models.sh" ]; then
    echo "❌ ERROR: Test infrastructure not found"
    exit 1
fi
echo "✓ Test infrastructure found"

# Check directories
for dir in models output cache; do
    if [ ! -d "$TEST_DIR/$dir" ]; then
        echo "❌ ERROR: $dir directory not found"
        exit 1
    fi
    echo "✓ $dir directory exists"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MODELS REQUIRED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To run tests, download these files to: $TEST_DIR/models/"
echo ""
echo "Minimum (to get started):"
echo "  1. wan2.1_t2v_1.3B_fp16.safetensors  (~2.5 GB)"
echo "  2. wan_2.1_vae.safetensors           (~1.2 GB)"
echo "  3. umt5_xxl_fp16.safetensors         (~4.6 GB)"
echo ""
echo "For high quality:"
echo "  4. wan2.1_t2v_14b_Q8_0.gguf          (~7.5 GB)"
echo "  5. umt5_xxl_Q8_0.gguf                (~2.4 GB)"
echo ""
echo "See $TEST_DIR/README.md for download links"
echo ""

# Check if models are present
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "AVAILABLE MODELS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

models_found=0
for model in $TEST_DIR/models/*; do
    if [ -f "$model" ]; then
        size=$(du -h "$model" | cut -f1)
        echo "  ✓ $(basename "$model") ($size)"
        ((models_found++))
    fi
done

if [ $models_found -eq 0 ]; then
    echo "  ⚠ No models found - please download them first"
else
    echo ""
    echo "✓ $models_found model file(s) found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "QUICK START"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create convenient aliases for the user's shell profile
SHELL_RC=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    echo "To add helpful functions to your shell, run:"
    echo ""
    echo "  echo 'source $TEST_DIR/wan-functions.sh' >> $SHELL_RC"
    echo "  source $SHELL_RC"
    echo ""
fi

echo "Then use these commands:"
echo ""
echo "  wan_t2v_small \"a cat\" output.mp4"
echo "  wan_t2v_large \"ocean waves\" output.mp4"
echo "  wan_help"
echo ""

echo "Or run the automated test suite:"
echo ""
echo "  cd $TEST_DIR"
echo "  ./test-wan-models.sh wan2-1-t2v-small"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SYSTEM INFO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Metal GPU
if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    echo "✓ Metal GPU: Available"
else
    echo "⚠ Metal GPU: Not detected"
fi

# Check memory
if command -v vm_stat &> /dev/null; then
    total_mem=$(vm_stat 2>/dev/null | grep "Pages wired" | awk '{print $4}' || echo "unknown")
    echo "  System Memory: Check with 'vm_stat' or 'memory_pressure'"
fi

echo ""
echo "✅ VERIFICATION COMPLETE"
echo ""
echo "Next step: Download model files to $TEST_DIR/models/"
echo ""
