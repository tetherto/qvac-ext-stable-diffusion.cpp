#!/bin/bash
mkdir -p models
cd models
echo "Downloading Wan2.1 1.3B T2V Model..."
curl -L -o wan2.1_t2v_1.3B_fp16.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"
echo "Downloading Wan 2.1 VAE..."
curl -L -o wan_2.1_vae.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
echo "Downloading UMT5-XXL Text Encoder..."
curl -L -o umt5_xxl_fp16.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"
echo "✓ All downloads complete!"
ls -lh
