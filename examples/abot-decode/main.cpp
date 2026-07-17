// sd-abot-decode — MVP stepping stone for ABot-World in QVAC sd.cpp.
//
// Loads the taehv (taew2_2) decoder GGUF and a fixed-scene "walk" of precomputed
// latents (raw f32, produced offline by the reference pipeline), decodes them to
// frames with the native ggml engine, and writes a video. No Python, no DiT,
// no T5 — this proves the native taehv decode + the S3/GPU/build/video/validate
// CI pipeline end to end. The DiT causal denoise replaces the "load latents"
// step in the follow-up (see docs/abot_world_mvp_spec.md).
//
// Latent .bin layout: raw little-endian float32 in torch [T, C, H, W] order
// (row-major, W fastest), which equals ggml [W, H, C, T]. Dims are passed on
// the command line (from the scene's walk.json).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ggml_extend.hpp"
#include "ggml_extend_backend.h"
#include "model.h"
#include "stable-diffusion.h"
#include "tensor.hpp"
#include "vae.hpp"
#include "tae.hpp"

#include "../common/media_io.h"

struct Args {
    std::string taehv;
    std::string latents;
    std::string out = "walk.avi";
    int W = 0, H = 0, C = 48, T = 0;
    int fps = 12;
    int n_threads = 4;
    std::string backend = "gpu";
};

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (k == "--taehv") a.taehv = next();
        else if (k == "--latents") a.latents = next();
        else if (k == "--out") a.out = next();
        else if (k == "--width") a.W = std::stoi(next());
        else if (k == "--height") a.H = std::stoi(next());
        else if (k == "--channels") a.C = std::stoi(next());
        else if (k == "--frames") a.T = std::stoi(next());  // latent frames total
        else if (k == "--fps") a.fps = std::stoi(next());
        else if (k == "--threads") a.n_threads = std::stoi(next());
        else if (k == "--backend") a.backend = next();
        else { fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    // latent grid: stream/16 -> here we receive latent W/H directly
    if (a.taehv.empty() || a.latents.empty() || a.W == 0 || a.H == 0 || a.T == 0) {
        fprintf(stderr, "usage: sd-abot-decode --taehv <gguf> --latents <f32.bin> "
                        "--width <latW> --height <latH> --channels <C=48> "
                        "--frames <T> [--out walk.avi] [--fps 12] [--backend gpu]\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    // ── load latents (raw f32, [T,C,H,W] == ggml [W,H,C,T]) ──
    size_t numel = (size_t)a.W * a.H * a.C * a.T;
    std::vector<float> data(numel);
    {
        std::ifstream f(a.latents, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open latents: %s\n", a.latents.c_str()); return 1; }
        f.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(numel * sizeof(float)));
        if ((size_t)f.gcount() != numel * sizeof(float)) {
            fprintf(stderr, "latents size mismatch: got %lld bytes, expected %zu\n",
                    (long long)f.gcount(), numel * sizeof(float));
            return 1;
        }
    }
    sd::Tensor<float> z({a.W, a.H, a.C, a.T}, std::move(data));
    printf("loaded latents [W=%d H=%d C=%d T=%d]\n", a.W, a.H, a.C, a.T);

    // ── backend + model load (mirror upscaler.cpp standalone pattern) ──
    SDBackendManager backend_manager;
    std::string error;
    if (!backend_manager.init(a.backend.c_str(), nullptr, false, false, false, false, &error)) {
        fprintf(stderr, "backend init failed: %s\n", error.c_str());
        return 1;
    }
    ggml_backend_t backend = backend_manager.runtime_backend(SDBackendModule::VAE);
    ggml_backend_t pbackend = backend_manager.params_backend(SDBackendModule::VAE);
    if (backend == nullptr || pbackend == nullptr) {
        fprintf(stderr, "VAE backend init failed\n");
        return 1;
    }
    printf("taehv backend: %s\n", sd_backend_is_cpu(backend) ? "CPU" : "GPU");

    ModelLoader ml;
    if (!ml.init_from_file(a.taehv, "decoder")) {
        fprintf(stderr, "init model loader failed: %s\n", a.taehv.c_str());
        return 1;
    }

    auto vae = std::make_shared<TinyVideoAutoEncoder>(
        backend, pbackend, ml.get_tensor_storage_map(), "decoder",
        /*decoder_only=*/true, VERSION_ABOT_WORLD);
    if (!vae->alloc_params_buffer()) {
        fprintf(stderr, "alloc params buffer failed\n");
        return 1;
    }
    std::map<std::string, ggml_tensor*> tensors;
    vae->get_param_tensors(tensors, "decoder");
    if (!ml.load_tensors(tensors, {}, a.n_threads)) {
        fprintf(stderr, "load taehv tensors failed\n");
        return 1;
    }
    printf("taehv loaded (%zu tensors)\n", tensors.size());

    // ── decode ──
    sd_tiling_params_t tiling = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
    sd::Tensor<float> out = vae->decode(a.n_threads, z, tiling, /*decode_video=*/true);
    if (out.empty()) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    int num_frames = (int)out.shape()[2];
    printf("decoded %d frames, shape [%lld,%lld,%lld,%lld]\n", num_frames,
           (long long)out.shape()[0], (long long)out.shape()[1],
           (long long)out.shape()[2], (long long)out.shape()[3]);

    std::vector<sd_image_t> images((size_t)num_frames);
    for (int i = 0; i < num_frames; i++) {
        images[(size_t)i] = tensor_to_sd_image(out, i);
    }

    int rc = create_mjpg_avi_from_sd_images(a.out.c_str(), images.data(), num_frames, a.fps, 90);
    if (rc != 0) {
        fprintf(stderr, "video write failed (rc=%d)\n", rc);
        return 1;
    }
    printf("wrote %s (%d frames @ %d fps)\n", a.out.c_str(), num_frames, a.fps);
    return 0;
}
