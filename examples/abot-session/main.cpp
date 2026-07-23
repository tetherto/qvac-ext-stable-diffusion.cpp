// sd-abot-session — drive the public sd_abot_session_* C API for a fixed-scene
// interactive walk, writing decoded frames as PNGs.
//
// Modes:
//   --mode walk    (default) full native pipeline via the C API: DiT walk +
//                  taehv decode, one action per block.
//   --mode decode  decode-only cross-check: load latents produced by
//                  sd-abot-walk (raw f32 [T,C,H,W]) and decode them natively
//                  block-by-block with the session's overlap logic, so the
//                  PNGs can be PSNR-compared against the reference (Python)
//                  decode of the same latents.
//
// Usage:
//   sd-abot-session --dit dit.gguf --taehv taew2_2.gguf --scene scene.safetensors
//                   [--actions idle:1,W:3] [--threads N] [--seed 42] [--outdir frames]
//   sd-abot-session --mode decode --taehv taew2_2.gguf --latents walk.bin
//                   --lat-w 52 --lat-h 30 [--outdir frames]

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "stable-diffusion.h"

#include "abot_world.hpp"
#include "ggml_extend_backend.h"

static std::vector<uint8_t> parse_actions(const std::string& spec);

// minimal .npy f32 loader (little-endian, C-order) for golden replay
static bool load_npy_f32(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    char magic[6];
    f.read(magic, 6);
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint16_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    f.seekg(0, std::ios::end);
    const std::streamoff total = f.tellg();
    const std::streamoff data0 = 10 + hlen;
    out.resize(static_cast<size_t>((total - data0) / 4));
    f.seekg(data0);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * 4));
    return f.good();
}

// Validation walk driving AbotWalkSession directly (bypasses the C API so the
// golden noise can be injected via noise_override and final latents dumped in
// sd-abot-walk's format for compare_walk.py). Exercises whichever walk path
// ABOT_KV_CACHE selects.
static int run_walkval(const std::string& dit, const std::string& taehv, const std::string& scene,
                       const std::string& golden, const std::string& actions_spec,
                       const std::string& latents_out, int blocks_n,
                       int n_threads, uint64_t seed, const std::string& backend_spec) {
    SDBackendManager bm;
    std::string err;
    if (!bm.init(backend_spec.c_str(), nullptr, false, false, false, false, &err)) {
        fprintf(stderr, "backend init failed: %s\n", err.c_str());
        return 1;
    }
    ABOT::AbotWalkSession session;
    if (!session.load(bm.runtime_backend(SDBackendModule::DIFFUSION),
                      bm.params_backend(SDBackendModule::DIFFUSION),
                      bm.runtime_backend(SDBackendModule::VAE),
                      bm.params_backend(SDBackendModule::VAE),
                      dit, taehv, scene, {}, seed, n_threads)) {
        fprintf(stderr, "session load failed\n");
        return 1;
    }
    if (!golden.empty()) {
        session.noise_override = [golden](int block, int step, float* dst, size_t n) -> bool {
            char nm[512];
            if (step < 0) {
                snprintf(nm, sizeof(nm), "%s/b%d_s0_xt.npy", golden.c_str(), block);
            } else {
                snprintf(nm, sizeof(nm), "%s/b%d_s%d_eps.npy", golden.c_str(), block, step);
            }
            std::vector<float> g;
            if (!load_npy_f32(nm, g) || g.size() != n) {
                fprintf(stderr, "golden noise load failed: %s (%zu vs %zu)\n", nm, g.size(), n);
                return false;
            }
            memcpy(dst, g.data(), n * sizeof(float));
            return true;
        };
    }
    std::vector<uint8_t> acts = parse_actions(actions_spec);
    if (blocks_n > 0) {
        acts.resize(static_cast<size_t>(blocks_n), acts.empty() ? 0 : acts.back());
    }
    std::ofstream lat_f;
    if (!latents_out.empty()) {
        lat_f.open(latents_out, std::ios::binary);
    }
    for (size_t b = 0; b < acts.size(); b++) {
        std::vector<std::vector<float>> frames;
        const int64_t t0 = ggml_time_ms();
        if (!session.step_latents(acts[b], frames)) {
            fprintf(stderr, "step %zu failed\n", b);
            return 1;
        }
        const int64_t t1 = ggml_time_ms();
        printf("block %zu (mask 0x%02x): latents in %.1fs\n", b, acts[b], (t1 - t0) / 1000.0f);
        if (lat_f.is_open()) {
            for (auto& fr : frames) {
                lat_f.write(reinterpret_cast<const char*>(fr.data()),
                            static_cast<std::streamsize>(fr.size() * sizeof(float)));
            }
        }
    }
    printf("walkval done: %zu blocks\n", acts.size());
    return 0;
}

static std::vector<uint8_t> parse_actions(const std::string& spec) {
    // "idle:1,W:3" -> per-block key bitmask (W,A,S,D,I,J,K,L = bits 0..7).
    // "idle"/"none" = no keys; any other character must be a valid walk key.
    std::vector<uint8_t> blocks;
    const std::string keys = "WASDIJKL";
    size_t p               = 0;
    while (p < spec.size()) {
        size_t c          = spec.find(',', p);
        std::string chunk = spec.substr(p, c == std::string::npos ? std::string::npos : c - p);
        p                 = c == std::string::npos ? spec.size() : c + 1;
        size_t colon      = chunk.find(':');
        std::string key   = chunk.substr(0, colon);
        int n             = colon == std::string::npos ? 1 : std::stoi(chunk.substr(colon + 1));
        uint8_t m         = 0;
        std::string upper;
        for (char ch : key) {
            upper.push_back(static_cast<char>(toupper(ch)));
        }
        if (upper != "IDLE" && upper != "NONE" && !upper.empty()) {
            for (char ch : upper) {
                size_t bit = keys.find(ch);
                if (bit == std::string::npos) {
                    fprintf(stderr, "unknown walk key '%c' in --actions chunk '%s' (valid: W,A,S,D,I,J,K,L or 'idle')\n",
                            ch, key.c_str());
                    exit(2);
                }
                m |= static_cast<uint8_t>(1u << bit);
            }
        }
        for (int i = 0; i < n; i++) {
            blocks.push_back(m);
        }
    }
    return blocks;
}

static bool write_png(const std::string& path, const uint8_t* rgb, int w, int h) {
    return stbi_write_png(path.c_str(), w, h, 3, rgb, w * 3) != 0;
}

static int run_walk(const std::string& dit, const std::string& taehv, const std::string& scene,
                    const std::string& actions_spec, const std::string& outdir,
                    int threads, int64_t seed, const std::string& backend) {
    sd_abot_session_params_t params;
    sd_abot_session_params_init(&params);
    params.dit_model_path = dit.c_str();
    params.taehv_path     = taehv.c_str();
    params.scene_path     = scene.c_str();
    params.backend        = backend.c_str();
    params.n_threads      = threads;
    params.seed           = seed;

    sd_abot_session_t* session = sd_abot_session_new(&params);
    if (session == nullptr) {
        fprintf(stderr, "session creation failed\n");
        return 1;
    }

    std::vector<uint8_t> actions = parse_actions(actions_spec);
    int frame_no                 = 0;
    for (size_t b = 0; b < actions.size(); b++) {
        int n_frames      = 0;
        int64_t t0        = ggml_time_ms();
        sd_image_t* fr    = sd_abot_session_step(session, actions[b], &n_frames);
        int64_t t1        = ggml_time_ms();
        if (fr == nullptr) {
            fprintf(stderr, "step %zu failed\n", b);
            sd_abot_session_free(session);
            return 1;
        }
        printf("block %zu (mask 0x%02x): %d frames %ux%u in %.1fs\n",
               b, actions[b], n_frames, fr[0].width, fr[0].height, (t1 - t0) / 1000.0f);
        for (int i = 0; i < n_frames; i++) {
            char name[512];
            snprintf(name, sizeof(name), "%s/frame_%04d.png", outdir.c_str(), frame_no++);
            if (!write_png(name, fr[i].data, static_cast<int>(fr[i].width), static_cast<int>(fr[i].height))) {
                fprintf(stderr, "cannot write %s\n", name);
            }
        }
        sd_abot_session_frames_free(fr, n_frames);
    }
    sd_abot_session_free(session);
    printf("done: %d frames -> %s\n", frame_no, outdir.c_str());
    return 0;
}

static int run_decode(const std::string& taehv, const std::string& latents_path,
                      int lat_w, int lat_h, int lat_c, int fpb,
                      const std::string& outdir, int threads, const std::string& backend) {
    std::ifstream f(latents_path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", latents_path.c_str());
        return 1;
    }
    f.seekg(0, std::ios::end);
    size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> lat(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(lat.data()), static_cast<std::streamsize>(bytes));
    const size_t fel = static_cast<size_t>(lat_c) * lat_h * lat_w;
    const int T      = static_cast<int>(lat.size() / fel);
    printf("latents: %d frames %dx%dx%d\n", T, lat_w, lat_h, lat_c);

    SDBackendManager bm;
    std::string err;
    if (!bm.init(backend.c_str(), nullptr, false, false, false, false, &err)) {
        fprintf(stderr, "backend init failed: %s\n", err.c_str());
        return 1;
    }
    ModelLoader tl;
    if (!tl.init_from_file(taehv, "tae.")) {
        return 1;
    }
    TinyVideoAutoEncoder tae(bm.runtime_backend(SDBackendModule::VAE),
                             bm.params_backend(SDBackendModule::VAE),
                             tl.get_tensor_storage_map(), "decoder", true, VERSION_ABOT_WORLD);
    if (!tae.alloc_params_buffer()) {
        return 1;
    }
    std::map<std::string, ggml_tensor*> tensors;
    tae.get_param_tensors(tensors, "tae");
    if (!tl.load_tensors(tensors, {}, threads)) {
        return 1;
    }
    printf("taehv loaded (is_wide path auto-detected)\n");

    int frame_no = 0;
    for (int b0 = 0; b0 < T; b0 += fpb) {
        const int lead = std::min(b0, 3);
        const int Tb   = std::min(fpb, T - b0) + lead;
        sd::Tensor<float> z({lat_w, lat_h, Tb, lat_c});
        for (int t = 0; t < Tb; t++) {
            const float* src = lat.data() + static_cast<size_t>(b0 - lead + t) * fel;
            for (int c = 0; c < lat_c; c++) {
                memcpy(z.data() + (static_cast<size_t>(c) * Tb + t) * lat_w * lat_h,
                       src + static_cast<size_t>(c) * lat_w * lat_h,
                       static_cast<size_t>(lat_w) * lat_h * sizeof(float));
            }
        }
        sd_tiling_params_t no_tiling = {};
        sd::Tensor<float> px         = tae.decode(threads, z, no_tiling, true);
        if (px.empty()) {
            fprintf(stderr, "decode failed at block %d\n", b0 / fpb);
            return 1;
        }
        const int64_t Wp = px.shape()[0], Hp = px.shape()[1], Tp = px.shape()[2];
        const int64_t keep  = lead > 0 ? 4 * (Tb - lead) : Tp;
        const int64_t start = Tp - keep;
        std::vector<uint8_t> rgb(static_cast<size_t>(Wp) * Hp * 3);
        for (int64_t t = 0; t < keep; t++) {
            for (int64_t ch = 0; ch < 3; ch++) {
                const float* sp = px.data() + (ch * Tp + start + t) * Wp * Hp;
                for (int64_t i = 0; i < Wp * Hp; i++) {
                    float v = sp[i] < 0.0f ? 0.0f : (sp[i] > 1.0f ? 1.0f : sp[i]);
                    rgb[static_cast<size_t>(i) * 3 + ch] = static_cast<uint8_t>(v * 255.0f + 0.5f);
                }
            }
            char name[512];
            snprintf(name, sizeof(name), "%s/frame_%04d.png", outdir.c_str(), frame_no++);
            if (!write_png(name, rgb.data(), static_cast<int>(Wp), static_cast<int>(Hp))) {
                fprintf(stderr, "cannot write %s\n", name);
            }
        }
        printf("block %d: +%lld px frames\n", b0 / fpb, static_cast<long long>(keep));
    }
    printf("done: %d frames -> %s\n", frame_no, outdir.c_str());
    return 0;
}

static void sd_log_to_stderr(enum sd_log_level_t level, const char* text, void* data) {
    (void)data;
    if (level >= SD_LOG_INFO) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

int main(int argc, char** argv) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    sd_set_log_callback(sd_log_to_stderr, nullptr);
    std::string mode = "walk", dit, taehv, scene, latents, outdir = ".";
    std::string actions = "idle:1,W:3", backend = "cpu";
    std::string golden, latents_out;
    int threads = 8, lat_w = 52, lat_h = 30, lat_c = 48, fpb = 3, blocks_n = -1;
    int64_t seed = 42;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next     = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (k == "--mode") mode = next();
        else if (k == "--dit") dit = next();
        else if (k == "--taehv") taehv = next();
        else if (k == "--scene") scene = next();
        else if (k == "--latents") latents = next();
        else if (k == "--actions") actions = next();
        else if (k == "--outdir") outdir = next();
        else if (k == "--threads") threads = std::stoi(next());
        else if (k == "--seed") seed = std::stoll(next());
        else if (k == "--backend") backend = next();
        else if (k == "--lat-w") lat_w = std::stoi(next());
        else if (k == "--lat-h") lat_h = std::stoi(next());
        else if (k == "--lat-c") lat_c = std::stoi(next());
        else if (k == "--fpb") fpb = std::stoi(next());
        else if (k == "--golden") golden = next();
        else if (k == "--latents-out") latents_out = next();
        else if (k == "--blocks") blocks_n = std::stoi(next());
    }
    if (mode == "decode") {
        if (taehv.empty() || latents.empty()) {
            fprintf(stderr, "decode mode needs --taehv and --latents\n");
            return 2;
        }
        return run_decode(taehv, latents, lat_w, lat_h, lat_c, fpb, outdir, threads, backend);
    }
    if (mode == "walkval") {
        if (dit.empty() || taehv.empty() || scene.empty()) {
            fprintf(stderr, "walkval mode needs --dit, --taehv and --scene\n");
            return 2;
        }
        return run_walkval(dit, taehv, scene, golden, actions, latents_out, blocks_n,
                           threads, static_cast<uint64_t>(seed), backend);
    }
    if (dit.empty() || taehv.empty() || scene.empty()) {
        fprintf(stderr, "walk mode needs --dit, --taehv and --scene\n");
        return 2;
    }
    return run_walk(dit, taehv, scene, actions, outdir, threads, seed, backend);
}
