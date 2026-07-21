// sd-abot-parity — per-step parity harness for the ABot-World causal walk.
//
// Loads the ABot DiT GGUF + a scene pack + one golden denoise-step input
// (xt latents, per-frame timesteps) dumped by the reference implementation
// (ABot-World/tools/dump_golden_steps.py), runs one causal forward with the
// recompute formulation, and writes the resulting x0 prediction for the block
// as raw float32 for offline comparison against the golden x0.
//
// Usage:
//   sd-abot-parity --dit <gguf> --scene <scene.safetensors> --golden <dir>
//                  --block <b> --step <s> [--threads N] [--out out.bin]
//
// History frames are the golden final latents of blocks < b (clean, t=0),
// actions replayed from the golden walk.json.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "abot_world.hpp"
#include "ggml_extend_backend.h"
#include "model.h"

// ---- minimal .npy reader (float32/float64/int64, C-order) ----
static bool load_npy(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "npy: cannot open %s\n", path.c_str());
        return false;
    }
    char magic[6];
    f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        return false;
    }
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        uint16_t h16;
        f.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);

    bool f4 = header.find("<f4") != std::string::npos;
    bool f8 = header.find("<f8") != std::string::npos;
    bool i8 = header.find("<i8") != std::string::npos;
    if (!f4 && !f8 && !i8) {
        fprintf(stderr, "npy: unsupported dtype in %s (%s)\n", path.c_str(), header.c_str());
        return false;
    }
    auto sp = header.find("'shape': (");
    if (sp == std::string::npos) {
        return false;
    }
    sp += 10;
    shape.clear();
    while (true) {
        while (sp < header.size() && (header[sp] == ' ' || header[sp] == ',')) {
            sp++;
        }
        if (sp >= header.size() || header[sp] == ')') {
            break;
        }
        shape.push_back(strtoll(header.c_str() + sp, nullptr, 10));
        while (sp < header.size() && header[sp] != ',' && header[sp] != ')') {
            sp++;
        }
    }
    int64_t n = 1;
    for (auto s : shape) {
        n *= s;
    }
    out.resize(static_cast<size_t>(n));
    if (f4) {
        f.read(reinterpret_cast<char*>(out.data()), n * 4);
    } else if (f8) {
        std::vector<double> tmp(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(tmp.data()), n * 8);
        for (int64_t i = 0; i < n; i++) {
            out[static_cast<size_t>(i)] = static_cast<float>(tmp[static_cast<size_t>(i)]);
        }
    } else {
        std::vector<int64_t> tmp(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(tmp.data()), n * 8);
        for (int64_t i = 0; i < n; i++) {
            out[static_cast<size_t>(i)] = static_cast<float>(tmp[static_cast<size_t>(i)]);
        }
    }
    return f.good();
}

int main(int argc, char** argv) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    std::string dit, scene_path, golden, out_path = "abot_parity_x0.bin";
    int block = 0, step = 0, n_threads = 8;
    std::string backend_spec = "cpu";
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next     = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (k == "--dit") dit = next();
        else if (k == "--scene") scene_path = next();
        else if (k == "--golden") golden = next();
        else if (k == "--block") block = std::stoi(next());
        else if (k == "--step") step = std::stoi(next());
        else if (k == "--threads") n_threads = std::stoi(next());
        else if (k == "--backend") backend_spec = next();
        else if (k == "--out") out_path = next();
    }
    if (dit.empty() || scene_path.empty() || golden.empty()) {
        fprintf(stderr, "usage: sd-abot-parity --dit x.gguf --scene scene.safetensors --golden dir --block b --step s\n");
        return 2;
    }

    // ---- backend + weights ----
    SDBackendManager backend_manager;
    std::string err;
    if (!backend_manager.init(backend_spec.c_str(), nullptr, false, false, false, false, &err)) {
        fprintf(stderr, "backend init failed: %s\n", err.c_str());
        return 1;
    }
    ggml_backend_t be  = backend_manager.runtime_backend(SDBackendModule::DIFFUSION);
    ggml_backend_t pbe = backend_manager.params_backend(SDBackendModule::DIFFUSION);

    ModelLoader ml;
    if (!ml.init_from_file(dit, "model.diffusion_model.")) {
        fprintf(stderr, "cannot open DiT: %s\n", dit.c_str());
        return 1;
    }

    ABOT::AbotWorldConfig cfg;  // window 8 / 3 fpb — matches the golden test config
    ABOT::AbotWorldRunner runner(be, pbe, ml.get_tensor_storage_map(), "model.diffusion_model.", cfg);
    if (!runner.alloc_params_buffer()) {
        fprintf(stderr, "alloc params failed\n");
        return 1;
    }
    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors, "model.diffusion_model");
    if (!ml.load_tensors(tensors, {}, n_threads)) {
        fprintf(stderr, "load DiT tensors failed\n");
        return 1;
    }
    printf("DiT loaded (%zu tensors)\n", tensors.size());

    if (!runner.scene.load(scene_path)) {
        return 1;
    }
    printf("scene pack loaded (ref slots: %d)\n", runner.scene.ref_slots);

    // ---- golden inputs ----
    std::vector<float> xt;
    std::vector<int64_t> xt_shape;  // [1, F, C, H, W]
    char name[256];
    snprintf(name, sizeof(name), "%s/b%d_s%d_xt.npy", golden.c_str(), block, step);
    if (!load_npy(name, xt, xt_shape)) {
        return 1;
    }
    std::vector<float> tsv;
    std::vector<int64_t> ts_shape;
    snprintf(name, sizeof(name), "%s/b%d_s%d_t.npy", golden.c_str(), block, step);
    if (!load_npy(name, tsv, ts_shape)) {
        return 1;
    }
    const int Fb = static_cast<int>(xt_shape[1]);
    const int C  = static_cast<int>(xt_shape[2]);
    const int H  = static_cast<int>(xt_shape[3]);
    const int W  = static_cast<int>(xt_shape[4]);
    runner.lat_w = W;
    runner.lat_h = H;
    runner.lat_c = C;

    // walk.json actions: rows of 8 ints (one row per block, W..L bit order).
    // Tiny hand parser: after "actions", every inner [...] is one row until the
    // outer array closes.
    std::vector<uint8_t> block_actions;
    {
        std::ifstream jf(golden + "/walk.json");
        std::string js((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
        size_t p     = js.find("\"actions\"");
        size_t outer = js.find('[', p);
        size_t pos   = outer + 1;
        while (true) {
            size_t rs = js.find_first_of("[]", pos);
            if (rs == std::string::npos || js[rs] == ']') {
                break;  // outer array closed
            }
            size_t re = js.find(']', rs);
            if (re == std::string::npos) {
                break;
            }
            uint8_t m = 0;
            int bit   = 0;
            for (size_t q = rs + 1; q < re && bit < 8; q++) {
                if (js[q] == '0' || js[q] == '1') {
                    m |= static_cast<uint8_t>((js[q] - '0') << bit);
                    bit++;
                }
            }
            block_actions.push_back(m);
            pos = re + 1;
        }
    }
    printf("parsed %zu action rows\n", block_actions.size());

    // history latents: golden final latents of blocks < block
    std::vector<std::vector<float>> hist_store;
    std::vector<const float*> frames;
    std::vector<uint8_t> actions;
    std::vector<float> tframes;
    const size_t frame_elems = static_cast<size_t>(C) * H * W;
    for (int b = 0; b < block; b++) {
        std::vector<float> lat;
        std::vector<int64_t> lshape;
        snprintf(name, sizeof(name), "%s/b%d_final_latent.npy", golden.c_str(), b);
        if (!load_npy(name, lat, lshape)) {
            return 1;
        }
        for (int f = 0; f < Fb; f++) {
            hist_store.emplace_back(lat.begin() + static_cast<long long>(f) * frame_elems,
                                    lat.begin() + static_cast<long long>(f + 1) * frame_elems);
            actions.push_back(b < static_cast<int>(block_actions.size()) ? block_actions[static_cast<size_t>(b)] : 0);
            tframes.push_back(0.0f);  // history at context_noise = 0
        }
    }
    for (auto& h : hist_store) {
        frames.push_back(h.data());
    }
    // current block xt frames
    for (int f = 0; f < Fb; f++) {
        frames.push_back(xt.data() + static_cast<size_t>(f) * frame_elems);
        actions.push_back(block < static_cast<int>(block_actions.size()) ? block_actions[static_cast<size_t>(block)] : 0);
        tframes.push_back(tsv[static_cast<size_t>(f)]);
    }

    printf("forward: F_vis=%zu block_frames=%d t=[", frames.size(), Fb);
    for (auto t : tframes) {
        printf("%.1f ", t);
    }
    printf("]\n");

    auto flow = runner.forward_step(frames, actions, tframes, Fb, n_threads);
    if (flow.empty()) {
        fprintf(stderr, "forward failed\n");
        return 1;
    }
    // unpatchify flow tokens {pt*ph*pw*C, tokens} -> [F, C, H, W] and x0 = xt - sigma*flow
    const int h_len = H / 2, w_len = W / 2;
    std::vector<float> x0(static_cast<size_t>(Fb) * frame_elems);
    for (int f = 0; f < Fb; f++) {
        float sigma = tframes[frames.size() - Fb + f] / 1000.0f;
        for (int hh = 0; hh < h_len; hh++) {
            for (int ww = 0; ww < w_len; ww++) {
                const float* tok = flow.data() + ((static_cast<size_t>(f) * h_len + hh) * w_len + ww) * (4 * C);
                for (int ph = 0; ph < 2; ph++) {
                    for (int pw = 0; pw < 2; pw++) {
                        for (int c = 0; c < C; c++) {
                            // token layout: [c fastest, then pw, then ph] (pt*ph*pw*C with C innermost)
                            float fl   = tok[(ph * 2 + pw) * C + c];
                            size_t idx = ((static_cast<size_t>(f) * C + c) * H + (hh * 2 + ph)) * W + (ww * 2 + pw);
                            x0[idx]    = xt[idx] - sigma * fl;
                        }
                    }
                }
            }
        }
    }
    // NOTE: x0 index math above intentionally kept explicit; xt is [F,C,H,W].

    std::ofstream of(out_path, std::ios::binary);
    of.write(reinterpret_cast<const char*>(x0.data()), static_cast<std::streamsize>(x0.size() * sizeof(float)));
    printf("wrote %s (%zu floats)\n", out_path.c_str(), x0.size());
    return 0;
}
