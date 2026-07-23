// sd-abot-walk — full causal walk through a fixed ABot-World scene.
//
// Runs the reference block loop natively: for each block of
// num_frame_per_block latent frames, 4 warped flow-matching denoise steps
// (x0 = xt - sigma*flow; renoise to the next step), history kept clean at
// t=0, actions injected per block. Two noise modes:
//   --golden <dir>   replay the reference's recorded noise (deterministic
//                    full-chain validation; final latents must match the
//                    golden b{b}_final_latent.npy)
//   (default)        xorshift/Box-Muller RNG with --seed (free walk)
//
// Output: raw f32 latents [T,C,H,W] (torch order) to --out; decode with
// taehv (native, once the taew GGUF is present) or the reference tools.
//
// Usage:
//   sd-abot-walk --dit dit.gguf --scene scene.safetensors
//                [--golden golden_steps_dir] [--actions idle:1,W:3]
//                [--blocks N] [--threads N] [--out walk_latents.bin]

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "abot_world.hpp"
#include "ggml_extend_backend.h"
#include "model.h"

static bool load_npy_f32(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape);

// xorshift128+ Box-Muller normal RNG (portable, seed-stable across platforms)
struct Rng {
    uint64_t s0, s1;
    bool have_spare = false;
    float spare     = 0.0f;
    explicit Rng(uint64_t seed) {
        s0 = seed ^ 0x9E3779B97F4A7C15ULL;
        s1 = (seed << 1) | 1;
        for (int i = 0; i < 16; i++) {
            next_u64();
        }
    }
    uint64_t next_u64() {
        uint64_t x = s0, y = s1;
        s0         = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }
    float uniform() {
        return static_cast<float>((next_u64() >> 11) * (1.0 / 9007199254740992.0));
    }
    float normal() {
        if (have_spare) {
            have_spare = false;
            return spare;
        }
        float u1 = std::max(uniform(), 1e-12f), u2 = uniform();
        float r = std::sqrt(-2.0f * std::log(u1)), a = 6.28318530718f * u2;
        spare      = r * std::sin(a);
        have_spare = true;
        return r * std::cos(a);
    }
};

static std::vector<uint8_t> parse_actions(const std::string& spec) {
    // "idle:1,W:3,WJ:2" -> per-block key bitmask (W,A,S,D,I,J,K,L = bits 0..7).
    // "idle" / "none" mean no keys held; otherwise every character of the
    // chunk must be a walk key (so a typo cannot silently become phantom keys).
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

int main(int argc, char** argv) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    std::string dit, scene_path, golden, out_path = "walk_latents.bin", actions_spec = "idle:1,W:3";
    int n_threads = 8, blocks_n = -1;
    uint64_t seed            = 42;
    std::string backend_spec = "cpu";
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next     = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (k == "--dit") dit = next();
        else if (k == "--scene") scene_path = next();
        else if (k == "--golden") golden = next();
        else if (k == "--actions") actions_spec = next();
        else if (k == "--blocks") blocks_n = std::stoi(next());
        else if (k == "--threads") n_threads = std::stoi(next());
        else if (k == "--seed") seed = std::stoull(next());
        else if (k == "--backend") backend_spec = next();
        else if (k == "--out") out_path = next();
    }
    if (dit.empty() || scene_path.empty()) {
        fprintf(stderr, "usage: sd-abot-walk --dit x.gguf --scene scene.safetensors [--golden dir] [--actions a] [--out o.bin]\n");
        return 2;
    }

    SDBackendManager backend_manager;
    std::string err;
    if (!backend_manager.init(backend_spec.c_str(), nullptr, false, false, false, false, &err)) {
        fprintf(stderr, "backend init failed: %s\n", err.c_str());
        return 1;
    }
    ModelLoader ml;
    if (!ml.init_from_file(dit, "model.diffusion_model.")) {
        return 1;
    }
    ABOT::AbotWorldConfig cfg;
    ABOT::AbotWorldRunner runner(backend_manager.runtime_backend(SDBackendModule::DIFFUSION),
                                 backend_manager.params_backend(SDBackendModule::DIFFUSION),
                                 ml.get_tensor_storage_map(), "model.diffusion_model.", cfg);
    if (const char* fa = std::getenv("ABOT_FLASH_ATTN"); fa != nullptr && fa[0] == '1') {
        runner.set_flash_attention_enabled(true);
        printf("flash attention enabled\n");
    }
    if (!runner.alloc_params_buffer()) {
        return 1;
    }
    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors, "model.diffusion_model");
    if (!ml.load_tensors(tensors, {}, n_threads)) {
        return 1;
    }
    if (!runner.scene.load(scene_path)) {
        return 1;
    }
    auto ffl     = runner.scene.first_frame_latents.shape();
    runner.lat_w = ffl[0];
    runner.lat_h = ffl[1];
    runner.lat_c = ffl[2];
    const int W = static_cast<int>(runner.lat_w), H = static_cast<int>(runner.lat_h), C = static_cast<int>(runner.lat_c);
    const size_t fel = static_cast<size_t>(C) * H * W;
    const int Fb     = cfg.num_frame_per_block;
    printf("scene: latent %dx%dx%d, ref slots %d\n", W, H, C, runner.scene.ref_slots);

    std::vector<uint8_t> acts = parse_actions(actions_spec);
    if (blocks_n > 0) {
        acts.resize(static_cast<size_t>(blocks_n), acts.empty() ? 0 : acts.back());
    }
    Rng rng(seed);

    // walk state (torch-order [C,H,W] frames)
    std::vector<std::vector<float>> hist;
    std::vector<uint8_t> hist_act;

    // first_frame latent in torch order: scene tensor is {W,H,C} ggml == torch [C,H,W] flat ✓
    std::vector<float> ff(runner.scene.first_frame_latents.data(),
                          runner.scene.first_frame_latents.data() + fel);

    for (size_t b = 0; b < acts.size(); b++) {
        const bool first = b == 0;
        // ---- init xt for the block ----
        std::vector<float> xt(static_cast<size_t>(Fb) * fel);
        if (!golden.empty()) {
            std::vector<float> g;
            std::vector<int64_t> gs;
            char nm[256];
            snprintf(nm, sizeof(nm), "%s/b%zu_s0_xt.npy", golden.c_str(), b);
            if (!load_npy_f32(nm, g, gs)) {
                return 1;
            }
            memcpy(xt.data(), g.data(), xt.size() * sizeof(float));
        } else {
            for (auto& v : xt) {
                v = rng.normal();
            }
            if (first) {
                memcpy(xt.data(), ff.data(), fel * sizeof(float));  // pin frame 0
            }
        }

        // ---- 4-step denoise ----
        for (size_t s = 0; s < cfg.denoise_steps.size(); s++) {
            float t_cur = cfg.denoise_steps[s];
            // assemble visible frames: history + block frames
            std::vector<const float*> frames;
            std::vector<uint8_t> facts;
            std::vector<float> fts;
            for (size_t h = 0; h < hist.size(); h++) {
                frames.push_back(hist[h].data());
                facts.push_back(hist_act[h]);
                fts.push_back(cfg.context_noise_t);
            }
            for (int f = 0; f < Fb; f++) {
                frames.push_back(xt.data() + static_cast<size_t>(f) * fel);
                facts.push_back(acts[b]);
                fts.push_back(first && f == 0 ? 0.0f : t_cur);
            }

            std::vector<int64_t> abs_ids(frames.size());
            for (size_t i = 0; i < abs_ids.size(); i++) {
                abs_ids[i] = static_cast<int64_t>(i);  // full contiguous history
            }
            auto flow = runner.forward_step(frames, facts, fts, abs_ids, Fb, n_threads);
            if (flow.empty()) {
                fprintf(stderr, "forward failed (block %zu step %zu)\n", b, s);
                return 1;
            }
            // unpatchify + x0 + renoise (host)
            const int h_len = H / 2, w_len = W / 2;
            std::vector<float> x0(static_cast<size_t>(Fb) * fel);
            for (int f = 0; f < Fb; f++) {
                float sigma = fts[hist.size() + f] / 1000.0f;
                const float* xf = xt.data() + static_cast<size_t>(f) * fel;
                float* of       = x0.data() + static_cast<size_t>(f) * fel;
                for (int hh = 0; hh < h_len; hh++) {
                    for (int ww = 0; ww < w_len; ww++) {
                        const float* tok = flow.data() + ((static_cast<size_t>(f) * h_len + hh) * w_len + ww) * (4 * C);
                        for (int ph = 0; ph < 2; ph++) {
                            for (int pw = 0; pw < 2; pw++) {
                                for (int c = 0; c < C; c++) {
                                    size_t idx = (static_cast<size_t>(c) * H + (hh * 2 + ph)) * W + (ww * 2 + pw);
                                    of[idx]    = xf[idx] - sigma * tok[(ph * 2 + pw) * C + c];
                                }
                            }
                        }
                    }
                }
            }
            if (s + 1 < cfg.denoise_steps.size()) {
                float t_next = cfg.denoise_steps[s + 1], sn = t_next / 1000.0f;
                std::vector<float> eps(static_cast<size_t>(Fb) * fel);
                if (!golden.empty()) {
                    std::vector<float> g;
                    std::vector<int64_t> gs;
                    char nm[256];
                    snprintf(nm, sizeof(nm), "%s/b%zu_s%zu_eps.npy", golden.c_str(), b, s);
                    if (!load_npy_f32(nm, g, gs)) {
                        return 1;
                    }
                    memcpy(eps.data(), g.data(), eps.size() * sizeof(float));
                } else {
                    for (auto& v : eps) {
                        v = rng.normal();
                    }
                }
                for (size_t i = 0; i < xt.size(); i++) {
                    xt[i] = (1.0f - sn) * x0[i] + sn * eps[i];
                }
            } else {
                xt = x0;
            }
            if (first) {
                memcpy(xt.data(), ff.data(), fel * sizeof(float));  // keep frame 0 pinned
            }
            printf("block %zu step %zu done (t=%.1f)\n", b, s, t_cur);
        }

        for (int f = 0; f < Fb; f++) {
            hist.emplace_back(xt.begin() + static_cast<long long>(f) * fel,
                              xt.begin() + static_cast<long long>(f + 1) * fel);
            hist_act.push_back(acts[b]);
        }
        printf("block %zu finalized (history %zu frames)\n", b, hist.size());
    }

    std::ofstream of(out_path, std::ios::binary);
    for (auto& fr : hist) {
        of.write(reinterpret_cast<const char*>(fr.data()), static_cast<std::streamsize>(fel * sizeof(float)));
    }
    printf("wrote %s (%zu frames x %zu floats)\n", out_path.c_str(), hist.size(), fel);
    return 0;
}

// ---- minimal .npy reader (f32/f64/i64, C-order) ----
static bool load_npy_f32(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
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
    if (!f4 && !f8) {
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
    } else {
        std::vector<double> tmp(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(tmp.data()), n * 8);
        for (int64_t i = 0; i < n; i++) {
            out[static_cast<size_t>(i)] = static_cast<float>(tmp[static_cast<size_t>(i)]);
        }
    }
    return f.good();
}
