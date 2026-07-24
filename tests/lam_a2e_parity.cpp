#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "lam_audio2expression.hpp"

static bool read_f32_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const auto bytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (bytes % sizeof(float) != 0) {
        return false;
    }
    out.resize(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(in);
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    float m = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: lam-a2e-parity <model.gguf> <fixture_dir> <case_prefix>\n");
        return 2;
    }

    const std::string model = argv[1];
    const std::string dir = argv[2];
    const std::string prefix = argv[3];
    const float tol = 1e-3F;

    LamAudio2Expression model_impl;
    if (!model_impl.load(model)) {
        std::fprintf(stderr, "load failed: %s\n", model_impl.lastError().c_str());
        return 1;
    }

    std::vector<float> pcm;
    std::vector<float> ref_expr;
    if (!read_f32_bin(dir + "/" + prefix + "_input_pcm.bin", pcm) ||
        !read_f32_bin(dir + "/" + prefix + "_expr.bin", ref_expr)) {
        std::fprintf(stderr, "failed to read fixture bins under %s\n", dir.c_str());
        return 1;
    }

    std::vector<float> out;
    std::map<std::string, std::vector<float>> taps;
    if (!model_impl.run(pcm, /*idIdx*/ 0, out, &taps)) {
        std::fprintf(stderr, "run failed: %s\n", model_impl.lastError().c_str());
        return 1;
    }

    // Critical path stages with matching layouts in the lipsync-ggml fixtures.
    const char* gated[] = {
        "fe_out", "fp_out", "pos_conv_out", "enc_pre_ln",
        "enc_layer_0", "enc_layer_11", "expr"
    };

    bool ok = true;
    for (const char* stage : gated) {
        std::vector<float> ref;
        const std::string ref_path = dir + "/" + prefix + "_" + stage + ".bin";
        if (!read_f32_bin(ref_path, ref)) {
            std::printf("skip missing stage %s\n", stage);
            continue;
        }
        const std::vector<float>* got = (std::string(stage) == "expr") ? &out : nullptr;
        std::vector<float> local;
        if (got == nullptr) {
            auto it = taps.find(stage);
            if (it == taps.end()) {
                std::printf("missing tap %s\n", stage);
                ok = false;
                continue;
            }
            local = it->second;
            got = &local;
        }
        const float err = max_abs_diff(*got, ref);
        std::printf("%s max_abs_diff=%.6g n=%zu\n", stage, err, got->size());
        if (!(err <= tol)) {
            ok = false;
        }
    }

    if (!ok) {
        std::fprintf(stderr, "parity failed\n");
        return 1;
    }

    std::printf("LAM-A2E parity passed for case %s\n", prefix.c_str());
    return 0;
}
