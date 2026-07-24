#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "lam-a2e.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: lam-a2e-smoke <model.gguf>\n");
        return 2;
    }

    lam_a2e_params params{};
    params.model_path     = argv[1];
    params.identity_index = 0;
    params.n_threads      = 4;
    params.use_gpu        = false;

    lam_a2e_context* ctx = lam_a2e_create(&params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create LAM-A2E context\n");
        return 1;
    }

    std::array<float, 16000> pcm{};
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = 0.05F * std::sin(2.0F * 3.14159265F * 220.0F *
                                  static_cast<float>(i) / 16000.0F);
    }

    lam_a2e_frame* frames = nullptr;
    int32_t frame_count   = 0;
    const auto status = lam_a2e_process_pcm_f32(
        ctx, pcm.data(), static_cast<int64_t>(pcm.size()), 16000, &frames, &frame_count);

    if (status != LAM_A2E_STATUS_OK || frames == nullptr || frame_count < 2) {
        std::fprintf(stderr, "LAM-A2E smoke failed (%d): %s\n",
                     static_cast<int>(status), lam_a2e_get_last_error(ctx));
        lam_a2e_free(ctx);
        return 1;
    }

    std::printf("LAM-A2E smoke passed: frames=%d ts0=%lld coeff0=%.6f\n",
                frame_count,
                static_cast<long long>(frames[0].timestamp_us),
                frames[0].arkit_52[0]);
    lam_a2e_free_frames(frames);
    lam_a2e_free(ctx);
    return 0;
}
