#include "lam-a2e.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "lam_audio2expression.hpp"

struct lam_a2e_context {
    std::string last_error;
    std::unique_ptr<LamAudio2Expression> model;
    int32_t identity_index = 0;
    int32_t n_threads = 0;
    bool use_gpu = false;
    bool model_loaded = false;
};

lam_a2e_context * lam_a2e_create(const lam_a2e_params * params) {
    if (params == nullptr || params->model_path == nullptr ||
        std::strlen(params->model_path) == 0) {
        return nullptr;
    }

    auto * ctx = new lam_a2e_context();
    ctx->identity_index = params->identity_index;
    ctx->n_threads = params->n_threads;
    ctx->use_gpu = params->use_gpu;

    if (params->use_gpu) {
        ctx->last_error =
            "LAM-A2E GPU backends are not enabled until CPU parity is complete.";
        return ctx;
    }

    ctx->model = std::make_unique<LamAudio2Expression>();
    if (!ctx->model->load(params->model_path, nullptr, params->n_threads)) {
        ctx->last_error = ctx->model->lastError();
        ctx->model.reset();
        return ctx;
    }

    ctx->model_loaded = true;
    ctx->last_error.clear();
    return ctx;
}

void lam_a2e_free(lam_a2e_context * ctx) {
    delete ctx;
}

lam_a2e_status lam_a2e_process_pcm_f32(
    lam_a2e_context * ctx,
    const float * pcm,
    int64_t pcm_sample_count,
    int32_t sample_rate,
    lam_a2e_frame ** frames,
    int32_t * frame_count) {
    if (ctx == nullptr || pcm == nullptr || pcm_sample_count <= 0 ||
        frames == nullptr || frame_count == nullptr) {
        return LAM_A2E_STATUS_INVALID_ARGUMENT;
    }

    *frames = nullptr;
    *frame_count = 0;

    if (!ctx->model_loaded || ctx->model == nullptr) {
        if (ctx->last_error.empty()) {
            ctx->last_error = "LAM-A2E model is not loaded.";
        }
        return LAM_A2E_STATUS_MODEL_LOAD_FAILED;
    }
    if (sample_rate != 16000) {
        ctx->last_error = "LAM-A2E requires 16 kHz mono PCM input.";
        return LAM_A2E_STATUS_INVALID_ARGUMENT;
    }

    std::vector<float> pcm_vec(pcm, pcm + pcm_sample_count);
    std::vector<float> coeffs;
    if (!ctx->model->run(pcm_vec, static_cast<uint32_t>(ctx->identity_index), coeffs)) {
        ctx->last_error = ctx->model->lastError();
        return LAM_A2E_STATUS_INVALID_ARGUMENT;
    }

    const auto& hp = ctx->model->hparams();
    const int64_t n_frames = ctx->model->frameCount(pcm_sample_count);
    if (n_frames <= 0 ||
        coeffs.size() != static_cast<size_t>(n_frames) * hp.nCoeffs) {
        ctx->last_error = "LAM-A2E produced an unexpected coefficient buffer size.";
        return LAM_A2E_STATUS_INVALID_ARGUMENT;
    }

    auto * out = static_cast<lam_a2e_frame *>(
        std::malloc(sizeof(lam_a2e_frame) * static_cast<size_t>(n_frames)));
    if (out == nullptr) {
        ctx->last_error = "Failed to allocate LAM-A2E frame buffer.";
        return LAM_A2E_STATUS_INVALID_ARGUMENT;
    }

    for (int64_t i = 0; i < n_frames; ++i) {
        out[i].timestamp_us =
            (i * 1000000LL) / static_cast<int64_t>(hp.fps > 0 ? hp.fps : 30);
        std::memcpy(out[i].arkit_52, coeffs.data() + i * hp.nCoeffs,
                    sizeof(float) * hp.nCoeffs);
    }

    *frames = out;
    *frame_count = static_cast<int32_t>(n_frames);
    ctx->last_error.clear();
    return LAM_A2E_STATUS_OK;
}

void lam_a2e_free_frames(lam_a2e_frame * frames) {
    std::free(frames);
}

const char * lam_a2e_get_last_error(const lam_a2e_context * ctx) {
    return ctx == nullptr ? "LAM-A2E context is null." : ctx->last_error.c_str();
}
