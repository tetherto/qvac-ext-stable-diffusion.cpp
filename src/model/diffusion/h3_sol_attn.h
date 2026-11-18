#ifndef __SD_MODEL_DIFFUSION_H3_SOL_ATTN_H__
#define __SD_MODEL_DIFFUSION_H3_SOL_ATTN_H__

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "ggml.h"

namespace MiniMaxH3 {

    struct SolAttentionMetadata {
        int32_t magic               = 0x48335343;
        int32_t version             = 3;
        int32_t mode                = 0;
        int32_t video_start         = 0;
        int32_t video_end           = 0;
        int32_t frame_tokens        = 0;
        float beta                  = -1.f;
        int32_t dense_prefix_layers = 4;
        int32_t dense_suffix_layers = 4;
    };

    struct SolAttentionOptions {
        int mode                = 0;
        float beta              = -1.f;
        int warmup_steps        = 40;
        int cooldown_steps      = 2;
        int step_interval       = 1;
        int dense_prefix_layers = 4;
        int dense_suffix_layers = 4;
    };

    inline SolAttentionOptions sol_attention_options_from_environment() {
        SolAttentionOptions options;
        if (const char* mode = std::getenv("SD_H3_SOL_ATTENTION")) {
            if (std::strcmp(mode, "control") == 0 || std::strcmp(mode, "bf16-control") == 0)
                options.mode = 7;
            else if (std::strcmp(mode, "sol") == 0 || std::strcmp(mode, "sol-bf16") == 0)
                options.mode = 6;
        }
        if (const char* beta = std::getenv("SD_H3_SOL_BETA")) {
            char* end         = nullptr;
            errno             = 0;
            const float value = std::strtof(beta, &end);
            if (errno == ERANGE || end == beta || *end || !std::isfinite(value) || value < -10.f || value > 2.f) {
                options.mode = 0;
            } else {
                options.beta = value;
            }
        }
        auto integer_option = [&](const char* name, int& target, int minimum) {
            if (const char* text = std::getenv(name)) {
                char* end        = nullptr;
                errno            = 0;
                const long value = std::strtol(text, &end, 10);
                if (errno == ERANGE || end == text || *end || value < minimum || value > 200)
                    options.mode = 0;
                else
                    target = static_cast<int>(value);
            }
        };
        integer_option("SD_H3_SOL_WARMUP_STEPS", options.warmup_steps, 1);
        integer_option("SD_H3_SOL_COOLDOWN_STEPS", options.cooldown_steps, 1);
        integer_option("SD_H3_SOL_STEP_INTERVAL", options.step_interval, 1);
        integer_option("SD_H3_SOL_DENSE_PREFIX_LAYERS", options.dense_prefix_layers, 0);
        integer_option("SD_H3_SOL_DENSE_SUFFIX_LAYERS", options.dense_suffix_layers, 0);
        return options;
    }

    inline bool sol_attention_step_eligible(int step, int total_steps, const SolAttentionOptions& options) {
        return total_steps >= 10 && step >= options.warmup_steps && step < total_steps - options.cooldown_steps &&
               options.step_interval > 0 && (step - options.warmup_steps) % options.step_interval == 0;
    }

    inline void annotate_sol_attention(ggml_tensor* output, const SolAttentionMetadata* metadata) {
        if (!metadata || metadata->mode == 0)
            return;
        ggml_tensor* attention = output;
        for (int depth = 0; attention && depth < 8; ++depth) {
            if (attention->op != GGML_OP_SCALE && attention->op != GGML_OP_VIEW &&
                attention->op != GGML_OP_CONT && attention->op != GGML_OP_RESHAPE &&
                attention->op != GGML_OP_PERMUTE)
                break;
            attention = attention->src[0];
        }
        if (attention && attention->op == GGML_OP_FLASH_ATTN_EXT) {
            static_assert(sizeof(SolAttentionMetadata) <= GGML_MAX_OP_PARAMS - 4 * sizeof(int32_t), "FA metadata space");
            std::memcpy(attention->op_params + 4, metadata, sizeof(*metadata));
        }
    }

}  // namespace MiniMaxH3
#endif
