#include <cstdio>
#include <stdexcept>
#include "model/diffusion/h3_sol_attn.h"

static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

static void environment(const char* name, const char* value = nullptr) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

int main() {
    try {
        using namespace MiniMaxH3;
        for (const char* name : {"SD_H3_SOL_ATTENTION", "SD_H3_SOL_BETA", "SD_H3_SOL_WARMUP_STEPS",
                                 "SD_H3_SOL_COOLDOWN_STEPS", "SD_H3_SOL_STEP_INTERVAL", "SD_H3_SOL_DENSE_PREFIX_LAYERS", "SD_H3_SOL_DENSE_SUFFIX_LAYERS"})
            environment(name);
        require(sol_attention_options_from_environment().mode == 0, "Sparse attention must be default off");
        environment("SD_H3_SOL_ATTENTION", "unknown");
        require(sol_attention_options_from_environment().mode == 0, "Unknown policy enabled sparse attention");
        environment("SD_H3_SOL_ATTENTION", "control");
        require(sol_attention_options_from_environment().mode == 7, "Missing all-keep control");
        environment("SD_H3_SOL_ATTENTION", "sol-exact-control");
        require(sol_attention_options_from_environment().mode == 0, "Removed diagnostic mode enabled approximation");
        environment("SD_H3_SOL_ATTENTION", "sol-bf16");
        require(sol_attention_options_from_environment().mode == 6, "Missing BF16 proxy correction");
        environment("SD_H3_SOL_ATTENTION", "bf16-control");
        require(sol_attention_options_from_environment().mode == 7, "Missing BF16 all-keep control");
        environment("SD_H3_SOL_ATTENTION", "sol");
        require(sol_attention_options_from_environment().mode == 6, "Missing diagonal selector");
        for (const char* text : {"", "nan", "inf", "-11", "2.1", "1junk", "1e9999"}) {
            environment("SD_H3_SOL_BETA", text);
            require(sol_attention_options_from_environment().mode == 0, "Invalid threshold enabled approximation");
        }
        environment("SD_H3_SOL_BETA", "1");
        for (const char* name : {"SD_H3_SOL_WARMUP_STEPS", "SD_H3_SOL_COOLDOWN_STEPS", "SD_H3_SOL_STEP_INTERVAL"}) {
            for (const char* text : {"", "0", "201", "bad", "999999999999999999999999"}) {
                environment(name, text);
                require(sol_attention_options_from_environment().mode == 0, "Invalid step protection accepted");
            }
            environment(name);
        }
        for (const char* name : {"SD_H3_SOL_DENSE_PREFIX_LAYERS", "SD_H3_SOL_DENSE_SUFFIX_LAYERS"}) {
            for (const char* text : {"", "-1", "201", "bad"}) {
                environment(name, text);
                require(sol_attention_options_from_environment().mode == 0, "Invalid layer protection accepted");
            }
            environment(name);
        }
        environment("SD_H3_SOL_WARMUP_STEPS", "37");
        const auto options = sol_attention_options_from_environment();
        require(options.mode == 6 && options.beta == 1.f, "Valid configuration disabled");
        for (int step = -1; step <= 50; ++step) {
            require(sol_attention_step_eligible(step, 50, options) == (step >= 37 && step < 48), "Dense windows are incorrect");
            require(!sol_attention_step_eligible(step, 0, options), "Unknown scheduler accepted");
        }
        environment("SD_H3_SOL_WARMUP_STEPS", "10");
        environment("SD_H3_SOL_STEP_INTERVAL", "2");
        auto alternating = sol_attention_options_from_environment();
        require(alternating.mode == 6, "Alternating schedule disabled");
        const int sparse_steps[] = {10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46};
        for (int step = -1; step <= 50; ++step) {
            bool expected = false;
            for (int sparse_step : sparse_steps)
                expected = expected || step == sparse_step;
            require(sol_attention_step_eligible(step, 50, alternating) == expected, "Alternating schedule or dense boundary changed");
        }
        alternating.warmup_steps = 11;
        require(sol_attention_step_eligible(11, 50, alternating), "Interval must start at warmup boundary");
        require(!sol_attention_step_eligible(12, 50, alternating), "Intervening step must be dense");
        require(sol_attention_step_eligible(47, 50, alternating), "Odd warmup lost final eligible step");
        require(!sol_attention_step_eligible(48, 50, alternating), "Interval bypassed cooldown");
        require(!sol_attention_step_eligible(11, 12, alternating), "Overlapping dense windows enabled sparsity");
        alternating.step_interval = 0;
        require(!sol_attention_step_eligible(11, 50, alternating), "Zero interval accepted");
        auto ctx = ggml_init({32 * ggml_tensor_overhead(), nullptr, true});
        require(ctx != nullptr, "Context allocation failed");
        auto q         = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 1025, 56);
        auto k         = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 1025, 56);
        auto v         = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 1025, 56);
        auto attention = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 2.f, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
        int32_t original[4];
        std::memcpy(original, attention->op_params, sizeof(original));
        auto output = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_scale(ctx, attention, 0.5f)), 128 * 56, 1025);
        SolAttentionMetadata metadata;
        metadata.mode         = 6;
        metadata.video_start  = 65;
        metadata.video_end    = 961;
        metadata.frame_tokens = 128;
        annotate_sol_attention(output, &metadata);
        SolAttentionMetadata actual;
        std::memcpy(&actual, attention->op_params + 4, sizeof(actual));
        require(std::memcmp(original, attention->op_params, sizeof(original)) == 0, "Standard attention parameters changed");
        require(std::memcmp(&metadata, &actual, sizeof(actual)) == 0, "Attention annotation lost across wrappers");
        require(actual.version == 3 && actual.dense_prefix_layers == 4 && actual.dense_suffix_layers == 4, "Engine/backend metadata version changed");
        annotate_sol_attention(output, nullptr);
        ggml_free(ctx);
        std::puts("PASS default off, malformed options, step/layer guards, scheduler bounds and versioned attention metadata");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
