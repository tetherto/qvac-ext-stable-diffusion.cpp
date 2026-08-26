#include <stdio.h>
#include <string>
#include <vector>

#include "stable-diffusion.h"

#include "common/common.h"

struct SDFitCliParams {
    bool verbose   = false;
    bool color     = false;
    bool fit_print = false;

    ArgOptions get_options() {
        ArgOptions options;
        options.bool_options = {
            {"-v", "--verbose", "print extra info", true, &verbose},
            {"", "--color", "colors the logging tags", true, &color},
            {"", "--fit-print", "print the measured memory report to stdout instead of fitted arguments", true, &fit_print},
        };
        return options;
    }
};

static void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    fprintf(stderr, "usage: %s [arguments]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "Measures the memory the model needs for the requested generation parameters using\n");
    fprintf(stderr, "metadata-only dry runs (no weight data is read), then prints the CLI arguments that\n");
    fprintf(stderr, "make it fit into free device memory. Logs go to stderr, arguments to stdout:\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  %s -m model.gguf -W 1024 -H 1024 | tee args.txt\n", argv[0]);
    fprintf(stderr, "  cat args.txt | xargs sd-cli -m model.gguf -p \"a cat\" -W 1024 -H 1024\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments:\n");
    for (const auto& options : options_list) {
        options.print();
    }
}

// keep stdout clean for the fitted arguments
static void fit_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDFitCliParams* params = (SDFitCliParams*)data;
    if (!params->verbose && level == SD_LOG_DEBUG) {
        return;
    }
    const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    fprintf(stderr, "[%-5s] %s", level >= 0 && level <= 3 ? level_str[level] : "?", SAFE_STR(log));
    fflush(stderr);
}

int main(int argc, const char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        printf("%s\n", version_string().c_str());
        return 0;
    }

    SDFitCliParams fit_params;
    SDContextParams ctx_params;
    SDGenerationParams gen_params;

    std::vector<ArgOptions> options_vec = {fit_params.get_options(), ctx_params.get_options(), gen_params.get_options()};
    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        return 1;
    }

    sd_set_log_callback(fit_log_cb, (void*)&fit_params);

    SDMode mode = gen_params.video_frames > 1 ? VID_GEN : IMG_GEN;
    if (!ctx_params.resolve_and_validate(mode) ||
        !gen_params.resolve_and_validate(mode, ctx_params.lora_model_dir, ctx_params.hires_upscalers_dir)) {
        print_usage(argc, argv, options_vec);
        return 1;
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(false);

    sd_fit_workload_t workload;
    sd_fit_workload_init(&workload);
    workload.prompt            = gen_params.prompt.c_str();
    workload.width             = gen_params.get_resolved_width();
    workload.height            = gen_params.get_resolved_height();
    workload.video_frames      = gen_params.video_frames;
    workload.vae_tiling_params = gen_params.vae_tiling_params;

    sd_fit_result_t result;
    enum sd_fit_status_t status = sd_fit_params(&sd_ctx_params, &workload, &result);
    if (status != SD_FIT_SUCCESS) {
        if (fit_params.fit_print && result.report != nullptr) {
            printf("%s", result.report);
        }
        fprintf(stderr, "failed to fit CLI arguments to free memory, exiting...\n");
        sd_fit_result_free(&result);
        return 1;
    }

    if (fit_params.fit_print) {
        printf("%s", SAFE_STR(result.report));
    } else if (result.changed) {
        std::string args;
        if (result.backend != nullptr) {
            args += std::string("--backend \"") + result.backend + "\"";
        }
        if (result.params_backend != nullptr) {
            if (!args.empty()) {
                args += " ";
            }
            args += std::string("--params-backend \"") + result.params_backend + "\"";
        }
        if (result.vae_tiling && !workload.vae_tiling_params.enabled) {
            if (!args.empty()) {
                args += " ";
            }
            args += "--vae-tiling";
        }
        if (result.stream_layers && strlen(SAFE_STR(sd_ctx_params.max_vram)) > 0) {
            if (!args.empty()) {
                args += " ";
            }
            args += std::string("--max-vram \"") + sd_ctx_params.max_vram + "\"";
        }
        if (result.stream_layers && !sd_ctx_params.stream_layers) {
            if (!args.empty()) {
                args += " ";
            }
            args += "--stream-layers";
        }
        fprintf(stderr, "printing fitted CLI arguments to stdout...\n");
        printf("%s\n", args.c_str());
    } else {
        fprintf(stderr, "current parameters already fit into free device memory, no changes needed\n");
        printf("\n");
    }

    sd_fit_result_free(&result);
    return 0;
}
