// SDXS benchmark: loads the model once, then generates 1 warm-up and 3 timed 512x512 images
// (1 step, CFG 1, seed 42). Stage times come from sd.cpp's own log lines.
// usage: sd-bench bench <model.gguf> <unet.mlmodelc|-> <out_prefix> [tae_decoder.mlmodelc]
//   "-" runs the UNet with ggml (Metal); a .mlmodelc runs it with the CoreML sidecar. The optional
//   last argument runs the tiny decoder with its CoreML sidecar too.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "stable-diffusion.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

struct StageTimes {
    double text_ms = 0, unet_ms = 0, decode_ms = 0;
};
StageTimes g_stage;

double after(const char* text, const char* key) {
    const char* p = strstr(text, key);
    return p ? atof(p + strlen(key)) : -1;
}

void log_cb(enum sd_log_level_t level, const char* text, void*) {
    double v;
    if ((v = after(text, "get_learned_condition completed, taking ")) >= 0) {
        g_stage.text_ms = v;  // ms
    } else if ((v = after(text, "sampling completed, taking ")) >= 0) {
        g_stage.unet_ms = v * 1000;  // s
    } else if ((v = after(text, "decode_first_stage completed, taking ")) >= 0) {
        g_stage.decode_ms = v * 1000;  // s
    }
    if (level >= SD_LOG_WARN || strstr(text, "coreml unet") || strstr(text, "coreml tae") || strstr(text, "total params memory")) {
        fputs(text, stdout);
    }
}

}  // namespace

int sd_bench_main(int argc, const char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s bench <model.gguf> <unet.mlmodelc|-> <out_prefix> [tae_decoder.mlmodelc]\n", argv[0]);
        return 1;
    }
    const char* model  = argv[2];
    const char* coreml = strcmp(argv[3], "-") == 0 ? "" : argv[3];
    std::string prefix = argv[4];
    const char* tae    = argc > 5 ? argv[5] : "";

    struct Job {
        const char* name;
        const char* prompt;
    };
    const std::vector<Job> jobs = {
        {"warmup", "a red apple on a wooden table"},
        {"cat", "portrait of a lovely cat"},
        {"cabin", "a cozy mountain cabin at sunset, snow, warm window lights, photorealistic"},
        {"billboard", "a large billboard on a city street showing an advertisement with the text \"COFFEE\", photo, daytime"},
    };

    sd_set_log_callback(log_cb, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path              = model;
    ctx_params.coreml_unet_path        = coreml;
    ctx_params.coreml_tae_decoder_path = tae;
    ctx_params.free_params_immediately = false;  // reuse the context for every image

    auto t0       = std::chrono::steady_clock::now();
    sd_ctx_t* sd  = new_sd_ctx(&ctx_params);
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sd == nullptr) {
        fprintf(stdout, "[bench] new_sd_ctx failed\n");
        return 1;
    }
    fprintf(stdout, "[bench] unet=%s decoder=%s load=%.2f s\n", *coreml ? "coreml" : "metal", *tae ? "coreml" : "metal", load_s);

    for (const auto& job : jobs) {
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt                         = job.prompt;
        params.width                          = 512;
        params.height                         = 512;
        params.seed                           = 42;
        params.batch_count                    = 1;
        params.sample_params.sample_steps     = 1;
        params.sample_params.guidance.txt_cfg = 1.0f;
        params.sample_params.sample_method    = sd_get_default_sample_method(sd);
        params.sample_params.scheduler        = sd_get_default_scheduler(sd, params.sample_params.sample_method);

        g_stage         = {};
        t0              = std::chrono::steady_clock::now();
        sd_image_t* img = generate_image(sd, &params);
        double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (img == nullptr || img->data == nullptr) {
            fprintf(stdout, "[bench] %s failed\n", job.name);
            return 1;
        }
        std::string path = prefix + "_" + job.name + ".png";
        stbi_write_png(path.c_str(), img->width, img->height, img->channel, img->data, 0);
        free(img->data);
        free(img);
        fprintf(stdout, "[bench] %s,text_ms=%.0f,unet_ms=%.0f,decode_ms=%.0f,total_ms=%.0f\n",
                job.name, g_stage.text_ms, g_stage.unet_ms, g_stage.decode_ms, total_ms);
    }
    free_sd_ctx(sd);
    return 0;
}

#ifndef SDBENCH_NO_MAIN
int main(int argc, const char* argv[]) {
    return sd_bench_main(argc, argv);
}
#endif
