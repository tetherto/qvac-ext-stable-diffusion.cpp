// Metal vs CoreML benchmark: loads the model once, then generates 1 warm-up and up to 3 timed images
// (512x512 unless --size is given, seed 42). Stage times come from sd.cpp's own log lines. Defaults fit SDXS (1 step, CFG 1).
// usage: sd-bench bench <model.gguf|-> <diffusion.mlmodelc|chunk folder|-> <out_prefix> [tae_decoder.mlmodelc] [options]
//   "-" runs the diffusion model with ggml (Metal); a .mlmodelc or a folder of chunks runs it with the CoreML
//   sidecar. The optional tae_decoder.mlmodelc runs the tiny decoder with its CoreML sidecar too.
//   For a model in separate files (FLUX.2-klein), pass "-" as the model and --diffusion-model, --vae and --llm.
// options: --steps N  --cfg F  --sampler NAME  --taesd FILE (ggml TAESD, for models without a built-in
//   tiny decoder such as SD 2.1)  --images N (timed images, 1-3)  --mmap (map the model file instead of
//   reading it into memory; on iOS the mapped pages do not count against the app's memory limit)
//   --size WxH (image size; a CoreML UNet only runs the size it was exported for, such as 768x768 for SDXL)
//   --prediction eps|v (for a CoreML UNet that does not record it, such as Apple's exports)
//   --diffusion-model FILE  --vae FILE  --llm FILE (a model in separate files, such as FLUX.2-klein)
//   --backend ASSIGNMENTS (as sd-cli's --backend, e.g. te=cpu: with --mmap the text encoder then reads the mapped
//   file, whose pages do not count against iOS's memory limit)
//   --coreml-te DIR (FLUX.2-klein's text encoder as CoreML chunks; --llm then only provides the tokenizer)
//   --params-backend ASSIGNMENTS (as sd-cli's, e.g. te=disk: reload the text encoder's weights from the file and
//   release them after use; with SD_COREML_LOAD_ON_DEMAND=1 the CoreML denoiser is loaded only while denoising)
#include <algorithm>
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
        g_stage.text_ms = v * 1000;  // s
    } else if ((v = after(text, "sampling completed, taking ")) >= 0) {
        g_stage.unet_ms = v * 1000;  // s
    } else if ((v = after(text, "decode_first_stage completed, taking ")) >= 0) {
        g_stage.decode_ms = v * 1000;  // s
    }
    static const bool verbose = getenv("SDBENCH_VERBOSE") != nullptr;
    if (verbose || level >= SD_LOG_WARN || strstr(text, "coreml unet") || strstr(text, "coreml tae") ||
        strstr(text, "coreml flux") || strstr(text, "Core ML FLUX.2") || strstr(text, "coreml te") ||
        strstr(text, "Core ML text encoder") ||
        strstr(text, "total params memory")) {
        fputs(text, stdout);
    }
}

}  // namespace

int sd_bench_main(int argc, const char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s bench <model.gguf|-> <diffusion.mlmodelc|-> <out_prefix> [tae_decoder.mlmodelc]\n", argv[0]);
        return 1;
    }
    const char* model   = strcmp(argv[2], "-") == 0 ? "" : argv[2];
    const char* coreml  = strcmp(argv[3], "-") == 0 ? "" : argv[3];
    std::string prefix  = argv[4];
    const char* tae     = argc > 5 && strncmp(argv[5], "--", 2) != 0 ? argv[5] : "";
    int steps           = 1;
    float cfg           = 1.0f;
    const char* sampler = "";
    const char* taesd   = "";
    int images          = 3;
    bool mmap           = false;
    const char* pred    = "";
    int width           = 512;
    int height          = 512;
    const char* diffusion_model = "";
    const char* vae             = "";
    const char* llm             = "";
    const char* backend         = "";
    const char* params_backend  = "";
    const char* coreml_te       = "";
    for (int i = *tae ? 6 : 5; i < argc; i += 2) {
        if (strcmp(argv[i], "--mmap") == 0) {
            mmap = true;
            i--;  // no value
        } else if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", argv[i]);
            return 1;
        } else if (strcmp(argv[i], "--steps") == 0) {
            steps = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--cfg") == 0) {
            cfg = (float)atof(argv[i + 1]);
        } else if (strcmp(argv[i], "--sampler") == 0) {
            sampler = argv[i + 1];
        } else if (strcmp(argv[i], "--taesd") == 0) {
            taesd = argv[i + 1];
        } else if (strcmp(argv[i], "--prediction") == 0) {
            pred = argv[i + 1];
        } else if (strcmp(argv[i], "--size") == 0) {
            if (sscanf(argv[i + 1], "%dx%d", &width, &height) != 2) {
                fprintf(stderr, "--size expects WxH, got %s\n", argv[i + 1]);
                return 1;
            }
        } else if (strcmp(argv[i], "--diffusion-model") == 0) {
            diffusion_model = argv[i + 1];
        } else if (strcmp(argv[i], "--vae") == 0) {
            vae = argv[i + 1];
        } else if (strcmp(argv[i], "--llm") == 0) {
            llm = argv[i + 1];
        } else if (strcmp(argv[i], "--backend") == 0) {
            backend = argv[i + 1];
        } else if (strcmp(argv[i], "--params-backend") == 0) {
            params_backend = argv[i + 1];
        } else if (strcmp(argv[i], "--coreml-te") == 0) {
            coreml_te = argv[i + 1];
        } else if (strcmp(argv[i], "--images") == 0) {
            images = atoi(argv[i + 1]);
        } else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 1;
        }
    }

    struct Job {
        const char* name;
        const char* prompt;
    };
    std::vector<Job> jobs = {
        {"warmup", "a red apple on a wooden table"},
        {"cat", "portrait of a lovely cat"},
        {"cabin", "a cozy mountain cabin at sunset, snow, warm window lights, photorealistic"},
        {"billboard", "a large billboard on a city street showing an advertisement with the text \"COFFEE\", photo, daytime"},
    };
    jobs.resize(1 + std::max(1, std::min(images, 3)));

    sd_set_log_callback(log_cb, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path              = model;
    ctx_params.diffusion_model_path    = diffusion_model;
    ctx_params.vae_path                = vae;
    ctx_params.llm_path                = llm;
    ctx_params.backend                 = backend;
    ctx_params.params_backend          = params_backend;
    ctx_params.coreml_te_path          = coreml_te;
    ctx_params.coreml_diffusion_path   = coreml;
    ctx_params.coreml_tae_decoder_path = tae;
    ctx_params.taesd_path              = taesd;
    ctx_params.enable_mmap             = mmap;
    if (*pred) {
        ctx_params.prediction = str_to_prediction(pred);
    }

    auto t0       = std::chrono::steady_clock::now();
    sd_ctx_t* sd  = new_sd_ctx(&ctx_params);
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sd == nullptr) {
        fprintf(stdout, "[bench] new_sd_ctx failed\n");
        return 1;
    }
    fprintf(stdout, "[bench] diffusion=%s decoder=%s size=%dx%d steps=%d cfg=%.1f load=%.2f s\n", *coreml ? "coreml" : "metal",
            *tae ? "coreml" : (*taesd ? "taesd-metal" : "metal"), width, height, steps, cfg, load_s);

    for (const auto& job : jobs) {
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt                         = job.prompt;
        params.width                          = width;
        params.height                         = height;
        params.seed                           = 42;
        params.batch_count                    = 1;
        params.sample_params.sample_steps     = steps;
        params.sample_params.guidance.txt_cfg = cfg;
        params.sample_params.sample_method    = *sampler ? str_to_sample_method(sampler) : sd_get_default_sample_method(sd);
        params.sample_params.scheduler        = sd_get_default_scheduler(sd, params.sample_params.sample_method);

        g_stage         = {};
        t0              = std::chrono::steady_clock::now();
        sd_image_t* img = nullptr;
        int n_images    = 0;
        bool ok         = generate_image(sd, &params, &img, &n_images);
        double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok || n_images < 1 || img == nullptr || img->data == nullptr) {
            free_sd_images(img, n_images);
            fprintf(stdout, "[bench] %s failed\n", job.name);
            return 1;
        }
        std::string path = prefix + "_" + job.name + ".png";
        stbi_write_png(path.c_str(), img->width, img->height, img->channel, img->data, 0);
        free_sd_images(img, n_images);
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
