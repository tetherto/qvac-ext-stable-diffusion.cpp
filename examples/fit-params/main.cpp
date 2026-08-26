#include <stdio.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#include "stable-diffusion.h"

#include "common/common.h"
#include "common/media_io.h"

namespace fs = std::filesystem;

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

static bool load_images_from_dir(const std::string& dir,
                                 std::vector<SDImageOwner>& images,
                                 int expected_width,
                                 int expected_height,
                                 int max_image_num,
                                 bool verbose) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        fprintf(stderr, "'%s' is not a valid directory\n", dir.c_str());
        return false;
    }

    std::vector<fs::directory_entry> entries;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.path().filename().string() < b.path().filename().string();
    });

    for (const auto& entry : entries) {
        std::string path = entry.path().string();
        std::string ext  = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp" && ext != ".webp") {
            continue;
        }
        if (verbose) {
            fprintf(stderr, "load image %zu from '%s'\n", images.size(), path.c_str());
        }
        int width             = 0;
        int height            = 0;
        uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height, expected_width, expected_height);
        if (image_buffer == nullptr) {
            fprintf(stderr, "load image from '%s' failed\n", path.c_str());
            return false;
        }
        images.emplace_back(sd_image_t{static_cast<uint32_t>(width),
                                       static_cast<uint32_t>(height),
                                       3,
                                       image_buffer});
        if (max_image_num > 0 && static_cast<int>(images.size()) >= max_image_num) {
            break;
        }
    }
    return true;
}

static bool load_generation_inputs(SDGenerationParams& params, SDMode mode, bool verbose) {
    auto load_image = [&](const std::string& path,
                          SDImageOwner& image,
                          bool resize_image = true,
                          int channels      = 3) {
        int width  = resize_image && params.width_and_height_are_set() ? params.width : 0;
        int height = resize_image && params.width_and_height_are_set() ? params.height : 0;
        if (!load_sd_image_from_file(image.put(), path.c_str(), width, height, channels)) {
            fprintf(stderr, "failed to load image from '%s'\n", path.c_str());
            return false;
        }
        params.set_width_and_height_if_unset(image.get().width, image.get().height);
        return true;
    };
    auto load_audio = [&](const std::string& path, SDAudioOwner& audio) {
        std::vector<float> samples;
        uint32_t sample_rate = 0;
        uint32_t channels    = 0;
        if (!load_wav_from_file(path, samples, sample_rate, channels)) {
            fprintf(stderr, "failed to load WAV audio from '%s'\n", path.c_str());
            return false;
        }
        audio.reset(std::move(samples), sample_rate, channels);
        return true;
    };

    if ((!params.init_image_path.empty() && !load_image(params.init_image_path, params.init_image)) ||
        (!params.end_image_path.empty() && !load_image(params.end_image_path, params.end_image))) {
        return false;
    }
    params.ref_images.clear();
    for (const auto& path : params.ref_image_paths) {
        SDImageOwner image({0, 0, 3, nullptr});
        if (!load_image(path, image, false)) {
            return false;
        }
        params.ref_images.push_back(std::move(image));
    }
    if (!params.validate(mode)) {
        return false;
    }

    params.ref_videos.clear();
    for (const auto& path : params.ref_video_paths) {
        std::vector<SDImageOwner> frames;
        if (!load_images_from_dir(path, frames, 0, 0, 0, verbose) || frames.empty()) {
            fprintf(stderr, "failed to load reference video frames from '%s'\n", path.c_str());
            return false;
        }
        params.ref_videos.push_back(std::move(frames));
    }
    params.ref_video_audios.clear();
    params.ref_video_audios.resize(params.ref_videos.size());
    for (size_t i = 0; i < params.ref_video_audio_paths.size(); ++i) {
        if (!load_audio(params.ref_video_audio_paths[i], params.ref_video_audios[i])) {
            return false;
        }
    }
    params.ref_audios.clear();
    params.ref_audios.resize(params.ref_audio_paths.size());
    for (size_t i = 0; i < params.ref_audio_paths.size(); ++i) {
        if (!load_audio(params.ref_audio_paths[i], params.ref_audios[i])) {
            return false;
        }
    }

    if (!params.mask_image_path.empty() &&
        !load_image(params.mask_image_path, params.mask_image, true, 1)) {
        return false;
    }
    if (!params.control_image_path.empty() &&
        !load_image(params.control_image_path, params.control_image)) {
        return false;
    }
    if (!params.ip_adapter_image_path.empty() &&
        !load_image(params.ip_adapter_image_path, params.ip_adapter_image, false)) {
        return false;
    }
    if (!params.control_video_path.empty()) {
        params.control_frames.clear();
        if (!load_images_from_dir(params.control_video_path,
                                  params.control_frames,
                                  params.get_resolved_width(),
                                  params.get_resolved_height(),
                                  params.video_frames,
                                  verbose)) {
            return false;
        }
    }
    if (!params.pm_id_images_dir.empty()) {
        params.pm_id_images.clear();
        if (!load_images_from_dir(params.pm_id_images_dir,
                                  params.pm_id_images,
                                  0,
                                  0,
                                  0,
                                  verbose)) {
            return false;
        }
    }
    return true;
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
    if (!load_generation_inputs(gen_params, mode, fit_params.verbose)) {
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

    sd_img_gen_params_t image_request;
    sd_vid_gen_params_t video_request;
    if (mode == VID_GEN) {
        video_request             = gen_params.to_sd_vid_gen_params_t();
        workload.video_gen_params = &video_request;
    } else {
        image_request             = gen_params.to_sd_img_gen_params_t();
        workload.image_gen_params = &image_request;
    }

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
