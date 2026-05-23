#include "esrgan.hpp"
#include "ggml_extend.hpp"
#include "model.h"
#include "stable-diffusion.h"
#include <cstring>
#include <cstdlib>

static const char* ggml_backend_device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "ACCEL";
        default:
            return "UNKNOWN";
    }
}

struct UpscalerGGML {
    ggml_backend_t backend    = nullptr;  // general backend
    ggml_type model_data_type = GGML_TYPE_F16;
    std::shared_ptr<ESRGAN> esrgan_upscaler;
    std::string esrgan_path;
    int n_threads;
    bool direct   = false;
    int tile_size = 128;
    // Post-init truth for stats: 0 = CPU, 1 = GPU (matches qvac RuntimeStats backendDevice).
    int actual_backend_device = 0;

    UpscalerGGML(int n_threads,
                 bool direct   = false,
                 int tile_size = 128)
        : n_threads(n_threads),
          direct(direct),
          tile_size(tile_size) {
    }

    // Keep aligned with StableDiffusionGGML::init_backend() in stable-diffusion.cpp.
    bool init_upscaler_backend(enum sd_backend_preference_t preferred_backend) {
        const char* pref_name = "gpu";
        if (preferred_backend == SD_BACKEND_PREF_CPU) {
            pref_name = "cpu";
        } else if (preferred_backend == SD_BACKEND_PREF_GPU) {
            pref_name = "gpu";
        } else if (preferred_backend == SD_BACKEND_PREF_OPENCL) {
            pref_name = "opencl";
        }
        LOG_INFO("ESRGAN upscaler backend preference: %s", pref_name);

        if (std::getenv("SD_CPU_ONLY")) {
            LOG_INFO("ESRGAN upscaler: SD_CPU_ONLY set - using CPU backend");
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
            if (!backend) {
                LOG_ERROR("ESRGAN upscaler: SD_CPU_ONLY set but CPU backend failed");
            }
            return backend != nullptr;
        }

        if (preferred_backend == SD_BACKEND_PREF_CPU) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
            if (backend) {
                LOG_INFO("ESRGAN upscaler: initialized CPU backend from preference");
            } else {
                LOG_WARN("ESRGAN upscaler: CPU backend preference requested but CPU backend initialization failed");
            }
            return backend != nullptr;
        }

        if (preferred_backend == SD_BACKEND_PREF_OPENCL) {
            const size_t n_devices = ggml_backend_dev_count();
            bool found_opencl_device = false;
            bool failed_opencl_init = false;
            LOG_INFO("ESRGAN upscaler OpenCL preference: probing %zu device(s)", n_devices);
            for (size_t i = 0; i < n_devices; ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
                if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                    dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    continue;
                }
                const char* name = ggml_backend_dev_name(dev);
                if (!name) {
                    continue;
                }
                const bool is_opencl = strstr(name, "opencl") != NULL ||
                                       strstr(name, "OpenCL") != NULL;
                if (!is_opencl) {
                    continue;
                }
                found_opencl_device = true;
                backend = ggml_backend_dev_init(dev, NULL);
                if (backend) {
                    LOG_INFO("ESRGAN upscaler: using OpenCL backend '%s'", name);
                    return true;
                }
                failed_opencl_init = true;
                LOG_WARN("ESRGAN upscaler: OpenCL candidate '%s' failed to initialize", name);
            }
            if (!found_opencl_device) {
                LOG_WARN("ESRGAN upscaler: OpenCL preference but no OpenCL device enumerated; falling back to generic GPU");
            } else if (failed_opencl_init) {
                LOG_WARN("ESRGAN upscaler: OpenCL init failed; falling back to generic GPU");
            }
        }

        const size_t n_devices = ggml_backend_dev_count();
        bool attempted_gpu_device_init = false;

        // Prefer dedicated GPUs first.
        for (size_t i = 0; i < n_devices; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
            if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            const char* name = ggml_backend_dev_name(dev);
            const char* desc = ggml_backend_dev_description(dev);
            LOG_INFO("ESRGAN upscaler GPU init candidate[%zu]: name='%s' desc='%s' type=%s", i, name ? name : "<null>", desc ? desc : "<null>", ggml_backend_device_type_name(dev_type));
            attempted_gpu_device_init = true;
            backend = ggml_backend_dev_init(dev, NULL);
            if (backend) {
                LOG_INFO("ESRGAN upscaler: initialized GPU backend from explicit device candidate[%zu] '%s'", i, name ? name : "<null>");
                break;
            }
            LOG_WARN("ESRGAN upscaler: failed to initialize GPU device candidate[%zu] '%s'", i, name ? name : "<null>");
        }

        // If no dedicated GPU worked, try integrated GPUs.
        if (!backend) {
            for (size_t i = 0; i < n_devices; ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
                if (dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    continue;
                }
                const char* name = ggml_backend_dev_name(dev);
                const char* desc = ggml_backend_dev_description(dev);
                LOG_INFO("ESRGAN upscaler IGPU init candidate[%zu]: name='%s' desc='%s' type=%s", i, name ? name : "<null>", desc ? desc : "<null>", ggml_backend_device_type_name(dev_type));
                attempted_gpu_device_init = true;
                backend = ggml_backend_dev_init(dev, NULL);
                if (backend) {
                    LOG_INFO("ESRGAN upscaler: initialized IGPU backend from explicit device candidate[%zu] '%s'", i, name ? name : "<null>");
                    break;
                }
                LOG_WARN("ESRGAN upscaler: failed to initialize IGPU device candidate[%zu] '%s'", i, name ? name : "<null>");
            }
        }

        if (!backend) {
            if (attempted_gpu_device_init) {
                LOG_WARN("ESRGAN upscaler: all explicit GPU device init attempts failed; trying generic GPU init by type");
            }
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
        }

        if (!backend) {
            LOG_WARN("ESRGAN upscaler: GPU init_by_type failed; falling back to CPU");
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
            if (!backend) {
                LOG_ERROR("ESRGAN upscaler: CPU fallback backend initialization failed");
            }
        } else {
            LOG_INFO("ESRGAN upscaler: initialized generic GPU backend");
        }
        return backend != nullptr;
    }

    bool load_from_file(const std::string& esrgan_path,
                        bool offload_params_to_cpu,
                        int n_threads,
                        sd_upscaler_device_t device_preference,
                        sd_backend_preference_t gpu_backend_pref) {
        ggml_log_set(ggml_log_callback_default, nullptr);

        sd_backend_preference_t backend_pref = gpu_backend_pref;
        if (device_preference == SD_UPSCALER_DEVICE_CPU) {
            backend_pref = SD_BACKEND_PREF_CPU;
        }

        backend = nullptr;
        if (!init_upscaler_backend(backend_pref)) {
            LOG_ERROR("ESRGAN upscaler: could not create any compute backend");
            return false;
        }

        actual_backend_device = ggml_backend_is_cpu(backend) ? 0 : 1;
        if (actual_backend_device == 0) {
            LOG_INFO("ESRGAN upscaler compute backend: CPU");
        } else {
            LOG_INFO("ESRGAN upscaler compute backend: GPU (accelerated)");
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(esrgan_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
        }
        model_loader.set_wtype_override(model_data_type);
        LOG_INFO("Upscaler weight type: %s", ggml_type_name(model_data_type));
        esrgan_upscaler = std::make_shared<ESRGAN>(backend, offload_params_to_cpu, tile_size, model_loader.get_tensor_storage_map());
        if (direct) {
            esrgan_upscaler->set_conv2d_direct_enabled(true);
        }
        if (!esrgan_upscaler->load_from_file(esrgan_path, n_threads)) {
            return false;
        }
        return true;
    }

    sd_image_t upscale(sd_image_t input_image, uint32_t upscale_factor) {
        // upscale_factor, unused for RealESRGAN_x4plus_anime_6B.pth
        sd_image_t upscaled_image = {0, 0, 0, nullptr};
        int output_width          = (int)input_image.width * esrgan_upscaler->scale;
        int output_height         = (int)input_image.height * esrgan_upscaler->scale;
        LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
                 input_image.width, input_image.height, output_width, output_height);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        // draft context
        struct ggml_context* upscale_ctx = ggml_init(params);
        if (!upscale_ctx) {
            LOG_ERROR("ggml_init() failed");
            return upscaled_image;
        }
        // LOG_DEBUG("upscale work buffer size: %.2f MB", params.mem_size / 1024.f / 1024.f);
        ggml_tensor* input_image_tensor = ggml_new_tensor_4d(upscale_ctx, GGML_TYPE_F32, input_image.width, input_image.height, 3, 1);
        sd_image_to_ggml_tensor(input_image, input_image_tensor);

        ggml_tensor* upscaled = ggml_new_tensor_4d(upscale_ctx, GGML_TYPE_F32, output_width, output_height, 3, 1);
        auto on_tiling        = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
            return esrgan_upscaler->compute(n_threads, in, &out);
        };
        int64_t t0 = ggml_time_ms();
        sd_tiling(input_image_tensor, upscaled, esrgan_upscaler->scale, esrgan_upscaler->tile_size, 0.25f, on_tiling);
        esrgan_upscaler->free_compute_buffer();
        ggml_ext_tensor_clamp_inplace(upscaled, 0.f, 1.f);
        uint8_t* upscaled_data = ggml_tensor_to_sd_image(upscaled);
        ggml_free(upscale_ctx);
        int64_t t3 = ggml_time_ms();
        LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
        upscaled_image = {
            (uint32_t)output_width,
            (uint32_t)output_height,
            3,
            upscaled_data,
        };
        return upscaled_image;
    }
};

struct upscaler_ctx_t {
    UpscalerGGML* upscaler = nullptr;
};

upscaler_ctx_t* new_upscaler_ctx_with_device(const char* esrgan_path_c_str,
                                            bool offload_params_to_cpu,
                                            bool direct,
                                            int n_threads,
                                            int tile_size,
                                            sd_upscaler_device_t device,
                                            sd_backend_preference_t gpu_backend_pref) {
    upscaler_ctx_t* upscaler_ctx = (upscaler_ctx_t*)malloc(sizeof(upscaler_ctx_t));
    if (upscaler_ctx == nullptr) {
        return nullptr;
    }
    std::string esrgan_path(esrgan_path_c_str);

    upscaler_ctx->upscaler = new UpscalerGGML(n_threads, direct, tile_size);
    if (upscaler_ctx->upscaler == nullptr) {
        free(upscaler_ctx);
        return nullptr;
    }

    if (!upscaler_ctx->upscaler->load_from_file(
            esrgan_path, offload_params_to_cpu, n_threads, device, gpu_backend_pref)) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
        free(upscaler_ctx);
        return nullptr;
    }
    return upscaler_ctx;
}

upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path_c_str,
                                 bool offload_params_to_cpu,
                                 bool direct,
                                 int n_threads,
                                 int tile_size) {
    return new_upscaler_ctx_with_device(
        esrgan_path_c_str, offload_params_to_cpu, direct, n_threads, tile_size,
        SD_UPSCALER_DEVICE_GPU, SD_BACKEND_PREF_GPU);
}

int get_upscaler_backend_device(const upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr) {
        return -1;
    }
    return upscaler_ctx->upscaler->actual_backend_device;
}

sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor) {
    return upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
}

int get_upscale_factor(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr || upscaler_ctx->upscaler->esrgan_upscaler == nullptr) {
        return 1;
    }
    return upscaler_ctx->upscaler->esrgan_upscaler->scale;
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx->upscaler != nullptr) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
    }
    free(upscaler_ctx);
}
