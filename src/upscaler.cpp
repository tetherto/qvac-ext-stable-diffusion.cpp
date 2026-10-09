#include "upscaler.h"
#include "core/ggml_extend.hpp"
#include "core/util.h"
#include "model_loader.h"
#include "stable-diffusion.h"

#include <cstdlib>
#include <limits>
#include <utility>

UpscalerGGML::UpscalerGGML(int n_threads,
                           bool direct,
                           int tile_size,
                           std::string backend_spec,
                           std::string params_backend_spec)
    : n_threads(n_threads),
      direct(direct),
      tile_size(tile_size),
      backend_spec(std::move(backend_spec)),
      params_backend_spec(std::move(params_backend_spec)) {
}

UpscalerGGML::~UpscalerGGML() {
    // ModelManager holds raw ggml tensor pointers owned by the runner context.
    model_manager.reset();
    esrgan_upscaler.reset();
}

void UpscalerGGML::set_max_graph_vram_bytes(size_t max_vram_bytes) {
    max_graph_vram_bytes = max_vram_bytes;
    if (esrgan_upscaler) {
        esrgan_upscaler->set_max_graph_vram_bytes(max_vram_bytes);
    }
}

void UpscalerGGML::set_stream_layers_enabled(bool enabled) {
    stream_layers_enabled = enabled;
    if (esrgan_upscaler) {
        esrgan_upscaler->set_stream_layers_enabled(enabled);
    }
}

bool UpscalerGGML::load_from_file(const std::string& esrgan_path,
                                  int n_threads) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    std::string error;
    if (!backend_manager.init(backend_spec.c_str(),
                              params_backend_spec.c_str(),
                              /*split_mode_spec=*/nullptr,
                              false,
                              &error)) {
        LOG_ERROR("upscaler backend config failed: %s", error.c_str());
        return false;
    }
    auto backend_for = [&](SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.runtime_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
        }
        return module_backend;
    };
    auto params_backend_for = [&](SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.params_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
        }
        return module_backend;
    };
    auto ensure_backend_pair = [&](SDBackendModule module) {
        if (backend_for(module) == nullptr) {
            return false;
        }
        return params_backend_for(module) != nullptr;
    };
    if (!ensure_backend_pair(SDBackendModule::UPSCALER)) {
        return false;
    }
    actual_backend_device = sd_backend_is_cpu(backend_for(SDBackendModule::UPSCALER)) ? 0 : 1;
    LOG_INFO("ESRGAN upscaler compute backend: %s", actual_backend_device == 0 ? "CPU" : "GPU (accelerated)");

    model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(n_threads);
    model_manager->set_enable_mmap(false);

    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file_and_convert_name(esrgan_path, "", VERSION_ESRGAN)) {
        LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
        return false;
    }
    model_loader.set_wtype_override(model_data_type);
    LOG_INFO("Upscaler weight type: %s", ggml_type_name(model_data_type));
    esrgan_upscaler = std::make_shared<ESRGAN>(backend_for(SDBackendModule::UPSCALER),
                                               model_loader.get_tensor_storage_map(),
                                               model_manager);
    if (esrgan_upscaler == nullptr || esrgan_upscaler->rrdb_net == nullptr) {
        LOG_ERROR("init esrgan model from metadata failed: '%s'", esrgan_path.c_str());
        return false;
    }
    esrgan_upscaler->set_fit_module(SDBackendModule::UPSCALER);
    esrgan_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
    esrgan_upscaler->set_stream_layers_enabled(stream_layers_enabled);
    if (direct) {
        esrgan_upscaler->set_conv2d_direct_enabled(true);
    }

    std::map<std::string, ggml_tensor*> tensors;
    esrgan_upscaler->get_param_tensors(tensors);
    if (!model_manager->register_param_tensors("ESRGAN",
                                               std::move(tensors),
                                               backend_manager.params_backend_is_disk(SDBackendModule::UPSCALER) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend,
                                               backend_for(SDBackendModule::UPSCALER),
                                               params_backend_for(SDBackendModule::UPSCALER)) ||
        !model_manager->validate_registered_tensors()) {
        LOG_ERROR("register esrgan tensors with model manager failed");
        return false;
    }
    return true;
}

sd::Tensor<float> UpscalerGGML::upscale_tensor(const sd::Tensor<float>& input_tensor) {
    sd::Tensor<float> upscaled;
    const int scale = esrgan_upscaler->config.scale;
    if (tile_size <= 0 || (input_tensor.shape()[0] <= tile_size && input_tensor.shape()[1] <= tile_size)) {
        upscaled = esrgan_upscaler->compute(n_threads, input_tensor);
    } else {
        auto on_processing = [&](const sd::Tensor<float>& input_tile) -> sd::Tensor<float> {
            auto output_tile = esrgan_upscaler->compute(n_threads, input_tile);
            if (output_tile.empty()) {
                LOG_ERROR("esrgan compute failed while processing a tile");
                return {};
            }
            return output_tile;
        };

        upscaled = process_tiles_2d(input_tensor,
                                    static_cast<int>(input_tensor.shape()[0] * scale),
                                    static_cast<int>(input_tensor.shape()[1] * scale),
                                    scale,
                                    tile_size,
                                    tile_size,
                                    0.25f,
                                    false,
                                    false,
                                    on_processing);
    }
    esrgan_upscaler->free_compute_buffer();
    if (upscaled.empty()) {
        LOG_ERROR("esrgan compute failed");
        return {};
    }
    return upscaled;
}

sd_image_t UpscalerGGML::upscale(sd_image_t input_image, uint32_t upscale_factor) {
    // upscale_factor, unused for RealESRGAN_x4plus_anime_6B.pth
    sd_image_t upscaled_image = {0, 0, 0, nullptr};
    const int scale           = esrgan_upscaler->config.scale;
    int output_width          = (int)input_image.width * scale;
    int output_height         = (int)input_image.height * scale;
    LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
             input_image.width, input_image.height, output_width, output_height);

    sd::Tensor<float> input_tensor = sd_image_to_tensor(input_image);
    sd::Tensor<float> upscaled;
    int64_t t0 = ggml_time_ms();
    upscaled   = upscale_tensor(input_tensor);
    if (upscaled.empty()) {
        return upscaled_image;
    }
    sd_image_t upscaled_data = tensor_to_sd_image(upscaled);
    int64_t t3               = ggml_time_ms();
    LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
    upscaled_image = upscaled_data;
    return upscaled_image;
}

struct upscaler_ctx_t {
    UpscalerGGML* upscaler = nullptr;
};

upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path_c_str,
                                 bool direct,
                                 int n_threads,
                                 int tile_size,
                                 const char* backend,
                                 const char* params_backend) {
    upscaler_ctx_t* upscaler_ctx = (upscaler_ctx_t*)malloc(sizeof(upscaler_ctx_t));
    if (upscaler_ctx == nullptr) {
        return nullptr;
    }
    std::string esrgan_path(esrgan_path_c_str);

    upscaler_ctx->upscaler = new UpscalerGGML(n_threads, direct, tile_size, SAFE_STR(backend), SAFE_STR(params_backend));
    if (upscaler_ctx->upscaler == nullptr) {
        return nullptr;
    }

    if (!upscaler_ctx->upscaler->load_from_file(esrgan_path, n_threads)) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
        free(upscaler_ctx);
        return nullptr;
    }
    return upscaler_ctx;
}

// qvac: map a high-level (device, gpu_backend_pref) preference onto upstream's
// backend_manager spec string so the downstream device-selection API keeps
// working on top of the refactored backend system.
static const char* upscaler_pref_to_backend_spec(sd_upscaler_device_t device,
                                                  sd_backend_preference_t gpu_backend_pref) {
    if (device == SD_UPSCALER_DEVICE_CPU) {
        return "cpu";
    }
    switch (gpu_backend_pref) {
        case SD_BACKEND_PREF_CPU: return "cpu";
        case SD_BACKEND_PREF_OPENCL: return "opencl";
        case SD_BACKEND_PREF_GPU:
        default: return "gpu";
    }
}

void sd_upscaler_fit_params_init(sd_upscaler_fit_params_t* params) {
    if (params == nullptr) {
        return;
    }
    *params = {};
    params->n_threads = -1;
    params->tile_size = 128;
    params->width = 512;
    params->height = 512;
    params->repeats          = 1;
    params->device           = SD_UPSCALER_DEVICE_GPU;
    params->gpu_backend_pref = SD_BACKEND_PREF_GPU;
}

int sd_upscaler_model_scale(const char* esrgan_path) {
    if (esrgan_path == nullptr || esrgan_path[0] == '\0') {
        return 0;
    }
    try {
        SDMetadataOnlyReadScope metadata_only;
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(esrgan_path, "", VERSION_ESRGAN)) {
            return 0;
        }
        const auto& tensors = loader.get_tensor_storage_map();
        if (tensors.count("conv_first.weight") == 0 || tensors.count("conv_last.weight") == 0) {
            return 0;
        }
        return ESRGANConfig::detect_from_weights(tensors).scale;
    } catch (const std::exception&) {
        return 0;
    }
}

bool sd_measure_upscaler(const sd_upscaler_fit_params_t& params,
                         sd::fit_params::ModuleMemory& memory) {
    if (params.esrgan_path == nullptr || params.esrgan_path[0] == '\0' ||
        params.width <= 0 || params.height <= 0 || params.repeats <= 0 ||
        (params.device != SD_UPSCALER_DEVICE_CPU && params.device != SD_UPSCALER_DEVICE_GPU) ||
        (params.gpu_backend_pref != SD_BACKEND_PREF_CPU && params.gpu_backend_pref != SD_BACKEND_PREF_GPU &&
         params.gpu_backend_pref != SD_BACKEND_PREF_OPENCL)) {
        return false;
    }
    SDMetadataOnlyReadScope metadata_only;
    UpscalerGGML upscaler(params.n_threads, params.direct, params.tile_size,
                          upscaler_pref_to_backend_spec(params.device, params.gpu_backend_pref),
                          params.offload_params_to_cpu ? "cpu" : "");
    if (!upscaler.load_from_file(params.esrgan_path, params.n_threads)) {
        return false;
    }
    const int scale = upscaler.esrgan_upscaler->config.scale;
    if (scale <= 0) {
        return false;
    }
    int width  = params.width;
    int height = params.height;
    if (scale > 1) {
        for (int repeat = 1; repeat < params.repeats; ++repeat) {
            if (width > std::numeric_limits<int>::max() / scale ||
                height > std::numeric_limits<int>::max() / scale) {
                return false;
            }
            width *= scale;
            height *= scale;
        }
    }
    if (width > std::numeric_limits<int>::max() / scale ||
        height > std::numeric_limits<int>::max() / scale) {
        return false;
    }
    const uint64_t input_pixels_u64  = (uint64_t)width * height;
    const uint64_t output_pixels_u64 = (uint64_t)(width * scale) * (height * scale);
    // Covers float input/output, tile copies, RGB output and the caller's input.
    // Keep arithmetic below the signed byte counts used by the placement planner.
    const uint64_t byte_limit = std::min<uint64_t>(std::numeric_limits<size_t>::max() / 64,
                                                   std::numeric_limits<int64_t>::max() / 64);
    if (input_pixels_u64 > byte_limit || output_pixels_u64 > byte_limit) {
        return false;
    }
    const size_t input_pixels    = (size_t)input_pixels_u64;
    const size_t output_pixels   = (size_t)output_pixels_u64;
    const bool tiled             = params.tile_size > 0 && (width > params.tile_size || height > params.tile_size);
    const int tile_width         = tiled ? std::min(width, params.tile_size) : width;
    const int tile_height        = tiled ? std::min(height, params.tile_size) : height;
    const size_t tile_pixels     = (size_t)tile_width * tile_height;
    const size_t original_bytes  = (size_t)params.width * params.height * 3;
    const size_t retained_input  = params.repeats > 1 ? input_pixels * 3 : 0;
    const size_t conversion_peak = original_bytes + retained_input + input_pixels * 12 + output_pixels * 15;
    const size_t tile_peak       = original_bytes + retained_input + input_pixels * 12 + output_pixels * 12 +
                                   (tiled ? tile_pixels * 12 * (1 + scale * scale) : 0);
    memory                       = {};
    memory.module                = SDBackendModule::UPSCALER;
    memory.host_bytes            = std::max(conversion_peak, tile_peak);
    memory.runtime_on_cpu        = params.device == SD_UPSCALER_DEVICE_CPU || params.gpu_backend_pref == SD_BACKEND_PREF_CPU;
    memory.params_on_cpu         = params.offload_params_to_cpu;
    memory.fixed_residency       = true;
    std::vector<GGMLRunner::graph_memory_measurement> records;
    struct MeasureModeGuard {
        explicit MeasureModeGuard(std::vector<GGMLRunner::graph_memory_measurement>* records) {
            GGMLRunner::set_measure_mode(true, records);
        }
        ~MeasureModeGuard() {
            GGMLRunner::set_measure_mode(false);
        }
    } measure_guard(&records);
    if (!upscaler.esrgan_upscaler->measure_memory(tile_width, tile_height, params.n_threads) || records.empty()) {
        return false;
    }
    std::map<std::string, ggml_tensor*> tensors;
    upscaler.esrgan_upscaler->get_param_tensors(tensors);
    for (const auto& entry : tensors) {
        memory.params_bytes += ggml_nbytes(entry.second);
    }
    for (const auto& record : records) {
        memory.compute_bytes = std::max(memory.compute_bytes, record.compute_bytes);
    }
    return true;
}

enum sd_fit_status_t sd_upscaler_fit_params(const sd_upscaler_fit_params_t* params,
                                            sd_fit_result_t* result) {
    if (result == nullptr) {
        return SD_FIT_ERROR;
    }
    *result = {};
    if (params == nullptr) {
        return SD_FIT_ERROR;
    }
    if (params->gpu_backend_pref == SD_BACKEND_PREF_OPENCL && params->device != SD_UPSCALER_DEVICE_CPU) {
        return SD_FIT_FAILURE;
    }
    try {
        ggml_time_init();
        sd::fit_params::ModuleMemory memory;
        if (!sd_measure_upscaler(*params, memory)) {
            return SD_FIT_ERROR;
        }
        sd::ggml_graph_cut::MaxVramAssignment budgets;
        budgets.reset(0.f);
        sd::fit_params::FitPlan plan;
        const bool planned = sd::fit_params::plan_placement(
            {memory}, budgets, &plan, false,
            params->device == SD_UPSCALER_DEVICE_CPU || params->gpu_backend_pref == SD_BACKEND_PREF_CPU);
        auto copy_string = [](const std::string& value) {
            auto* copy = static_cast<char*>(malloc(value.size() + 1));
            if (copy == nullptr) {
                throw std::bad_alloc();
            }
            memcpy(copy, value.c_str(), value.size() + 1);
            return copy;
        };
        result->report  = copy_string(plan.report);
        result->changed = plan.changed;
        if (!plan.runtime_spec.empty()) {
            result->backend = copy_string(plan.runtime_spec);
        }
        if (!plan.params_spec.empty()) {
            result->params_backend = copy_string(plan.params_spec);
        }
        return planned && plan.valid ? SD_FIT_SUCCESS : SD_FIT_FAILURE;
    } catch (const std::exception& error) {
        sd_fit_result_free(result);
        LOG_ERROR("upscaler fit: %s", error.what());
        return SD_FIT_ERROR;
    }
}

upscaler_ctx_t* new_upscaler_ctx_with_device(const char* esrgan_path_c_str,
                                             bool offload_params_to_cpu,
                                             bool direct,
                                             int n_threads,
                                             int tile_size,
                                             sd_upscaler_device_t device,
                                             sd_backend_preference_t gpu_backend_pref) {
    const char* backend_spec = upscaler_pref_to_backend_spec(device, gpu_backend_pref);
    const char* params_backend_spec = offload_params_to_cpu ? "cpu" : nullptr;
    return new_upscaler_ctx(esrgan_path_c_str,
                            direct,
                            n_threads,
                            tile_size,
                            backend_spec,
                            params_backend_spec);
}

int get_upscaler_backend_device(const upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr) {
        return -1;
    }
    return upscaler_ctx->upscaler->actual_backend_device;
}

bool upscale(upscaler_ctx_t* upscaler_ctx,
             sd_image_t input_image,
             uint32_t upscale_factor,
             sd_image_t** images_out,
             int* num_images_out) {
    if (images_out != nullptr) {
        *images_out = nullptr;
    }
    if (num_images_out != nullptr) {
        *num_images_out = 0;
    }
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr) {
        return false;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(1, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return false;
    }

    result_images[0] = upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
    if (result_images[0].data == nullptr) {
        free(result_images);
        return false;
    }

    if (num_images_out != nullptr) {
        *num_images_out = 1;
    }
    if (images_out != nullptr) {
        *images_out = result_images;
    } else {
        free_sd_images(result_images, 1);
    }
    return true;
}

int get_upscale_factor(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr || upscaler_ctx->upscaler->esrgan_upscaler == nullptr) {
        return 1;
    }
    return upscaler_ctx->upscaler->esrgan_upscaler->config.scale;
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx->upscaler != nullptr) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
    }
    free(upscaler_ctx);
}
