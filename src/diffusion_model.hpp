#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "anima.hpp"
#include "flux.hpp"
#include "mmdit.hpp"
#include "qwen_image.hpp"
#include "unet.hpp"
#include "wan.hpp"
#include "z_image.hpp"
#ifdef SD_USE_COREML
#include "flux_coreml.h"
#endif

struct DiffusionParams {
    struct ggml_tensor* x                     = nullptr;
    struct ggml_tensor* timesteps             = nullptr;
    struct ggml_tensor* context               = nullptr;
    struct ggml_tensor* c_concat              = nullptr;
    struct ggml_tensor* y                     = nullptr;
    struct ggml_tensor* guidance              = nullptr;
    std::vector<ggml_tensor*> ref_latents     = {};
    bool increase_ref_index                   = false;
    int num_video_frames                      = -1;
    std::vector<struct ggml_tensor*> controls = {};
    float control_strength                    = 0.f;
    struct ggml_tensor* vace_context          = nullptr;
    float vace_strength                       = 1.f;
    std::vector<int> skip_layers              = {};
};

// Capture real invocations for comparison with another denoiser backend.
// SDCPP_FLUX_CAPTURE_DIR writes the first call. Set SDCPP_FLUX_CAPTURE_ALL=1
// to write every call into a numbered subdirectory.
inline bool capture_flux_call(const char* directory, const DiffusionParams& params, ggml_tensor* output) {
    if (params.x == nullptr || params.timesteps == nullptr ||
        params.context == nullptr || output == nullptr ||
        params.c_concat != nullptr || !params.ref_latents.empty() || !params.skip_layers.empty()) {
        LOG_WARN("FLUX capture supports text-to-image calls without reference latents or skipped layers");
        return false;
    }
    std::filesystem::path dir(directory);
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if (error || std::filesystem::exists(dir / "manifest.json")) {
        LOG_WARN("FLUX capture directory is unavailable or already contains a manifest: %s", directory);
        return false;
    }

    std::vector<std::pair<const char*, ggml_tensor*>> tensors = {
        {"latent", params.x},
        {"timesteps", params.timesteps},
        {"context", params.context},
        {"pooled", params.y},
        {"guidance", params.guidance},
        {"output", output},
    };
    std::ostringstream manifest;
    manifest << "{\n  \"format\": 1,\n  \"tensors\": {\n";
    bool first = true;
    for (const auto& item : tensors) {
        ggml_tensor* tensor = item.second;
        if (tensor == nullptr)
            continue;
        if (!ggml_is_contiguous(tensor) || ggml_nbytes(tensor) == 0 ||
            (tensor->buffer == nullptr && tensor->data == nullptr)) {
            LOG_WARN("FLUX capture cannot read tensor %s", item.first);
            return false;
        }
        const size_t size = ggml_nbytes(tensor);
        std::vector<char> bytes(size);
        if (tensor->buffer != nullptr) {
            ggml_backend_tensor_get(tensor, bytes.data(), 0, size);
        } else {
            std::memcpy(bytes.data(), tensor->data, size);
        }
        std::string filename = std::string(item.first) + ".bin";
        std::ofstream file(dir / filename, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(size));
        if (!file) {
            LOG_WARN("FLUX capture could not write %s", filename.c_str());
            return false;
        }
        if (!first)
            manifest << ",\n";
        first = false;
        manifest << "    \"" << item.first << "\": {\"file\": \"" << filename
                 << "\", \"type\": \"" << ggml_type_name(tensor->type)
                 << "\", \"ne\": [";
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (i != 0)
                manifest << ", ";
            manifest << tensor->ne[i];
        }
        manifest << "], \"bytes\": " << size << "}";
    }
    manifest << "\n  }\n}\n";
    std::ofstream file(dir / "manifest.json", std::ios::binary);
    file << manifest.str();
    return static_cast<bool>(file);
}

struct DiffusionModel {
    virtual ~DiffusionModel()                                                           = default;
    virtual std::string get_desc()                                                      = 0;
    virtual bool compute(int n_threads,
                         DiffusionParams diffusion_params,
                         struct ggml_tensor** output     = nullptr,
                         struct ggml_context* output_ctx = nullptr)                     = 0;
    virtual void alloc_params_buffer()                                                  = 0;
    virtual void free_params_buffer()                                                   = 0;
    virtual void free_compute_buffer()                                                  = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                             = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter){};
    virtual int64_t get_adm_in_channels()                            = 0;
    virtual void set_flash_attention_enabled(bool enabled)           = 0;
    virtual void set_circular_axes(bool circular_x, bool circular_y) = 0;
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_SD1)
        : unet(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model", version) {
    }

    std::string get_desc() override {
        return unet.get_desc();
    }

    void alloc_params_buffer() override {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() override {
        unet.free_params_buffer();
    }

    void free_compute_buffer() override {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return unet.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        unet.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return unet.unet.adm_in_channels;
    }

    void set_flash_attention_enabled(bool enabled) {
        unet.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        unet.set_circular_axes(circular_x, circular_y);
    }

    bool compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return unet.compute(n_threads,
                            diffusion_params.x,
                            diffusion_params.timesteps,
                            diffusion_params.context,
                            diffusion_params.c_concat,
                            diffusion_params.y,
                            diffusion_params.num_video_frames,
                            diffusion_params.controls,
                            diffusion_params.control_strength, output, output_ctx);
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {})
        : mmdit(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model") {
    }

    std::string get_desc() override {
        return mmdit.get_desc();
    }

    void alloc_params_buffer() override {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() override {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() override {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return mmdit.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        mmdit.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768 + 1280;
    }

    void set_flash_attention_enabled(bool enabled) {
        mmdit.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        mmdit.set_circular_axes(circular_x, circular_y);
    }

    bool compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return mmdit.compute(n_threads,
                             diffusion_params.x,
                             diffusion_params.timesteps,
                             diffusion_params.context,
                             diffusion_params.y,
                             output,
                             output_ctx,
                             diffusion_params.skip_layers);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;
    bool coreml_requested      = false;
    bool coreml_adapter_active = false;
#ifdef SD_USE_COREML
    FluxCoreMLModel* coreml = nullptr;
#endif

    FluxModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_FLUX,
              bool use_mask                                  = false)
        : flux(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model", version, use_mask) {
        const char* path = std::getenv("SDCPP_FLUX2_COREML_MODEL");
        coreml_requested = path != nullptr && path[0] != '\0';
        if (coreml_requested && version != VERSION_FLUX2_KLEIN) {
            LOG_ERROR("SDCPP_FLUX2_COREML_MODEL only supports FLUX.2-klein");
        }
#ifdef SD_USE_COREML
        else if (coreml_requested) {
            std::string error;
            coreml = flux_coreml_open(path, &error);
            if (coreml == nullptr)
                LOG_ERROR("Core ML FLUX.2 model load failed: %s", error.c_str());
            else
                LOG_INFO("Using Core ML FLUX.2 denoiser: %s", path);
        }
#else
        else if (coreml_requested) {
            LOG_ERROR("SDCPP_FLUX2_COREML_MODEL requires a build with SD_COREML=ON");
        }
#endif
    }

    ~FluxModel() override {
#ifdef SD_USE_COREML
        flux_coreml_close(coreml);
#endif
    }

    bool is_coreml_requested() const { return coreml_requested; }
    bool is_coreml_ready() const {
#ifdef SD_USE_COREML
        return coreml_requested && coreml != nullptr;
#else
        return false;
#endif
    }

    std::string get_desc() override {
        return flux.get_desc();
    }

    void alloc_params_buffer() override {
        if (!coreml_requested)
            flux.alloc_params_buffer();
    }

    void free_params_buffer() override {
        if (!coreml_requested)
            flux.free_params_buffer();
    }

    void free_compute_buffer() override {
        if (!coreml_requested)
            flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        if (!coreml_requested)
            flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return coreml_requested ? 0 : flux.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (coreml_requested) {
            coreml_adapter_active = adapter != nullptr;
            if (coreml_adapter_active) {
                LOG_ERROR("LoRA adapters are not supported by the Core ML FLUX.2 denoiser");
            }
            return;
        }
        flux.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        flux.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        flux.set_circular_axes(circular_x, circular_y);
    }

    bool compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        if (diffusion_params.x != nullptr && diffusion_params.context != nullptr) {
            LOG_DEBUG("flux denoiser input: latent=[%lld,%lld,%lld,%lld] context=[%lld,%lld,%lld,%lld]",
                      static_cast<long long>(diffusion_params.x->ne[0]),
                      static_cast<long long>(diffusion_params.x->ne[1]),
                      static_cast<long long>(diffusion_params.x->ne[2]),
                      static_cast<long long>(diffusion_params.x->ne[3]),
                      static_cast<long long>(diffusion_params.context->ne[0]),
                      static_cast<long long>(diffusion_params.context->ne[1]),
                      static_cast<long long>(diffusion_params.context->ne[2]),
                      static_cast<long long>(diffusion_params.context->ne[3]));
        }
        const int64_t start_ms = ggml_time_ms();
        bool success           = false;
#ifdef SD_USE_COREML
        if (coreml_requested) {
            if (coreml_adapter_active) {
                LOG_ERROR("Core ML FLUX.2 computation cannot use a LoRA adapter");
                return false;
            }
            if (coreml == nullptr || output == nullptr ||
                (*output == nullptr && output_ctx == nullptr) ||
                diffusion_params.x == nullptr || diffusion_params.timesteps == nullptr ||
                diffusion_params.context == nullptr || diffusion_params.c_concat != nullptr ||
                !diffusion_params.ref_latents.empty() || !diffusion_params.skip_layers.empty()) {
                LOG_ERROR("Core ML FLUX.2 denoiser received unsupported inputs");
                return false;
            }
            auto read_input = [](ggml_tensor* tensor, std::vector<float>& storage, FluxCoreMLTensor& input) -> bool {
                if (!tensor || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor))
                    return false;
                storage.resize(static_cast<size_t>(ggml_nelements(tensor)));
                if (tensor->buffer)
                    ggml_backend_tensor_get(tensor, storage.data(), 0, ggml_nbytes(tensor));
                else if (tensor->data)
                    std::memcpy(storage.data(), tensor->data, ggml_nbytes(tensor));
                else
                    return false;
                input.data = storage.data();
                for (int i = 0; i < 4; ++i)
                    input.ne[i] = tensor->ne[i];
                return true;
            };
            std::vector<float> latent_data, timestep_data, context_data, pooled_data, guidance_data;
            FluxCoreMLTensor latent{}, timesteps{}, context{}, pooled{}, guidance{};
            if (!read_input(diffusion_params.x, latent_data, latent) ||
                !read_input(diffusion_params.timesteps, timestep_data, timesteps) ||
                !read_input(diffusion_params.context, context_data, context) ||
                (diffusion_params.y && !read_input(diffusion_params.y, pooled_data, pooled)) ||
                (diffusion_params.guidance && !read_input(diffusion_params.guidance, guidance_data, guidance))) {
                LOG_ERROR("Core ML FLUX.2 denoiser requires contiguous float32 inputs");
                return false;
            }
            if (*output == nullptr)
                *output = ggml_dup_tensor(output_ctx, diffusion_params.x);
            bool output_matches_latent = *output != nullptr &&
                                         (*output)->type == GGML_TYPE_F32 &&
                                         (*output)->data != nullptr &&
                                         ggml_is_contiguous(*output);
            if (output_matches_latent) {
                for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                    if ((*output)->ne[i] != diffusion_params.x->ne[i]) {
                        output_matches_latent = false;
                        break;
                    }
                }
            }
            if (!output_matches_latent) {
                LOG_ERROR("Core ML FLUX.2 denoiser could not allocate output tensor");
                return false;
            }
            std::string error;
            success = flux_coreml_predict(coreml, latent, timesteps, context,
                                          diffusion_params.y ? &pooled : nullptr,
                                          diffusion_params.guidance ? &guidance : nullptr,
                                          static_cast<float*>((*output)->data), &error);
            if (!success)
                LOG_ERROR("Core ML FLUX.2 prediction failed: %s", error.c_str());
        } else
#endif
            success = flux.compute(n_threads,
                                   diffusion_params.x,
                                   diffusion_params.timesteps,
                                   diffusion_params.context,
                                   diffusion_params.c_concat,
                                   diffusion_params.y,
                                   diffusion_params.guidance,
                                   diffusion_params.ref_latents,
                                   diffusion_params.increase_ref_index,
                                   output,
                                   output_ctx,
                                   diffusion_params.skip_layers);
        LOG_DEBUG("flux denoiser compute completed, taking %lld ms",
                  static_cast<long long>(ggml_time_ms() - start_ms));
        const char* capture_dir = std::getenv("SDCPP_FLUX_CAPTURE_DIR");
        if (success && capture_dir != nullptr && capture_dir[0] != '\0' &&
            output != nullptr && *output != nullptr) {
            static std::atomic<unsigned int> capture_index{0};
            const unsigned int index    = capture_index.fetch_add(1) + 1;
            const char* capture_all_env = std::getenv("SDCPP_FLUX_CAPTURE_ALL");
            const bool capture_all      = capture_all_env != nullptr && std::strcmp(capture_all_env, "1") == 0;
            if (index == 1 || capture_all) {
                const std::string directory = capture_all
                                                  ? (std::filesystem::path(capture_dir) /
                                                     ("call-" + std::to_string(index)))
                                                        .string()
                                                  : capture_dir;
                if (!capture_flux_call(directory.c_str(), diffusion_params, *output)) {
                    LOG_WARN("FLUX denoiser fixture capture failed: %s", directory.c_str());
                }
            }
        }
#ifdef SD_USE_COREML
        if (success && coreml_requested) {
            const float* values = static_cast<const float*>((*output)->data);
            for (int64_t i = 0; i < ggml_nelements(*output); ++i) {
                if (!std::isfinite(values[i])) {
                    LOG_ERROR("Core ML FLUX.2 prediction returned a non-finite value at element %lld",
                              static_cast<long long>(i));
                    return false;
                }
            }
        }
#endif
        return success;
    }
};

struct AnimaModel : public DiffusionModel {
    std::string prefix;
    Anima::AnimaRunner anima;

    AnimaModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {},
               const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), anima(backend, offload_params_to_cpu, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return anima.get_desc();
    }

    void alloc_params_buffer() override {
        anima.alloc_params_buffer();
    }

    void free_params_buffer() override {
        anima.free_params_buffer();
    }

    void free_compute_buffer() override {
        anima.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        anima.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return anima.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        anima.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        anima.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        anima.set_circular_axes(circular_x, circular_y);
    }

    bool compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return anima.compute(n_threads,
                             diffusion_params.x,
                             diffusion_params.timesteps,
                             diffusion_params.context,
                             diffusion_params.c_concat,
                             diffusion_params.y,
                             output,
                             output_ctx);
    }
};

struct WanModel : public DiffusionModel {
    std::string prefix;
    WAN::WanRunner wan;

    WanModel(ggml_backend_t backend,
             bool offload_params_to_cpu,
             const String2TensorStorage& tensor_storage_map = {},
             const std::string prefix                       = "model.diffusion_model",
             SDVersion version                              = VERSION_WAN2)
        : prefix(prefix), wan(backend, offload_params_to_cpu, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return wan.get_desc();
    }

    void alloc_params_buffer() override {
        wan.alloc_params_buffer();
    }

    void free_params_buffer() override {
        wan.free_params_buffer();
    }

    void free_compute_buffer() override {
        wan.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        wan.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return wan.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        wan.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        wan.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        wan.set_circular_axes(circular_x, circular_y);
    }

    bool compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return wan.compute(n_threads,
                           diffusion_params.x,
                           diffusion_params.timesteps,
                           diffusion_params.context,
                           diffusion_params.y,
                           diffusion_params.c_concat,
                           nullptr,
                           diffusion_params.vace_context,
                           diffusion_params.vace_strength,
                           output,
                           output_ctx);
    }
};

struct QwenImageModel : public DiffusionModel {
    std::string prefix;
    Qwen::QwenImageRunner qwen_image;

    QwenImageModel(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "model.diffusion_model",
                   SDVersion version                              = VERSION_QWEN_IMAGE,
                   bool zero_cond_t                               = false)
        : prefix(prefix), qwen_image(backend, offload_params_to_cpu, tensor_storage_map, prefix, version, zero_cond_t) {
    }

    std::string get_desc() override {
        return qwen_image.get_desc();
    }

    void alloc_params_buffer() override {
        qwen_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        qwen_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        qwen_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        qwen_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return qwen_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        qwen_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        qwen_image.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        qwen_image.set_circular_axes(circular_x, circular_y);
    }

    bool compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return qwen_image.compute(n_threads,
                                  diffusion_params.x,
                                  diffusion_params.timesteps,
                                  diffusion_params.context,
                                  diffusion_params.ref_latents,
                                  true,  // increase_ref_index
                                  output,
                                  output_ctx);
    }
};

struct ZImageModel : public DiffusionModel {
    std::string prefix;
    ZImage::ZImageRunner z_image;

    ZImageModel(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                const std::string prefix                       = "model.diffusion_model",
                SDVersion version                              = VERSION_Z_IMAGE)
        : prefix(prefix), z_image(backend, offload_params_to_cpu, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return z_image.get_desc();
    }

    void alloc_params_buffer() override {
        z_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        z_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        z_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        z_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return z_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        z_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        z_image.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        z_image.set_circular_axes(circular_x, circular_y);
    }

    bool compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return z_image.compute(n_threads,
                               diffusion_params.x,
                               diffusion_params.timesteps,
                               diffusion_params.context,
                               diffusion_params.ref_latents,
                               true,  // increase_ref_index
                               output,
                               output_ctx);
    }
};

#endif
