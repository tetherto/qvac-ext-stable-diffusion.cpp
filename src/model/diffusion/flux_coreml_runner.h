#ifndef __SD_MODEL_DIFFUSION_FLUX_COREML_RUNNER_H__
#define __SD_MODEL_DIFFUSION_FLUX_COREML_RUNNER_H__

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "model/diffusion/flux.hpp"
#ifdef SD_USE_COREML
#include "flux_coreml.h"
#endif

// Capture real invocations for comparison with another denoiser backend.
// SDCPP_FLUX_CAPTURE_DIR writes the first call. Set SDCPP_FLUX_CAPTURE_ALL=1
// to write every call into a numbered subdirectory.
inline bool capture_flux_call(const char* directory, const DiffusionParams& params, const sd::Tensor<float>& output) {
    const auto* extra = std::get_if<FluxDiffusionExtra>(&params.extra);
    if (params.x == nullptr || params.timesteps == nullptr ||
        params.context == nullptr || output.empty() ||
        params.c_concat != nullptr || (params.ref_latents && !params.ref_latents->empty()) ||
        (extra && extra->skip_layers && !extra->skip_layers->empty())) {
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

    std::vector<std::pair<const char*, const sd::Tensor<float>*>> tensors = {
        {"latent", params.x},
        {"timesteps", params.timesteps},
        {"context", params.context},
        {"pooled", params.y},
        {"guidance", extra ? extra->guidance : nullptr},
        {"output", &output},
    };
    std::ostringstream manifest;
    manifest << "{\n  \"format\": 1,\n  \"tensors\": {\n";
    bool first = true;
    for (const auto& item : tensors) {
        const auto* tensor = item.second;
        if (tensor == nullptr || tensor->empty())
            continue;
        if (tensor->dim() > 4) {
            LOG_WARN("FLUX capture cannot read tensor %s", item.first);
            return false;
        }
        const size_t size    = tensor->numel() * sizeof(float);
        std::string filename = std::string(item.first) + ".bin";
        std::ofstream file(dir / filename, std::ios::binary);
        file.write(reinterpret_cast<const char*>(tensor->data()), static_cast<std::streamsize>(size));
        if (!file) {
            LOG_WARN("FLUX capture could not write %s", filename.c_str());
            return false;
        }
        if (!first)
            manifest << ",\n";
        first = false;
        manifest << "    \"" << item.first << "\": {\"file\": \"" << filename
                 << "\", \"type\": \"" << "f32"
                 << "\", \"ne\": [";
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (i != 0)
                manifest << ", ";
            manifest << (i < tensor->dim() ? tensor->shape()[i] : 1);
        }
        manifest << "], \"bytes\": " << size << "}";
    }
    manifest << "\n  }\n}\n";
    std::ofstream file(dir / "manifest.json", std::ios::binary);
    file << manifest.str();
    return static_cast<bool>(file);
}

// Keep the native runner for model metadata and the normal GGML path. When a
// sidecar is selected, its weights are excluded from model-manager registration.
struct FluxCoreMLRunner : public Flux::FluxRunner {
    bool coreml_requested = false;
#ifdef SD_USE_COREML
    FluxCoreMLModel* coreml = nullptr;
#endif

    FluxCoreMLRunner(ggml_backend_t backend,
                     const String2TensorStorage& tensors          = {},
                     const std::string prefix                     = "",
                     SDVersion version                            = VERSION_FLUX,
                     std::shared_ptr<RunnerWeightManager> manager = nullptr,
                     const char* model_args                       = nullptr)
        : Flux::FluxRunner(backend, tensors, prefix, version, manager, model_args) {
        const char* path = std::getenv("SDCPP_FLUX2_COREML_MODEL");
        coreml_requested = path && path[0] != '\0';
        if (coreml_requested && version != VERSION_FLUX2_KLEIN) {
            LOG_ERROR("SDCPP_FLUX2_COREML_MODEL only supports FLUX.2-klein");
        }
#ifdef SD_USE_COREML
        else if (coreml_requested) {
            std::string error;
            coreml = flux_coreml_open(path, &error);
            if (!coreml)
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

    ~FluxCoreMLRunner() override {
#ifdef SD_USE_COREML
        flux_coreml_close(coreml);
#endif
    }

    bool is_coreml_requested() const { return coreml_requested; }
    bool is_coreml_ready() const {
#ifdef SD_USE_COREML
        return coreml_requested && coreml;
#else
        return false;
#endif
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
        if (!coreml_requested)
            Flux::FluxRunner::get_param_tensors(tensors, prefix);
    }

    sd::Tensor<float> compute(int n_threads, const DiffusionParams& params) override {
        const int64_t start_ms = ggml_time_ms();
        sd::Tensor<float> output;
        if (coreml_requested) {
#ifdef SD_USE_COREML
            const auto* extra = std::get_if<FluxDiffusionExtra>(&params.extra);
            if (!coreml || weight_adapter || !params.x || params.x->empty() ||
                !params.timesteps || !params.context ||
                (params.c_concat && !params.c_concat->empty()) ||
                (params.ref_latents && !params.ref_latents->empty()) ||
                (extra && extra->skip_layers && !extra->skip_layers->empty()) ||
                (extra && extra->pulid_id && !extra->pulid_id->empty())) {
                LOG_ERROR("Core ML FLUX.2 denoiser received unsupported inputs");
                return {};
            }
            auto read_input = [](const sd::Tensor<float>* tensor, FluxCoreMLTensor& input) {
                if (!tensor || tensor->empty() || tensor->dim() > 4)
                    return false;
                input.data = tensor->data();
                for (int i = 0; i < 4; ++i)
                    input.ne[i] = i < tensor->dim() ? tensor->shape()[i] : 1;
                return true;
            };
            const auto* guidance_tensor = extra ? extra->guidance : nullptr;
            FluxCoreMLTensor latent{}, timesteps{}, context{}, pooled{}, guidance{};
            const bool has_pooled   = params.y && !params.y->empty();
            const bool has_guidance = guidance_tensor && !guidance_tensor->empty();
            if (!read_input(params.x, latent) || !read_input(params.timesteps, timesteps) ||
                !read_input(params.context, context) ||
                (has_pooled && !read_input(params.y, pooled)) ||
                (has_guidance && !read_input(guidance_tensor, guidance))) {
                LOG_ERROR("Core ML FLUX.2 denoiser requires float32 inputs of rank <= 4");
                return {};
            }
            output = sd::Tensor<float>(params.x->shape());
            if (measure_mode_enabled())
                return output;
            std::string error;
            if (!flux_coreml_predict(coreml, latent, timesteps, context,
                                     has_pooled ? &pooled : nullptr,
                                     has_guidance ? &guidance : nullptr,
                                     output.data(), &error)) {
                LOG_ERROR("Core ML FLUX.2 prediction failed: %s", error.c_str());
                return {};
            }
#else
            return {};
#endif
        } else {
            output = Flux::FluxRunner::compute(n_threads, params);
        }
        if (measure_mode_enabled())
            return output;
        LOG_DEBUG("flux denoiser compute completed, taking %lld ms",
                  static_cast<long long>(ggml_time_ms() - start_ms));
        const char* capture_dir = std::getenv("SDCPP_FLUX_CAPTURE_DIR");
        if (!output.empty() && capture_dir && capture_dir[0] != '\0') {
            static std::atomic<unsigned int> capture_index{0};
            const unsigned int index    = capture_index.fetch_add(1) + 1;
            const char* capture_all_env = std::getenv("SDCPP_FLUX_CAPTURE_ALL");
            const bool capture_all      = capture_all_env && std::strcmp(capture_all_env, "1") == 0;
            if (index == 1 || capture_all) {
                const std::string directory = capture_all
                                                  ? (std::filesystem::path(capture_dir) / ("call-" + std::to_string(index))).string()
                                                  : capture_dir;
                if (!capture_flux_call(directory.c_str(), params, output))
                    LOG_WARN("FLUX denoiser fixture capture failed: %s", directory.c_str());
            }
        }
        if (coreml_requested) {
            for (int64_t i = 0; i < output.numel(); ++i) {
                if (!std::isfinite(output.data()[i])) {
                    LOG_ERROR("Core ML FLUX.2 prediction returned a non-finite value at element %lld",
                              static_cast<long long>(i));
                    return {};
                }
            }
        }
        return output;
    }
};
#endif
