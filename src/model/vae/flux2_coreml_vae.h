#ifndef __SD_MODEL_VAE_FLUX2_COREML_VAE_H__
#define __SD_MODEL_VAE_FLUX2_COREML_VAE_H__

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "model/vae/auto_encoder_kl.hpp"
#ifdef SD_USE_COREML
#include "flux_coreml.h"
#endif

struct Flux2CoreMLVAE : public AutoEncoderKL {
    bool coreml_requested_ = false;
    bool capture_written_  = false;
#ifdef SD_USE_COREML
    FluxCoreMLVAEModel* coreml_ = nullptr;
#endif

    Flux2CoreMLVAE(ggml_backend_t backend,
                   const String2TensorStorage& tensors,
                   const std::string& prefix,
                   SDVersion version,
                   std::shared_ptr<RunnerWeightManager> manager)
        : AutoEncoderKL(backend, tensors, prefix, false, false, version, manager) {
        const char* path  = std::getenv("SDCPP_FLUX2_COREML_VAE_MODEL");
        coreml_requested_ = path && path[0];
        if (!coreml_requested_)
            return;
        if (!sd_version_uses_flux2_vae(version)) {
            LOG_ERROR("SDCPP_FLUX2_COREML_VAE_MODEL requires a FLUX.2 VAE");
            return;
        }
#ifdef SD_USE_COREML
        std::string error;
        coreml_ = flux_coreml_vae_open(path, &error);
        if (!coreml_)
            LOG_ERROR("Core ML FLUX.2 VAE could not be loaded: %s", error.c_str());
        else
            LOG_INFO("Using Core ML FLUX.2 VAE decoder (CPU+GPU): %s", path);
#else
        LOG_ERROR("SDCPP_FLUX2_COREML_VAE_MODEL requires a build with SD_COREML=ON");
#endif
    }

    ~Flux2CoreMLVAE() override {
#ifdef SD_USE_COREML
        flux_coreml_vae_close(coreml_);
#endif
    }

    bool is_coreml_requested() const { return coreml_requested_; }
    bool is_coreml_ready() const {
#ifdef SD_USE_COREML
        return coreml_ != nullptr;
#else
        return false;
#endif
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (!coreml_requested_) {
            AutoEncoderKL::get_param_tensors(tensors);
            return;
        }
        std::map<std::string, ggml_tensor*> native;
        AutoEncoderKL::get_param_tensors(native);
        for (const auto& entry : native) {
            if (coreml_requested_ &&
                (starts_with(entry.first, weight_prefix + ".decoder.") ||
                 starts_with(entry.first, weight_prefix + ".post_quant_conv.")))
                continue;
            tensors.insert(entry);
        }
    }

    sd::Tensor<float> decode(int n_threads,
                             const sd::Tensor<float>& x,
                             sd_tiling_params_t tiling,
                             bool decode_video = false,
                             bool circular_x   = false,
                             bool circular_y   = false,
                             bool silent       = false) override {
        if (coreml_requested_ && (decode_video || tiling.enabled || tiling.temporal_tiling || circular_x || circular_y)) {
            LOG_ERROR("Core ML FLUX.2 VAE does not support video, tiling, or circular padding");
            return {};
        }
        return VAE::decode(n_threads, x, tiling, decode_video, circular_x, circular_y, silent);
    }

    bool capture_decode(const char* directory, const sd::Tensor<float>& input, const sd::Tensor<float>& output) {
        std::filesystem::path dir(directory);
        std::error_code error;
        std::filesystem::create_directories(dir, error);
        if (error || std::filesystem::exists(dir / "manifest.json"))
            return false;
        std::ostringstream manifest;
        manifest << "{\n  \"format\": 1,\n  \"kind\": \"flux2-vae-decode\",\n  \"tensors\": {\n";
        bool first = true;
        for (const auto& entry : {std::pair<const char*, const sd::Tensor<float>*>("latent", &input),
                                  std::pair<const char*, const sd::Tensor<float>*>("output", &output)}) {
            const auto& tensor          = *entry.second;
            const size_t bytes          = static_cast<size_t>(tensor.numel()) * sizeof(float);
            const std::string file_name = std::string(entry.first) + ".bin";
            std::ofstream file(dir / file_name, std::ios::binary);
            file.write(reinterpret_cast<const char*>(tensor.data()), static_cast<std::streamsize>(bytes));
            if (!file)
                return false;
            if (!first)
                manifest << ",\n";
            first = false;
            manifest << "    \"" << entry.first << "\": {\"file\": \"" << file_name
                     << "\", \"type\": \"f32\", \"ne\": [";
            for (int i = 0; i < 4; ++i) {
                if (i)
                    manifest << ", ";
                manifest << (i < tensor.dim() ? tensor.shape()[i] : 1);
            }
            manifest << "], \"bytes\": " << bytes << "}";
        }
        manifest << "\n  }\n}\n";
        std::ofstream file(dir / "manifest.json");
        file << manifest.str();
        return static_cast<bool>(file);
    }

    sd::Tensor<float> _compute(int n_threads, const sd::Tensor<float>& z, bool decode_graph) override {
        sd::Tensor<float> output;
        if (decode_graph && coreml_requested_) {
#ifdef SD_USE_COREML
            if (!coreml_ || weight_adapter || z.empty() || z.dim() < 3 || z.dim() > 4) {
                LOG_ERROR("Core ML FLUX.2 VAE received unsupported inputs");
                return {};
            }
            FluxCoreMLTensor input{};
            input.data = z.data();
            for (int i = 0; i < 4; ++i)
                input.ne[i] = i < z.dim() ? z.shape()[i] : 1;
            if (!flux_coreml_vae_matches_input(coreml_, input.ne)) {
                LOG_ERROR("Core ML VAE latent shape does not match the compiled decoder");
                return {};
            }
            auto shape = z.shape();
            shape[0] *= 16;
            shape[1] *= 16;
            shape[2] = 3;
            output   = sd::Tensor<float>(shape);
            if (measure_mode_enabled())
                return output;
            std::string error;
            if (!flux_coreml_vae_predict(coreml_, input, output.data(), &error)) {
                LOG_ERROR("Core ML FLUX.2 VAE decode failed: %s", error.c_str());
                return {};
            }
            for (int64_t i = 0; i < output.numel(); ++i) {
                if (!std::isfinite(output[i])) {
                    LOG_ERROR("Core ML FLUX.2 VAE returned a non-finite value");
                    return {};
                }
            }
#else
            return {};
#endif
        } else {
            output = AutoEncoderKL::_compute(n_threads, z, decode_graph);
        }
        if (!measure_mode_enabled() && decode_graph && sd_version_uses_flux2_vae(version) && !output.empty() && !capture_written_) {
            const char* directory = std::getenv("SDCPP_VAE_CAPTURE_DIR");
            if (directory && directory[0]) {
                capture_written_ = true;
                if (!capture_decode(directory, z, output))
                    LOG_WARN("FLUX.2 VAE fixture capture failed: %s", directory);
            }
        }
        return output;
    }
};

#endif  // __SD_MODEL_VAE_FLUX2_COREML_VAE_H__
