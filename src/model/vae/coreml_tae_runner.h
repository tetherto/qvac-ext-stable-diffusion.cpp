#ifndef __SD_MODEL_VAE_COREML_TAE_RUNNER_H__
#define __SD_MODEL_VAE_COREML_TAE_RUNNER_H__

#include <memory>
#include <string>

#include "model/vae/tae.hpp"

#ifdef SD_USE_COREML
#include "coreml_tae.h"
#endif

// Tiny image autoencoder whose untiled decode runs on a CoreML model (sd_ctx_params_t::coreml_tae_decoder_path,
// --coreml-tae; see docs/coreml.md). Encoding, tiled decoding and circular padding use the ggml decoder,
// which stays loaded: the CoreML graph has one fixed latent shape and the padding it was exported with.
struct CoreMLTinyImageAutoEncoder : public TinyImageAutoEncoder {
    bool coreml_requested = false;
#ifdef SD_USE_COREML
    std::unique_ptr<CoreMLTAEDecoder> coreml;
#endif

    CoreMLTinyImageAutoEncoder(ggml_backend_t backend,
                               const String2TensorStorage& tensor_storage_map,
                               const std::string prefix,
                               bool decoder_only,
                               SDVersion version,
                               std::shared_ptr<RunnerWeightManager> weight_manager,
                               const char* coreml_path)
        : TinyImageAutoEncoder(backend, tensor_storage_map, prefix, decoder_only, version, weight_manager) {
        coreml_requested = coreml_path && coreml_path[0] != '\0';
#ifdef SD_USE_COREML
        if (coreml_requested) {
            coreml = CoreMLTAEDecoder::load(coreml_path);
        }
#else
        if (coreml_requested) {
            LOG_ERROR("coreml_tae_decoder_path requires a build with SD_COREML=ON");
        }
#endif
    }

    bool is_coreml_ready() const {
#ifdef SD_USE_COREML
        return coreml_requested && coreml;
#else
        return false;
#endif
    }

    sd::Tensor<float> decode(int n_threads,
                             const sd::Tensor<float>& x,
                             sd_tiling_params_t tiling_params,
                             bool decode_video = false,
                             bool circular_x   = false,
                             bool circular_y   = false,
                             bool silent       = false) override {
#ifdef SD_USE_COREML
        if (coreml && !resolve_tiling_params(tiling_params).enabled && !decode_video && !circular_x && !circular_y &&
            (x.dim() == 3 || x.dim() == 4)) {
            // sd::Tensor shapes are in ggml order: [w, h, C, N] latent, [W, H, 3, N] image (N may be dropped when 1).
            const auto& s                  = x.shape();
            const int64_t n                = x.dim() == 4 ? s[3] : 1;
            const int64_t f                = coreml->scale();
            const int64_t latent_shape[4]  = {n, s[2], s[1], s[0]};
            const int64_t image_shape[4]   = {n, 3, s[1] * f, s[0] * f};
            std::vector<int64_t> out_shape = {s[0] * f, s[1] * f, 3};
            if (x.dim() == 4) {
                out_shape.push_back(n);
            }
            sd::Tensor<float> image(out_shape);
            if (measure_mode_enabled() || coreml->decode(x.data(), latent_shape, image.data(), image_shape)) {
                return image;
            }
            return {};
        }
#endif
        return TinyImageAutoEncoder::decode(n_threads, x, tiling_params, decode_video, circular_x, circular_y, silent);
    }
};

#endif  // __SD_MODEL_VAE_COREML_TAE_RUNNER_H__
