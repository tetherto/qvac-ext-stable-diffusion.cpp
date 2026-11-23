#ifndef __SD_MODEL_DIFFUSION_COREML_UNET_RUNNER_H__
#define __SD_MODEL_DIFFUSION_COREML_UNET_RUNNER_H__

#include <memory>
#include <string>

#include "model/diffusion/unet.hpp"

#ifdef SD_USE_COREML
#include "coreml_unet.h"
#endif

// SD 1.x/2.x/SDXL UNet on a CoreML model (sd_ctx_params_t::coreml_unet_path, --coreml-unet; see docs/coreml.md).
// The native runner is kept for its config; with a sidecar its weights are not registered, so they are
// never loaded. Batched CFG (batch 2) works with models exported for batch 1 or 2.
struct CoreMLUNetRunner : public UNetModelRunner {
    bool coreml_requested = false;
#ifdef SD_USE_COREML
    std::unique_ptr<CoreMLUNet> coreml;
#endif

    CoreMLUNetRunner(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map,
                     const std::string prefix,
                     SDVersion version,
                     std::shared_ptr<RunnerWeightManager> weight_manager,
                     const char* coreml_path)
        : UNetModelRunner(backend, tensor_storage_map, prefix, version, weight_manager) {
        coreml_requested = coreml_path && coreml_path[0] != '\0';
#ifdef SD_USE_COREML
        if (coreml_requested) {
            coreml = CoreMLUNet::load(coreml_path);
        }
#else
        if (coreml_requested) {
            LOG_ERROR("coreml_unet_path requires a build with SD_COREML=ON");
        }
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

    // "epsilon" or "v_prediction" as recorded by convert_unet.py, or "" if the model does not say.
    std::string prediction_type() const {
#ifdef SD_USE_COREML
        return coreml ? coreml->prediction_type() : "";
#else
        return "";
#endif
    }

    std::string get_desc() override {
#ifdef SD_USE_COREML
        if (coreml) {
            return coreml->get_desc();
        }
#endif
        return UNetModelRunner::get_desc();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
        if (!coreml_requested) {
            UNetModelRunner::get_param_tensors(tensors, prefix);
        }
    }

    sd::Tensor<float> compute(int n_threads, const DiffusionParams& params) override {
        if (!coreml_requested) {
            return UNetModelRunner::compute(n_threads, params);
        }
#ifdef SD_USE_COREML
        const auto* extra = std::get_if<UNetDiffusionExtra>(&params.extra);
        if (!coreml) {
            return {};
        }
        // The CoreML graph is fixed: no LoRA, circular padding, ControlNet, IP-Adapter or c_concat.
        if (weight_adapter || circular_x_enabled || circular_y_enabled ||
            (params.c_concat && !params.c_concat->empty()) ||
            (extra && ((extra->controls && !extra->controls->empty()) || (extra->ip_context && !extra->ip_context->empty())))) {
            LOG_ERROR("coreml unet: LoRA, circular padding, ControlNet, IP-Adapter and c_concat are not supported");
            return {};
        }
        // y is SDXL's vector conditioning, [A] or [A, N] in ggml order.
        const bool has_y = params.y && !params.y->empty();
        if (has_y != (coreml->y_dim() > 0) || (has_y && params.y->shape()[0] != coreml->y_dim())) {
            LOG_ERROR("coreml unet: the model takes %lld values of vector conditioning (y), the pipeline gives %lld",
                      (long long)coreml->y_dim(), has_y ? (long long)params.y->shape()[0] : 0LL);
            return {};
        }
        if (!params.x || params.x->dim() < 3 || params.x->dim() > 4 || !params.timesteps || params.timesteps->empty() ||
            !params.context || params.context->dim() < 2 || params.context->dim() > 3) {
            LOG_ERROR("coreml unet: expects a [W, H, C, N] latent, timesteps and a [D, T, N] context");
            return {};
        }
        // sd::Tensor shapes are in ggml order: [W, H, C, N] is CoreML's [N, C, H, W], [D, T, N] is [N, T, D].
        const auto& xs         = params.x->shape();
        const auto& cs         = params.context->shape();
        const int64_t x_shape[4]   = {params.x->dim() == 4 ? xs[3] : 1, xs[2], xs[1], xs[0]};
        const int64_t ctx_shape[3] = {params.context->dim() > 2 ? cs[2] : 1, cs[1], cs[0]};
        sd::Tensor<float> output(xs);
        if (measure_mode_enabled()) {
            return output;
        }
        if (!coreml->run(params.x->data(), x_shape, params.timesteps->data(), params.timesteps->numel(),
                         params.context->data(), ctx_shape,
                         has_y ? params.y->data() : nullptr, has_y ? params.y->numel() / coreml->y_dim() : 0, coreml->y_dim(),
                         output.data())) {
            return {};
        }
        return output;
#else
        SD_UNUSED(n_threads);
        return {};
#endif
    }
};

#endif  // __SD_MODEL_DIFFUSION_COREML_UNET_RUNNER_H__
