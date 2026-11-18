#ifndef __SD_MODEL_DIFFUSION_H3_CACHE_GRAPH_H__
#define __SD_MODEL_DIFFUSION_H3_CACHE_GRAPH_H__

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/diffusion/model.hpp"
#include "runtime/h3_teacache.h"

namespace MiniMaxH3 {

    inline bool direct_backend_supported(ggml_backend_t backend) {
        if (!backend)
            return false;
        auto device = ggml_backend_get_device(backend);
        if (!device)
            return false;
        using Query   = bool (*)(ggml_backend_t);
        auto registry = ggml_backend_dev_backend_reg(device);
        auto query    = reinterpret_cast<Query>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_cuda_h3_direct_supported"));
        return query && query(backend);
    }

    struct H3GpuTensor {
        ggml_context* ctx            = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* tensor          = nullptr;
        ~H3GpuTensor() {
            if (buffer)
                ggml_backend_buffer_free(buffer);
            if (ctx)
                ggml_free(ctx);
        }
    };

    struct H3GpuCacheStore {
        const void* owner      = nullptr;
        ggml_backend_t backend = nullptr;
        std::map<std::string, std::unique_ptr<H3GpuTensor>> tensors;
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pending;
        ~H3GpuCacheStore() {
            if (backend)
                ggml_backend_synchronize(backend);
        }

        ggml_tensor* get(const std::string& name) {
            const auto found = tensors.find(name);
            return found == tensors.end() ? nullptr : found->second->tensor;
        }

        ggml_tensor* capture(const std::string& name, ggml_tensor* source) {
            auto* target = get(name);
            if (target && (target->type != source->type || !ggml_are_same_shape(target, source))) {
                ggml_backend_synchronize(backend);
                tensors.erase(name);
                target = nullptr;
            }
            if (!target) {
                auto allocation = std::make_unique<H3GpuTensor>();
                allocation->ctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
                if (!allocation->ctx)
                    return nullptr;
                allocation->tensor = ggml_dup_tensor(allocation->ctx, source);
                allocation->buffer = ggml_backend_alloc_ctx_tensors(allocation->ctx, backend);
                if (!allocation->buffer)
                    return nullptr;
                target = allocation->tensor;
                tensors.emplace(name, std::move(allocation));
            }
            ggml_set_output(source);
            pending.emplace_back(source, target);
            return source;
        }

        bool commit() {
            for (const auto& copy : pending) {
                if (!copy.first->buffer || !copy.first->data)
                    return false;
                ggml_backend_tensor_copy(copy.first, copy.second);
            }
            ggml_backend_synchronize(backend);
            pending.clear();
            return true;
        }
    };

    struct H3CacheGraph {
        enum class Phase { PROBE,
                           FULL,
                           DENSE,
                           REUSE };
        Phase phase                            = Phase::FULL;
        H3GpuCacheStore* store                 = nullptr;
        sd_sample::H3CacheEntry* entry         = nullptr;
        const sd_sample::H3CacheConfig* config = nullptr;
        std::string prefix;
        ggml_tensor* cached_output   = nullptr;
        bool failed                  = false;
    };

    inline uint64_t cache_signature(const DiffusionParams& params) {
        uint64_t hash = 1469598103934665603ULL;
        auto bytes    = [&](const void* data, size_t count) {
            const auto* p = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < count; ++i)
                hash = (hash ^ p[i]) * 1099511628211ULL;
        };
        auto tensor = [&](const auto* value, bool include_data = true) {
            if (!value) {
                const int absent = -1;
                bytes(&absent, sizeof(absent));
                return;
            }
            const auto shape  = value->shape();
            const size_t rank = shape.size();
            bytes(&rank, sizeof(rank));
            for (auto dimension : shape)
                bytes(&dimension, sizeof(dimension));
            if (include_data && !value->empty())
                bytes(value->data(), value->numel() * sizeof(*value->data()));
        };
        tensor(params.x, false);
        tensor(params.context);
        tensor(params.c_concat);
        tensor(params.y);
        if (params.ref_latents)
            for (const auto& ref : *params.ref_latents)
                tensor(&ref);
        if (const auto* extra = diffusion_extra_as<MiniMaxH3DiffusionExtra>(params)) {
            tensor(extra->text_token_tags);
            tensor(extra->keyframe_indices);
            if (extra->reference_audio_latents)
                for (const auto& ref : *extra->reference_audio_latents)
                    tensor(&ref);
            if (extra->reference_blocks)
                for (const auto& ref : *extra->reference_blocks) {
                    bytes(&ref.kind, sizeof(ref.kind));
                    bytes(&ref.video_index, sizeof(ref.video_index));
                    bytes(&ref.audio_index, sizeof(ref.audio_index));
                }
            bytes(&extra->audio_length, sizeof(extra->audio_length));
            bytes(&extra->video_sigma_shift, sizeof(extra->video_sigma_shift));
            bytes(&extra->audio_sigma_shift, sizeof(extra->audio_sigma_shift));
        }
        return hash;
    }

}  // namespace MiniMaxH3
#endif
