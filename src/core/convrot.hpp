#ifndef __SD_CORE_CONVROT_H__
#define __SD_CORE_CONVROT_H__
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "core/util.h"
#include "ggml-backend.h"
#include "model.h"

// Construct only the operation metadata needed for the normal backend
// supports_op query.  This stays private to the loader policy: backend
// capabilities are expressed through the existing ggml interface, not a new
// public ConvRot-specific API.
inline bool ggml_backend_supports_convrot_op(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    std::vector<uint8_t> storage(4 * ggml_tensor_overhead() + 1024);
    ggml_init_params params = {
        /*.mem_size   =*/storage.size(),
        /*.mem_buffer =*/storage.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor* activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor* weights     = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 256, 1);
    ggml_tensor* scales      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor* op          = ggml_mul_mat_convrot(ctx, activations, weights, scales, 256);
    const bool supported     = ggml_backend_supports_op(backend, op);
    ggml_free(ctx);
    return supported;
}

inline bool ggml_backend_supports_convrot_rotation_op(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    std::vector<uint8_t> storage(3 * ggml_tensor_overhead() + 1024);
    ggml_init_params params = {
        /*.mem_size   =*/storage.size(),
        /*.mem_buffer =*/storage.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor* activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor* op          = ggml_convrot(ctx, activations, 256);
    const bool supported     = ggml_backend_supports_op(backend, op);
    ggml_free(ctx);
    return supported;
}

enum class ConvRotExecutionPath {
    NATIVE,
    DENSE_H256,
    ROTATION_OP,
    COMPAT,
};

inline ConvRotExecutionPath select_convrot_execution_path(const char* backend_name,
                                                          bool rotation_op_supported,
                                                          const char* mode) {
    if (mode != nullptr && std::strcmp(mode, "compat") == 0) {
        return ConvRotExecutionPath::COMPAT;
    }
    if (mode != nullptr && std::strcmp(mode, "native") == 0) {
        return ConvRotExecutionPath::NATIVE;
    }
    if (mode != nullptr && std::strcmp(mode, "dense") == 0) {
        return ConvRotExecutionPath::DENSE_H256;
    }
    if (mode != nullptr && std::strcmp(mode, "op") == 0) {
        return rotation_op_supported ? ConvRotExecutionPath::ROTATION_OP
                                     : ConvRotExecutionPath::NATIVE;
    }
    if (mode != nullptr && std::strcmp(mode, "q8") != 0 && std::strcmp(mode, "auto") != 0) {
        throw std::runtime_error("invalid SD_CONVROT_MODE; expected 'auto', 'native', 'q8', 'dense', 'op', or 'compat'");
    }

    // The CUDA radix-4 transform is mathematically correct, but the H3 text
    // encoder amplifies its rounding difference enough to change the scene.
    // CUDA therefore keeps the numerically stable dense transform. Other
    // backends use the standalone transform when they implement it.
    if (backend_name != nullptr && std::strncmp(backend_name, "CUDA", 4) == 0) {
        return ConvRotExecutionPath::DENSE_H256;
    }
    return rotation_op_supported ? ConvRotExecutionPath::ROTATION_OP
                                 : ConvRotExecutionPath::NATIVE;
}

// Select the compact representation before any model parameter tensor is
// created. SD_CONVROT_MODE can override the automatic per-backend policy.
inline String2TensorStorage select_convrot_tensor_storage(ggml_backend_t backend,
                                                          const String2TensorStorage& source,
                                                          const std::string& component,
                                                          const std::string& prefix = "") {
    // OrderedMap's default copy also copies its iterator index; rebuild it so
    // this independent policy view owns a valid index into its own list.
    String2TensorStorage selected;
    bool has_convrot = false;
    for (const auto& [name, storage] : source) {
        selected.insert({name, storage});
        has_convrot = has_convrot ||
                      (name.rfind(prefix, 0) == 0 && storage.is_comfy_int8_convrot_weight());
    }
    if (!has_convrot) {
        return selected;
    }

    const char* mode                 = std::getenv("SD_CONVROT_MODE");
    const char* backend_name         = backend != nullptr ? ggml_backend_name(backend) : "unknown";
    const bool rotation_op_supported = ggml_backend_supports_convrot_rotation_op(backend);
    const ConvRotExecutionPath path  = select_convrot_execution_path(backend_name, rotation_op_supported, mode);
    if (path == ConvRotExecutionPath::COMPAT) {
        LOG_INFO("ConvRot: using explicitly selected F16 compatibility path for %s on backend %s",
                 component.c_str(), backend_name);
        return selected;
    }
    if (path == ConvRotExecutionPath::DENSE_H256 || path == ConvRotExecutionPath::ROTATION_OP) {
        for (auto& [name, storage] : selected) {
            if (name.rfind(prefix, 0) == 0 && storage.is_comfy_int8_convrot_weight()) {
                storage.comfy_int8_native_enabled     = true;
                storage.comfy_int8_q8_decomp_enabled  = true;
                storage.comfy_int8_convrot_op_enabled = path == ConvRotExecutionPath::ROTATION_OP;
            }
        }
        LOG_INFO("ConvRot: selected Q8_0 %s activation transform for %s on backend %s",
                 path == ConvRotExecutionPath::ROTATION_OP ? "standalone" : "dense-H256",
                 component.c_str(), backend_name);
        return selected;
    }
    if (mode != nullptr && std::strcmp(mode, "op") == 0 && !rotation_op_supported) {
        LOG_WARN("ConvRot: standalone rotation is unavailable on backend %s; falling back to the native operation for %s",
                 backend_name, component.c_str());
    }
    if (!ggml_backend_supports_convrot_op(backend)) {
        throw std::runtime_error("ConvRot native support is required for " + component +
                                 " but backend '" + backend_name +
                                 "' lacks the 256-wide I8/F32 ConvRot operation; use a capable backend or set "
                                 "SD_CONVROT_MODE=compat to select the F16 compatibility path");
    }
    for (auto& [name, storage] : selected) {
        if (name.rfind(prefix, 0) == 0 && storage.is_comfy_int8_convrot_weight()) {
            storage.comfy_int8_native_enabled = true;
        }
    }
    LOG_INFO("ConvRot: selected native compact I8/F32 path for %s on backend %s",
             component.c_str(), backend_name);
    return selected;
}

#endif
