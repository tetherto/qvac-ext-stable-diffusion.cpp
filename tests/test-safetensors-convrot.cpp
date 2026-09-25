#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model_io/binary_io.h"
#include "model_loader.h"

namespace {

std::string make_header(const std::string& marker) {
    return "{\"layer.weight\":{\"dtype\":\"I8\",\"shape\":[4,256],\"data_offsets\":[0,1024]},"
           "\"layer.weight_scale\":{\"dtype\":\"F32\",\"shape\":[4,1],\"data_offsets\":[1024,1040]},"
           "\"layer.comfy_quant\":{\"dtype\":\"U8\",\"shape\":[" +
           std::to_string(marker.size()) + "],\"data_offsets\":[1040," +
           std::to_string(1040 + marker.size()) + "]}}";
}

void write_fixture(const std::filesystem::path& path, const std::string& marker, float scale_value = 0.5f) {
    const std::string header = make_header(marker);
    std::vector<int8_t> weights(4 * 256, 0);
    weights[0] = 2;
    const float scales[] = {scale_value, scale_value, scale_value, scale_value};

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    GGML_ASSERT(file.is_open());
    model_io::write_u64(file, header.size());
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size()));
    file.write(reinterpret_cast<const char*>(scales), sizeof(scales));
    file.write(marker.data(), static_cast<std::streamsize>(marker.size()));
    GGML_ASSERT(file.good());
}

const TensorStorage& find_tensor(const ModelLoader& loader, const std::string& name) {
    const auto& tensors = loader.get_tensor_storage_map();
    const auto it = tensors.find(name);
    GGML_ASSERT(it != tensors.end());
    return it->second;
}

int set_test_environment(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value);
#else
    return setenv(name, value, 1);
#endif
}

int unset_test_environment(const char* name) {
#ifdef _WIN32
    return _putenv_s(name, "");
#else
    return unsetenv(name);
#endif
}

ggml_backend_t init_cpu_backend() {
    ggml_backend_load_all();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

class InspectableLinear : public Linear {
public:
    using Linear::Linear;
    bool fast_h256_enabled() const { return use_convrot_fast_h256; }
    bool rotation_op_enabled() const { return use_convrot_rotation_op; }
};

}  // namespace

int main() {
    GGML_ASSERT(select_convrot_execution_path("CUDA0", true, nullptr) == ConvRotExecutionPath::DENSE_H256);
    GGML_ASSERT(select_convrot_execution_path("Vulkan0", true, nullptr) == ConvRotExecutionPath::ROTATION_OP);
    GGML_ASSERT(select_convrot_execution_path("MTL0", true, nullptr) == ConvRotExecutionPath::ROTATION_OP);
    GGML_ASSERT(select_convrot_execution_path("CPU", true, nullptr) == ConvRotExecutionPath::ROTATION_OP);
    GGML_ASSERT(select_convrot_execution_path("unknown", false, nullptr) == ConvRotExecutionPath::NATIVE);
    GGML_ASSERT(select_convrot_execution_path("CUDA0", true, "native") == ConvRotExecutionPath::NATIVE);
    GGML_ASSERT(select_convrot_execution_path("CUDA0", true, "dense") == ConvRotExecutionPath::DENSE_H256);
    GGML_ASSERT(select_convrot_execution_path("Vulkan0", true, "op") == ConvRotExecutionPath::ROTATION_OP);
    GGML_ASSERT(select_convrot_execution_path("Vulkan0", false, "op") == ConvRotExecutionPath::NATIVE);
    GGML_ASSERT(select_convrot_execution_path("CUDA0", true, "compat") == ConvRotExecutionPath::COMPAT);

    const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                       "stable-diffusion-convrot-test.safetensors";
    const std::string marker =
        "{\"format\":\"int8_tensorwise\",\"convrot\":true,\"convrot_groupsize\":256}";
    write_fixture(path, marker);

    ModelLoader loader;
    GGML_ASSERT(loader.init_from_file(path.string()));
    const TensorStorage& weight = find_tensor(loader, "layer.weight");
    GGML_ASSERT(weight.type == GGML_TYPE_I8);
    GGML_ASSERT(weight.expected_type == GGML_TYPE_F16);
    GGML_ASSERT(weight.is_comfy_int8_tensorwise);
    GGML_ASSERT(weight.comfy_int8_convrot);
    GGML_ASSERT(weight.comfy_int8_group_size == 256);
    GGML_ASSERT(weight.has_comfy_int8_scale());
    GGML_ASSERT(weight.comfy_int8_scale.name == "layer.weight_scale");
    GGML_ASSERT(weight.comfy_int8_scale.type == GGML_TYPE_F32);
    GGML_ASSERT(weight.comfy_int8_scale.n_dims == 2);
    GGML_ASSERT(weight.comfy_int8_scale.ne[0] == 1);
    GGML_ASSERT(weight.comfy_int8_scale.ne[1] == 4);
    GGML_ASSERT(weight.comfy_int8_scale.nbytes == 4 * sizeof(float));
    GGML_ASSERT(weight.is_comfy_int8_convrot_weight());
    GGML_ASSERT(loader.get_tensor_storage_map().find("layer.weight_scale") == loader.get_tensor_storage_map().end());

    ggml_init_params params = {4096, nullptr, false};
    ggml_context* ctx       = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor* decoded = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 4);
    GGML_ASSERT(loader.load_tensor(weight, decoded));

    std::vector<float> values(4 * 256);
    ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(decoded->data), values.data(), values.size());
    float energy = 0.f;
    for (float value : values) {
        energy += value * value;
    }
    // The normalized regular Hadamard matrix is orthogonal: ConvRot inversion
    // preserves the squared norm of the dequantized first row.
    GGML_ASSERT(std::fabs(energy - 1.f) < 0.002f);
    ggml_free(ctx);

    ggml_init_params native_params = {ggml_tensor_overhead() * 2 + 1024 + 4 * sizeof(float) + 4096, nullptr, false};
    ggml_context* native_ctx       = ggml_init(native_params);
    GGML_ASSERT(native_ctx != nullptr);
    ggml_tensor* raw_weight = ggml_new_tensor_2d(native_ctx, GGML_TYPE_I8, 256, 4);
    ggml_tensor* raw_scale  = ggml_new_tensor_1d(native_ctx, GGML_TYPE_F32, 4);
    GGML_ASSERT(loader.load_comfy_int8_tensorwise(weight, raw_weight, raw_scale));
    GGML_ASSERT(static_cast<const int8_t*>(raw_weight->data)[0] == 2);
    GGML_ASSERT(static_cast<const int8_t*>(raw_weight->data)[1] == 0);
    GGML_ASSERT(static_cast<const float*>(raw_scale->data)[0] == 0.5f);
    GGML_ASSERT(static_cast<const float*>(raw_scale->data)[3] == 0.5f);
    ggml_free(native_ctx);

    const size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_0, 256) * 4;
    ggml_init_params q8_params = {ggml_tensor_overhead() + q8_bytes + 4096, nullptr, false};
    ggml_context* q8_ctx       = ggml_init(q8_params);
    GGML_ASSERT(q8_ctx != nullptr);
    ggml_tensor* q8_weight = ggml_new_tensor_2d(q8_ctx, GGML_TYPE_Q8_0, 256, 4);
    GGML_ASSERT(loader.load_comfy_int8_tensorwise(weight, q8_weight, nullptr));
    ggml_fp16_t packed_scale;
    memcpy(&packed_scale, q8_weight->data, sizeof(packed_scale));
    GGML_ASSERT(ggml_fp16_to_fp32(packed_scale) == 0.5f);
    GGML_ASSERT(static_cast<const int8_t*>(q8_weight->data)[sizeof(packed_scale)] == 2);
    GGML_ASSERT(static_cast<const int8_t*>(q8_weight->data)[sizeof(packed_scale) + 1] == 0);
    ggml_free(q8_ctx);

    // FP16-subnormal scales are reported but remain usable; scales that would
    // underflow to zero must not silently produce an all-zero Q8_0 row.
    for (float scale : {1e-6f, 1e-9f}) {
        write_fixture(path, marker, scale);
        ModelLoader range_loader;
        GGML_ASSERT(range_loader.init_from_file(path.string()));
        const TensorStorage& range_weight = find_tensor(range_loader, "layer.weight");
        ggml_init_params range_params = {ggml_tensor_overhead() + q8_bytes + 4096, nullptr, false};
        ggml_context* range_ctx = ggml_init(range_params);
        GGML_ASSERT(range_ctx != nullptr);
        ggml_tensor* range_q8 = ggml_new_tensor_2d(range_ctx, GGML_TYPE_Q8_0, 256, 4);
        GGML_ASSERT(range_loader.load_comfy_int8_tensorwise(range_weight, range_q8, nullptr) == (scale == 1e-6f));
        if (scale == 1e-6f) {
            ggml_fp16_t range_half;
            memcpy(&range_half, range_q8->data, sizeof(range_half));
            GGML_ASSERT(ggml_fp16_to_fp32(range_half) > 0.0f);
        }
        ggml_free(range_ctx);
    }
    write_fixture(path, marker);

    ggml_backend_t cpu_backend = init_cpu_backend();
    GGML_ASSERT(cpu_backend != nullptr);
    GGML_ASSERT(set_test_environment("SD_CONVROT_MODE", "native") == 0);
    const auto native_selection = select_convrot_tensor_storage(cpu_backend,
                                                                 loader.get_tensor_storage_map(),
                                                                 "ConvRot loader test");
    GGML_ASSERT(native_selection.at("layer.weight").comfy_int8_native_enabled);
    GGML_ASSERT(!native_selection.at("layer.weight").comfy_int8_q8_decomp_enabled);
    GGML_ASSERT(ggml_backend_supports_convrot_op(cpu_backend));
    GGML_ASSERT(unset_test_environment("SD_CONVROT_MODE") == 0);

    const auto automatic_selection = select_convrot_tensor_storage(cpu_backend,
                                                                    loader.get_tensor_storage_map(),
                                                                    "ConvRot loader test");
    GGML_ASSERT(automatic_selection.at("layer.weight").comfy_int8_q8_decomp_enabled);
    GGML_ASSERT(automatic_selection.at("layer.weight").comfy_int8_convrot_op_enabled);

    GGML_ASSERT(set_test_environment("SD_CONVROT_MODE", "q8") == 0);
    const auto q8_selection = select_convrot_tensor_storage(cpu_backend,
                                                             loader.get_tensor_storage_map(),
                                                             "ConvRot loader test");
    GGML_ASSERT(q8_selection.at("layer.weight").comfy_int8_native_enabled);
    GGML_ASSERT(q8_selection.at("layer.weight").comfy_int8_q8_decomp_enabled);
    GGML_ASSERT(q8_selection.at("layer.weight").comfy_int8_convrot_op_enabled);
    GGML_ASSERT(unset_test_environment("SD_CONVROT_MODE") == 0);

    ggml_init_params op_graph_params = {1024 * 1024, nullptr, true};
    ggml_context* op_graph_ctx       = ggml_init(op_graph_params);
    GGML_ASSERT(op_graph_ctx != nullptr);
    InspectableLinear op_linear(256, 4, false);
    op_linear.init(op_graph_ctx, automatic_selection, "layer");
    GGML_ASSERT(op_linear.rotation_op_enabled());
    ggml_tensor* op_input = ggml_new_tensor_2d(op_graph_ctx, GGML_TYPE_F32, 256, 2);
    GGMLRunnerContext op_runner_ctx;
    op_runner_ctx.ggml_ctx = op_graph_ctx;
    ggml_tensor* op_output = op_linear.forward(&op_runner_ctx, op_input);
    GGML_ASSERT(op_output != nullptr && op_output->op == GGML_OP_MUL_MAT);
    GGML_ASSERT(op_output->src[1] != nullptr && op_output->src[1]->op == GGML_OP_CONVROT);
    ggml_free(op_graph_ctx);

    GGML_ASSERT(set_test_environment("SD_CONVROT_MODE", "dense") == 0);
    const auto dense_selection = select_convrot_tensor_storage(cpu_backend,
                                                                loader.get_tensor_storage_map(),
                                                                "ConvRot loader test");
    GGML_ASSERT(dense_selection.at("layer.weight").comfy_int8_q8_decomp_enabled);
    GGML_ASSERT(!dense_selection.at("layer.weight").comfy_int8_convrot_op_enabled);
    GGML_ASSERT(unset_test_environment("SD_CONVROT_MODE") == 0);

    auto h256_hint_for_mode = [&](const char* mode) {
        if (mode == nullptr) {
            GGML_ASSERT(unset_test_environment("SD_CONVROT_H256_MODE") == 0);
        } else {
            GGML_ASSERT(set_test_environment("SD_CONVROT_H256_MODE", mode) == 0);
        }
        ggml_init_params graph_params = {1024 * 1024, nullptr, true};
        ggml_context* graph_ctx       = ggml_init(graph_params);
        GGML_ASSERT(graph_ctx != nullptr);
        InspectableLinear linear(256, 4, false);
        linear.init(graph_ctx, dense_selection, "layer");
        GGML_ASSERT(linear.fast_h256_enabled() == (mode != nullptr && strcmp(mode, "fast") == 0));
        ggml_tensor* input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 256, 2);
        GGMLRunnerContext runner_ctx;
        runner_ctx.ggml_ctx = graph_ctx;
        ggml_tensor* output = linear.forward(&runner_ctx, input);
        GGML_ASSERT(output != nullptr && output->op == GGML_OP_MUL_MAT);
        ggml_tensor* rotated = output->src[1];
        if (rotated->op != GGML_OP_MUL_MAT) {
            rotated = rotated->src[0];
        }
        GGML_ASSERT(rotated != nullptr && rotated->op == GGML_OP_MUL_MAT);
        int32_t hint = GGML_HINT_NONE;
        memcpy(&hint, rotated->op_params + sizeof(int32_t), sizeof(hint));
        ggml_free(graph_ctx);
        return hint;
    };
    GGML_ASSERT(h256_hint_for_mode(nullptr) == GGML_HINT_NONE);
    GGML_ASSERT(h256_hint_for_mode("dense") == GGML_HINT_NONE);
    // The accessor assertion in h256_hint_for_mode verifies that "fast"
    // selects the opt-in path. The backend-specific hint itself is covered by
    // test-mul-mat-convrot-h256-hint.
    (void) h256_hint_for_mode("fast");
    GGML_ASSERT(unset_test_environment("SD_CONVROT_H256_MODE") == 0);

    // A runner for another component must not inherit this component's
    // ConvRot requirement when both live in the shared storage map.
    const auto other_component_selection = select_convrot_tensor_storage(nullptr,
                                                                          loader.get_tensor_storage_map(),
                                                                          "unrelated component",
                                                                          "other_component.");
    GGML_ASSERT(!other_component_selection.at("layer.weight").comfy_int8_native_enabled);

    GGML_ASSERT(set_test_environment("SD_CONVROT_MODE", "compat") == 0);
    const auto compatibility_selection = select_convrot_tensor_storage(cpu_backend,
                                                                        loader.get_tensor_storage_map(),
                                                                        "ConvRot loader test");
    GGML_ASSERT(!compatibility_selection.at("layer.weight").comfy_int8_native_enabled);
    GGML_ASSERT(unset_test_environment("SD_CONVROT_MODE") == 0);
    ggml_backend_free(cpu_backend);

    write_fixture(path, marker, std::numeric_limits<float>::quiet_NaN());
    ModelLoader invalid_scale_loader;
    GGML_ASSERT(invalid_scale_loader.init_from_file(path.string()));
    const TensorStorage& invalid_scale_weight = find_tensor(invalid_scale_loader, "layer.weight");
    ggml_init_params invalid_scale_params = {ggml_tensor_overhead() * 2 + 1024 + 4 * sizeof(float) + 4096, nullptr, false};
    ggml_context* invalid_scale_ctx       = ggml_init(invalid_scale_params);
    GGML_ASSERT(invalid_scale_ctx != nullptr);
    ggml_tensor* invalid_raw_weight = ggml_new_tensor_2d(invalid_scale_ctx, GGML_TYPE_I8, 256, 4);
    ggml_tensor* invalid_raw_scale  = ggml_new_tensor_1d(invalid_scale_ctx, GGML_TYPE_F32, 4);
    GGML_ASSERT(!invalid_scale_loader.load_comfy_int8_tensorwise(invalid_scale_weight, invalid_raw_weight, invalid_raw_scale));
    ggml_free(invalid_scale_ctx);

    const std::string unsupported_group =
        "{\"format\":\"int8_tensorwise\",\"convrot\":true,\"convrot_groupsize\":16}";
    write_fixture(path, unsupported_group);
    ModelLoader invalid_loader;
    GGML_ASSERT(!invalid_loader.init_from_file(path.string()));

    write_fixture(path, "not-json");
    ModelLoader malformed_loader;
    GGML_ASSERT(!malformed_loader.init_from_file(path.string()));

    if (const char* real_model_path = std::getenv("CONVROT_MODEL_PATH")) {
        ModelLoader real_loader;
        GGML_ASSERT(real_loader.init_from_file(real_model_path));
        const TensorStorage* real_weight = nullptr;
        for (const auto& [_, tensor] : real_loader.get_tensor_storage_map()) {
            if (tensor.is_comfy_int8_tensorwise) {
                real_weight = &tensor;
                break;
            }
        }
        GGML_ASSERT(real_weight != nullptr);
        const size_t elements = static_cast<size_t>(real_weight->ne[0]) *
                                static_cast<size_t>(real_weight->ne[1]);
        ggml_init_params real_params = {ggml_tensor_overhead() + elements * sizeof(ggml_fp16_t) + 4096,
                                        nullptr,
                                        false};
        ggml_context* real_ctx = ggml_init(real_params);
        GGML_ASSERT(real_ctx != nullptr);
        ggml_tensor* real_decoded = ggml_new_tensor_2d(real_ctx,
                                                        GGML_TYPE_F16,
                                                        real_weight->ne[0],
                                                        real_weight->ne[1]);
        GGML_ASSERT(real_loader.load_tensor(*real_weight, real_decoded));
        const float first_value = ggml_fp16_to_fp32(static_cast<ggml_fp16_t*>(real_decoded->data)[0]);
        GGML_ASSERT(std::isfinite(first_value));
        ggml_free(real_ctx);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}
