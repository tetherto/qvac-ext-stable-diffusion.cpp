#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/util.h"
#include "ggml-cpu.h"
#include "model_io/binary_io.h"
#include "model_io/safetensors_io.h"
#include "model_loader.h"

namespace {

    std::string make_header(const std::string& marker, size_t columns = 256) {
        const size_t weight_bytes = 4 * columns;
        return "{\"layer.weight\":{\"dtype\":\"I8\",\"shape\":[4," + std::to_string(columns) +
               "],\"data_offsets\":[0," + std::to_string(weight_bytes) + "]},"
               "\"layer.weight_scale\":{\"dtype\":\"F32\",\"shape\":[4,1],\"data_offsets\":[" +
               std::to_string(weight_bytes) + "," + std::to_string(weight_bytes + 16) + "]},"
               "\"layer.comfy_quant\":{\"dtype\":\"U8\",\"shape\":[" +
               std::to_string(marker.size()) + "],\"data_offsets\":[" +
               std::to_string(weight_bytes + 16) + "," +
               std::to_string(weight_bytes + 16 + marker.size()) + "]}}";
    }

    void write_fixture(const std::filesystem::path& path, const std::string& marker, float scale_value = 0.5f,
                       size_t columns = 256) {
        const std::string header = make_header(marker, columns);
        std::vector<int8_t> weights(4 * columns, 0);
        weights[0]           = 2;
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

    void write_markerless_fixture(const std::filesystem::path& path) {
        const std::string header =
            "{\"layer.weight\":{\"dtype\":\"I8\",\"shape\":[4,256],\"data_offsets\":[0,1024]},"
            "\"layer.weight_scale\":{\"dtype\":\"F32\",\"shape\":[4,1],\"data_offsets\":[1024,1040]}}";
        const std::vector<int8_t> weights(1024, 0);
        const float scales[] = {0.5f, 0.5f, 0.5f, 0.5f};
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        GGML_ASSERT(file.is_open());
        model_io::write_u64(file, header.size());
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size()));
        file.write(reinterpret_cast<const char*>(scales), sizeof(scales));
        GGML_ASSERT(file.good());
    }

    void write_marker_first_fixture(const std::filesystem::path& path, const std::string& marker, bool payload) {
        const size_t marker_end = marker.size();
        const std::string header =
            "{\"layer.comfy_quant\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(marker_end) +
            "],\"data_offsets\":[0," + std::to_string(marker_end) +
            "]},"
            "\"layer.weight\":{\"dtype\":\"I8\",\"shape\":[4,256],\"data_offsets\":[" +
            std::to_string(marker_end) + "," + std::to_string(marker_end + 1024) +
            "]},"
            "\"layer.weight_scale\":{\"dtype\":\"F32\",\"shape\":[4,1],\"data_offsets\":[" +
            std::to_string(marker_end + 1024) + "," + std::to_string(marker_end + 1040) + "]}}";
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        GGML_ASSERT(file.is_open());
        model_io::write_u64(file, header.size());
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(marker.data(), static_cast<std::streamsize>(marker.size()));
        if (payload) {
            const std::vector<int8_t> weights(1024, 0);
            const float scales[] = {0.5f, 0.5f, 0.5f, 0.5f};
            file.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size()));
            file.write(reinterpret_cast<const char*>(scales), sizeof(scales));
        }
        GGML_ASSERT(file.good());
    }

    const TensorStorage& find_tensor(const ModelLoader& loader, const std::string& name) {
        const auto& tensors = loader.get_tensor_storage_map();
        const auto it       = tensors.find(name);
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
        ggml_type weight_type() const { return params.at("weight")->type; }
        ggml_tensor* parameter(const std::string& name) const { return params.at(name); }
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
    bool invalid_mode_rejected = false;
    try {
        (void)select_convrot_execution_path("CUDA0", true, "invalid");
    } catch (const std::runtime_error&) {
        invalid_mode_rejected = true;
    }
    GGML_ASSERT(invalid_mode_rejected);

    const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                       "stable-diffusion-convrot-test.safetensors";
    const std::string marker =
        "{\"format\":\"int8_tensorwise\",\"convrot\":true,\"convrot_groupsize\":256}";
    write_fixture(path, marker);

    const std::filesystem::path sparse_path = std::filesystem::temp_directory_path() /
                                              "stable-diffusion-convrot-sparse-test.safetensors";
    write_marker_first_fixture(sparse_path, marker, true);
    std::vector<TensorStorage> full_tensors;
    GGML_ASSERT(read_safetensors_file(sparse_path.string(), full_tensors));
    write_marker_first_fixture(sparse_path, marker, false);
    std::vector<TensorStorage> metadata_tensors;
    std::string sparse_error;
    GGML_ASSERT(!read_safetensors_file(sparse_path.string(), metadata_tensors, &sparse_error));
    {
        SDMetadataOnlyReadScope scope;
        GGML_ASSERT(read_safetensors_file(sparse_path.string(), metadata_tensors, &sparse_error));
        GGML_ASSERT(metadata_tensors.size() == full_tensors.size());
        GGML_ASSERT(metadata_tensors[0].to_string() == full_tensors[0].to_string());
        GGML_ASSERT(metadata_tensors[0].is_comfy_int8_convrot_weight());
    }
    write_marker_first_fixture(sparse_path, marker, true);
    std::filesystem::resize_file(sparse_path, std::filesystem::file_size(sparse_path) - 1);
    GGML_ASSERT(!read_safetensors_file(sparse_path.string(), metadata_tensors, &sparse_error));
    // Restore the header and remove only the marker bytes.
    write_marker_first_fixture(sparse_path, marker, false);
    std::filesystem::resize_file(sparse_path, std::filesystem::file_size(sparse_path) - marker.size());
    {
        SDMetadataOnlyReadScope scope;
        GGML_ASSERT(!read_safetensors_file(sparse_path.string(), metadata_tensors, &sparse_error));
        GGML_ASSERT(sparse_error.find("marker") != std::string::npos);
    }
    std::error_code sparse_remove_error;
    std::filesystem::remove(sparse_path, sparse_remove_error);

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
    loader.set_wtype_override(GGML_TYPE_Q8_0, "");
    GGML_ASSERT(find_tensor(loader, "layer.weight").expected_type == GGML_TYPE_F16);

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

    const size_t q8_bytes      = ggml_row_size(GGML_TYPE_Q8_0, 256) * 4;
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
        ggml_init_params range_params     = {ggml_tensor_overhead() + q8_bytes + 4096, nullptr, false};
        ggml_context* range_ctx           = ggml_init(range_params);
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

    ggml_init_params precise_graph_params = {1024 * 1024, nullptr, true};
    ggml_context* precise_graph_ctx       = ggml_init(precise_graph_params);
    GGML_ASSERT(precise_graph_ctx != nullptr);
    InspectableLinear precise_linear(256, 4, false, false, true, 1.f / 128.f);
    precise_linear.init(precise_graph_ctx, automatic_selection, "layer");
    ggml_tensor* precise_input = ggml_new_tensor_2d(precise_graph_ctx, GGML_TYPE_F32, 256, 2);
    GGMLRunnerContext precise_runner_ctx;
    precise_runner_ctx.ggml_ctx = precise_graph_ctx;
    ggml_tensor* precise_output = precise_linear.forward(&precise_runner_ctx, precise_input);
    GGML_ASSERT(precise_output->op == GGML_OP_SCALE);
    ggml_tensor* precise_matmul = precise_output->src[0];
    GGML_ASSERT(precise_matmul != nullptr && precise_matmul->op == GGML_OP_MUL_MAT);
    int32_t precision = GGML_PREC_DEFAULT;
    memcpy(&precision, precise_matmul->op_params, sizeof(precision));
    GGML_ASSERT(precision == GGML_PREC_F32);
    GGML_ASSERT(precise_matmul->src[1]->op == GGML_OP_CONVROT);
    GGML_ASSERT(precise_matmul->src[1]->src[0]->op == GGML_OP_SCALE);
    GGML_ASSERT(precise_matmul->src[1]->src[0]->src[0] == precise_input);
    ggml_free(precise_graph_ctx);

    ggml_init_params forced_graph_params = {1024 * 1024, nullptr, true};
    ggml_context* forced_graph_ctx       = ggml_init(forced_graph_params);
    GGML_ASSERT(forced_graph_ctx != nullptr);
    InspectableLinear forced_linear(256, 4, false, true);
    forced_linear.init(forced_graph_ctx, automatic_selection, "layer");
    GGML_ASSERT(forced_linear.weight_type() == GGML_TYPE_F32);
    ggml_free(forced_graph_ctx);

    GGML_ASSERT(set_test_environment("SD_CONVROT_MODE", "dense") == 0);
    const auto dense_selection = select_convrot_tensor_storage(cpu_backend,
                                                               loader.get_tensor_storage_map(),
                                                               "ConvRot loader test");
    GGML_ASSERT(dense_selection.at("layer.weight").comfy_int8_q8_decomp_enabled);
    GGML_ASSERT(!dense_selection.at("layer.weight").comfy_int8_convrot_op_enabled);
    GGML_ASSERT(unset_test_environment("SD_CONVROT_MODE") == 0);

    ggml_init_params dense_precise_params = {1024 * 1024, nullptr, true};
    ggml_context* dense_precise_ctx       = ggml_init(dense_precise_params);
    GGML_ASSERT(dense_precise_ctx != nullptr);
    InspectableLinear dense_precise_linear(256, 4, false, false, true, 1.f / 128.f);
    dense_precise_linear.init(dense_precise_ctx, dense_selection, "layer");
    ggml_tensor* dense_precise_input = ggml_new_tensor_2d(dense_precise_ctx, GGML_TYPE_F32, 256, 2);
    GGMLRunnerContext dense_precise_runner;
    dense_precise_runner.ggml_ctx = dense_precise_ctx;
    ggml_tensor* dense_precise_output = dense_precise_linear.forward(&dense_precise_runner, dense_precise_input);
    GGML_ASSERT(dense_precise_output->op == GGML_OP_SCALE);
    ggml_tensor* dense_precise_matmul = dense_precise_output->src[0];
    GGML_ASSERT(dense_precise_matmul->op == GGML_OP_MUL_MAT);
    int32_t dense_precision = GGML_PREC_DEFAULT;
    memcpy(&dense_precision, dense_precise_matmul->op_params, sizeof(dense_precision));
    GGML_ASSERT(dense_precision == GGML_PREC_F32);
    ggml_tensor* dense_rotated = dense_precise_matmul->src[1];
    GGML_ASSERT(dense_rotated->op == GGML_OP_RESHAPE);
    GGML_ASSERT(dense_rotated->src[0]->op == GGML_OP_MUL_MAT);
    ggml_tensor* dense_blocks = dense_rotated->src[0]->src[1];
    GGML_ASSERT(dense_blocks->op == GGML_OP_RESHAPE);
    GGML_ASSERT(dense_blocks->src[0]->op == GGML_OP_SCALE);
    GGML_ASSERT(dense_blocks->src[0]->src[0] == dense_precise_input);
    ggml_free(dense_precise_ctx);

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
    (void)h256_hint_for_mode("fast");
    GGML_ASSERT(unset_test_environment("SD_CONVROT_H256_MODE") == 0);

    // A runner for another component must not inherit this component's
    // ConvRot requirement when both live in the shared storage map.
    const auto other_component_selection = select_convrot_tensor_storage(nullptr,
                                                                         loader.get_tensor_storage_map(),
                                                                         "unrelated component",
                                                                         "other_component.");
    GGML_ASSERT(!other_component_selection.at("layer.weight").comfy_int8_native_enabled);

    bool unsupported_backend_rejected = false;
    try {
        (void)select_convrot_tensor_storage(nullptr, loader.get_tensor_storage_map(), "ConvRot loader test");
    } catch (const std::runtime_error&) {
        unsupported_backend_rejected = true;
    }
    GGML_ASSERT(unsupported_backend_rejected);

    GGML_ASSERT(set_test_environment("SD_CONVROT_MODE", "compat") == 0);
    const auto compatibility_selection = select_convrot_tensor_storage(cpu_backend,
                                                                       loader.get_tensor_storage_map(),
                                                                       "ConvRot loader test");
    GGML_ASSERT(!compatibility_selection.at("layer.weight").comfy_int8_native_enabled);
    ggml_init_params compat_graph_params = {1024 * 1024, nullptr, true};
    ggml_context* compat_graph_ctx       = ggml_init(compat_graph_params);
    GGML_ASSERT(compat_graph_ctx != nullptr);
    InspectableLinear compat_linear(256, 4, false);
    compat_linear.init(compat_graph_ctx, compatibility_selection, "layer");
    GGML_ASSERT(compat_linear.weight_type() == GGML_TYPE_F16);
    ggml_free(compat_graph_ctx);

    const std::vector<float> bias_values = {0.25f, -0.5f, 1.0f, -1.25f};
    std::vector<float> input_values(256 * 2);
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = 0.5f + static_cast<float>(static_cast<int>((i * 17) % 29) - 14) * 0.125f;
    }
    auto run_linear = [&](const String2TensorStorage& selection) {
        ggml_init_params exec_params = {8 * 1024 * 1024, nullptr, false};
        ggml_context* exec_ctx       = ggml_init(exec_params);
        GGML_ASSERT(exec_ctx != nullptr);
        InspectableLinear linear(256, 4, true, false, true, 1.f / 128.f);
        linear.init(exec_ctx, selection, "layer");
        ggml_tensor* exec_weight = linear.parameter("weight");
        if (exec_weight->type == GGML_TYPE_Q8_0) {
            GGML_ASSERT(loader.load_comfy_int8_tensorwise(weight, exec_weight, nullptr));
            if (!linear.rotation_op_enabled()) {
                ggml_tensor* h256 = linear.parameter("weight.convrot_h256");
                std::vector<float> matrix(256 * 256, 0.f);
                for (size_t row = 0; row < 256; ++row) {
                    matrix[row * 256 + row] = 1.f;
                    float* values = matrix.data() + row * 256;
                    for (size_t stride = 1; stride < 256; stride *= 4) {
                        for (size_t base = 0; base < 256; base += 4 * stride) {
                            for (size_t i = 0; i < stride; ++i) {
                                float* v      = values + base + i;
                                const float a = v[0], b = v[stride], c = v[2 * stride], d = v[3 * stride];
                                v[0]          = (a + b + c - d) * 0.5f;
                                v[stride]     = (a + b - c + d) * 0.5f;
                                v[2 * stride] = (a - b + c + d) * 0.5f;
                                v[3 * stride] = (-a + b + c + d) * 0.5f;
                            }
                        }
                    }
                }
                memcpy(h256->data, matrix.data(), matrix.size() * sizeof(float));
            }
        } else {
            GGML_ASSERT(exec_weight->type == GGML_TYPE_F16 || exec_weight->type == GGML_TYPE_F32);
            GGML_ASSERT(loader.load_tensor(weight, exec_weight));
        }
        memcpy(linear.parameter("bias")->data, bias_values.data(), bias_values.size() * sizeof(float));
        ggml_tensor* exec_input = ggml_new_tensor_2d(exec_ctx, GGML_TYPE_F32, 256, 2);
        memcpy(exec_input->data, input_values.data(), input_values.size() * sizeof(float));
        GGMLRunnerContext exec_runner;
        exec_runner.ggml_ctx = exec_ctx;
        ggml_tensor* exec_output = linear.forward(&exec_runner, exec_input);
        ggml_cgraph* graph = ggml_new_graph(exec_ctx);
        ggml_build_forward_expand(graph, exec_output);
        GGML_ASSERT(ggml_graph_compute_with_ctx(exec_ctx, graph, 1) == GGML_STATUS_SUCCESS);
        std::vector<float> values(8);
        memcpy(values.data(), exec_output->data, values.size() * sizeof(float));
        ggml_free(exec_ctx);
        return values;
    };
    auto f32_compat_selection = compatibility_selection;
    f32_compat_selection.at("layer.weight").expected_type = GGML_TYPE_F32;
    const auto reference_values = run_linear(f32_compat_selection);
    GGML_ASSERT(std::fabs(reference_values[0] - bias_values[0]) > 0.1f);
    GGML_ASSERT(std::fabs(reference_values[4] - bias_values[0]) > 0.1f);
    for (const auto& selection : {automatic_selection, dense_selection}) {
        const auto q8_values = run_linear(selection);
        for (size_t i = 0; i < q8_values.size(); ++i) {
            GGML_ASSERT(std::isfinite(q8_values[i]));
            GGML_ASSERT(std::fabs(q8_values[i] - reference_values[i]) < 0.05f);
        }
    }
    GGML_ASSERT(unset_test_environment("SD_CONVROT_MODE") == 0);
    ggml_backend_free(cpu_backend);

    // Tensorwise Int8 without ConvRot can have a final row chunk shorter than
    // the loader's 4096-element conversion buffer.
    write_fixture(path, "{\"format\":\"int8_tensorwise\"}", 0.5f, 5000);
    ModelLoader wide_loader;
    GGML_ASSERT(wide_loader.init_from_file(path.string()));
    const TensorStorage& wide_weight = find_tensor(wide_loader, "layer.weight");
    ggml_init_params wide_params = {ggml_tensor_overhead() + 4 * 5000 * sizeof(ggml_fp16_t) + 4096,
                                    nullptr, false};
    ggml_context* wide_ctx = ggml_init(wide_params);
    GGML_ASSERT(wide_ctx != nullptr);
    ggml_tensor* wide_decoded = ggml_new_tensor_2d(wide_ctx, GGML_TYPE_F16, 5000, 4);
    GGML_ASSERT(wide_loader.load_tensor(wide_weight, wide_decoded));
    GGML_ASSERT(ggml_fp16_to_fp32(static_cast<const ggml_fp16_t*>(wide_decoded->data)[0]) == 1.0f);
    GGML_ASSERT(ggml_fp16_to_fp32(static_cast<const ggml_fp16_t*>(wide_decoded->data)[4999]) == 0.0f);
    ggml_free(wide_ctx);

    write_fixture(path, "{\"format\":\"int8_tensorwise\"}", 0.5f, 256);
    ModelLoader plain_loader;
    GGML_ASSERT(plain_loader.init_from_file(path.string()));
    plain_loader.set_wtype_override(GGML_TYPE_Q8_0, "");
    const TensorStorage& plain_weight = find_tensor(plain_loader, "layer.weight");
    GGML_ASSERT(plain_weight.is_comfy_int8_tensorwise && !plain_weight.comfy_int8_convrot);
    GGML_ASSERT(plain_weight.expected_type == GGML_TYPE_F16);
    ggml_init_params plain_params = {ggml_tensor_overhead() + 4 * 256 * sizeof(ggml_fp16_t) + 4096,
                                     nullptr, false};
    ggml_context* plain_ctx = ggml_init(plain_params);
    GGML_ASSERT(plain_ctx != nullptr);
    ggml_tensor* plain_decoded = ggml_new_tensor_2d(plain_ctx, GGML_TYPE_F16, 256, 4);
    GGML_ASSERT(plain_loader.load_tensor(plain_weight, plain_decoded));
    GGML_ASSERT(ggml_fp16_to_fp32(static_cast<const ggml_fp16_t*>(plain_decoded->data)[0]) == 1.f);
    ggml_free(plain_ctx);

    write_fixture(path, marker, std::numeric_limits<float>::quiet_NaN());
    ModelLoader invalid_scale_loader;
    GGML_ASSERT(invalid_scale_loader.init_from_file(path.string()));
    const TensorStorage& invalid_scale_weight = find_tensor(invalid_scale_loader, "layer.weight");
    ggml_init_params invalid_scale_params     = {ggml_tensor_overhead() * 2 + 1024 + 4 * sizeof(float) + 4096, nullptr, false};
    ggml_context* invalid_scale_ctx           = ggml_init(invalid_scale_params);
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

    write_markerless_fixture(path);
    std::vector<TensorStorage> markerless_tensors;
    std::string markerless_error;
    GGML_ASSERT(!read_safetensors_file(path.string(), markerless_tensors, &markerless_error));
    GGML_ASSERT(markerless_error.find("without a ComfyUI Int8 marker") != std::string::npos);

    write_fixture(path, "");
    ModelLoader empty_marker_loader;
    GGML_ASSERT(!empty_marker_loader.init_from_file(path.string()));

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
        ggml_context* real_ctx       = ggml_init(real_params);
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
