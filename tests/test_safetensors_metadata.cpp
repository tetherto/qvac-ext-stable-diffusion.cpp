#include "core/util.h"
#include "model/adapter/lora.hpp"
#include "model_io/binary_io.h"
#include "model_io/safetensors_io.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace safetensors_metadata_test {

    const std::string header =
        R"({"model.diffusion_model.test.weight.diff":{"dtype":"F32","shape":[6,4],"data_offsets":[0,96]}})";

    void write_file(const std::filesystem::path& path, const std::string& json, size_t payload = 0) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        GGML_ASSERT(file.is_open());
        model_io::write_u64(file, json.size());
        file.write(json.data(), static_cast<std::streamsize>(json.size()));
        for (size_t i = 0; i < payload; ++i) {
            file.put('\0');
        }
        GGML_ASSERT(file.good());
    }

    bool read_file(const std::filesystem::path& path) {
        std::vector<TensorStorage> tensors;
        std::string error;
        return read_safetensors_file(path.string(), tensors, &error);
    }

    void test_reader(const std::filesystem::path& path) {
        write_file(path, header, 96);
        std::vector<TensorStorage> full;
        GGML_ASSERT(read_safetensors_file(path.string(), full));
        GGML_ASSERT(full.size() == 1);
        write_file(path, header);
        GGML_ASSERT(is_safetensors_file(path.string()));
        GGML_ASSERT(!read_file(path));
        {
            SDMetadataOnlyReadScope scope;
            std::vector<TensorStorage> metadata;
            GGML_ASSERT(read_safetensors_file(path.string(), metadata));
            GGML_ASSERT(metadata.size() == full.size());
            GGML_ASSERT(metadata[0].to_string() == full[0].to_string());
            GGML_ASSERT(metadata[0].nbytes() == 96);
            {
                SDMetadataOnlyReadScope nested;
                GGML_ASSERT(read_file(path));
            }
            GGML_ASSERT(read_file(path));
            bool other_thread_accepted = true;
            std::thread other([&]() { other_thread_accepted = read_file(path); });
            other.join();
            GGML_ASSERT(!other_thread_accepted);
        }
        GGML_ASSERT(!read_file(path));
        write_file(path, header, 4);
        GGML_ASSERT(!read_file(path));
        {
            SDMetadataOnlyReadScope scope;
            GGML_ASSERT(read_file(path));
        }
    }

    void test_invalid_headers(const std::filesystem::path& path) {
        SDMetadataOnlyReadScope scope;
        for (const auto& json : {
                 R"({"x":{"dtype":"F32","shape":[-1000000],"data_offsets":[0,18446744073705551616]}})",
                 R"({"x":{"dtype":"F32","shape":[2305843009213693952],"data_offsets":[0,9223372036854775808]}})",
                 R"({"x":{"dtype":"F32","shape":[4294967296,4294967296,1,1,1],"data_offsets":[0,0]}})",
                 R"({"x":{"dtype":"F32","shape":[0,4294967296,4294967296,1,1],"data_offsets":[0,0]}})",
                 R"({"x":{"dtype":"F64","shape":[2305843009213693952],"data_offsets":[0,0]}})",
                 R"({"x":{"dtype":"I64","shape":[2305843009213693952],"data_offsets":[0,0]}})",
                 R"({"x":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,0]}})",
                 R"({"x":{"dtype":"F32","shape":[4.0],"data_offsets":[0,16]}})",
                 R"({"x":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}})",
                 R"({"x":{"dtype":"F32","shape":null,"data_offsets":[0,4]}})",
                 R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})",
                 R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[-1000,-996]}})",
                 R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0.0,4.0]}})",
                 R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0]}})",
                 R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})",
                 R"({"x":{"dtype":"U16","shape":[1],"data_offsets":[0,2]}})",
                 R"({"x":{"dtype":"F32","shape":[1,1,1,1,1,1],"data_offsets":[0,4]}})",
                 "{invalid json"}) {
            write_file(path, json);
            GGML_ASSERT(!read_file(path));
        }
        const auto max_offset = std::numeric_limits<size_t>::max();
        write_file(path, "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" +
                             std::to_string(max_offset - 4) + "," + std::to_string(max_offset) + "]}}");
        GGML_ASSERT(!read_file(path));
        write_file(path, "{\"x\":{\"dtype\":\"F8_E4M3\",\"shape\":[1],\"data_offsets\":[0," +
                             std::to_string(max_offset / 2 + 2) + "]}}");
        GGML_ASSERT(!read_file(path));
        write_file(path, header);
        std::filesystem::resize_file(path, 8 + header.size() - 1);
        GGML_ASSERT(!read_file(path));
    }

    void test_converted_types(const std::filesystem::path& path) {
        SDMetadataOnlyReadScope scope;
        for (const auto& json : {
                 R"({"x":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}})",
                 R"({"x":{"dtype":"F8_E5M2","shape":[2],"data_offsets":[0,2]}})",
                 R"({"x":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}})",
                 R"({"x":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}})",
                 R"({"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}})"}) {
            write_file(path, json);
            GGML_ASSERT(read_file(path));
        }
    }

    void test_shapes(const std::filesystem::path& path) {
        for (const std::string shape : {"[]", "[2,3,4,1,1]", "[0,2,3]", "[2,0,3]", "[2,3,0]"}) {
            const size_t payload   = shape == "[]" ? 4 : shape == "[2,3,4,1,1]" ? 96
                                                                                : 0;
            const std::string json = "{\"x\":{\"dtype\":\"F32\",\"shape\":" + shape +
                                     ",\"data_offsets\":[0," + std::to_string(payload) + "]}}";
            write_file(path, json, payload);
            GGML_ASSERT(read_file(path));
            write_file(path, json);
            SDMetadataOnlyReadScope scope;
            GGML_ASSERT(read_file(path));
        }
        for (const std::string shape : {"[-1,0]", "[4.0]", "[0,4294967296,4294967296,1,1]"}) {
            write_file(path, "{\"x\":{\"dtype\":\"F32\",\"shape\":" + shape + ",\"data_offsets\":[0,0]}}");
            GGML_ASSERT(!read_file(path));
            sd_ctx_params_t params;
            sd_ctx_params_init(&params);
            const auto model_path = path.string();
            params.model_path     = model_path.c_str();
            sd_fit_workload_t workload;
            sd_fit_workload_init(&workload);
            sd_fit_result_t result{};
            GGML_ASSERT(sd_fit_params(&params, &workload, &result) == SD_FIT_ERROR);
            sd_fit_result_free(&result);
            GGML_ASSERT(!sd_get_metadata_only_read());
        }
    }

    void test_shard_and_lora(const std::filesystem::path& path) {
        write_file(path, header);
        auto index = path;
        index += ".index.json";
        {
            std::ofstream file(index);
            file << "{\"weight_map\":{\"model.diffusion_model.test.weight.diff\":\""
                 << path.filename().string() << "\"}}";
        }
        ModelLoader strict;
        GGML_ASSERT(!strict.init_from_file(index.string()));
        ggml_backend_t cpu = sd_backend_cpu_init();
        GGML_ASSERT(cpu != nullptr);
        {
            SDMetadataOnlyReadScope scope;
            ModelLoader loader;
            GGML_ASSERT(loader.init_from_file(index.string()));
            GGML_ASSERT(loader.get_tensor_storage_map().size() == 1);
            GGMLRunner::set_measure_mode(true);
            {
                LoraModel lora("metadata", cpu, cpu, path.string());
                GGML_ASSERT(lora.load_from_file(1));
                GGML_ASSERT(!lora.lora_tensors.empty());
            }
            GGMLRunner::set_measure_mode(false);
        }
        {
            LoraModel lora("strict", cpu, cpu, path.string());
            GGML_ASSERT(!lora.load_from_file(1));
        }
        ggml_backend_free(cpu);
        std::filesystem::remove(index);
    }

    void test_lora_scalars(const std::filesystem::path& path) {
        ggml_backend_t cpu = sd_backend_cpu_init();
        GGML_ASSERT(cpu != nullptr);
        for (const std::string suffix : {"alpha", "scale"}) {
            const std::string json =
                R"({"model.diffusion_model.test.lora_down.weight":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]},)"
                R"("model.diffusion_model.test.lora_up.weight":{"dtype":"F32","shape":[6,2],"data_offsets":[32,80]},)"
                "\"model.diffusion_model.test." +
                suffix + R"(":{"dtype":"F32","shape":[1],"data_offsets":[80,84]}})";
            int full_nodes = 0;
            for (bool measure : {false, true}) {
                write_file(path, json, measure ? 0 : 84);
                SDMetadataOnlyReadScope scope;
                GGMLRunner::set_measure_mode(measure);
                {
                    LoraModel lora("scalars", cpu, cpu, path.string());
                    GGML_ASSERT(lora.load_from_file(1));
                    auto* scalar = lora.lora_tensors.at("lora.model.diffusion_model.test.weight." + suffix);
                    GGML_ASSERT(LoraModel::read_scale(scalar) == (measure ? 1.0f : 0.0f));
                    GGML_ASSERT((scalar->data == nullptr) == measure);
                    ggml_init_params init = {};
                    init.mem_size         = 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false);
                    init.no_alloc         = true;
                    auto* ctx             = ggml_init(init);
                    GGML_ASSERT(ctx != nullptr);
                    ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), "ggml_runner_build_in_tensor:one");
                    auto* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 6);
                    auto* diff   = lora.get_lora_weight_diff("model.diffusion_model.test.weight", ctx, cpu);
                    GGML_ASSERT(diff != nullptr && ggml_are_same_shape(diff, weight));
                    auto* x                             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
                    WeightAdapter::ForwardParams params = {};
                    params.op_type                      = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
                    auto* out                           = lora.get_out_diff(ctx, cpu, x, weight, params, "model.diffusion_model.test.weight");
                    GGML_ASSERT(out != nullptr && out->ne[0] == 6 && out->ne[1] == 1);
                    auto* graph = ggml_new_graph_custom(ctx, 128, false);
                    ggml_build_forward_expand(graph, diff);
                    ggml_build_forward_expand(graph, out);
                    if (measure) {
                        GGML_ASSERT(ggml_graph_n_nodes(graph) == full_nodes);
                    } else {
                        full_nodes = ggml_graph_n_nodes(graph);
                    }
                    ggml_free(ctx);
                }
                GGMLRunner::set_measure_mode(false);
            }
        }
        ggml_backend_free(cpu);
    }

    void test_exception_cleanup(const std::filesystem::path& path) {
        write_file(path, R"({"x":{"dtype":42,"shape":[1],"data_offsets":[0,4]}})");
        bool caught = false;
        try {
            SDMetadataOnlyReadScope scope;
            read_file(path);
        } catch (const std::exception&) {
            caught = true;
        }
        GGML_ASSERT(caught);
        GGML_ASSERT(!sd_get_metadata_only_read());
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        const auto model_path = path.string();
        params.model_path     = model_path.c_str();
        sd_fit_workload_t workload;
        sd_fit_workload_init(&workload);
        sd_fit_result_t result{};
        GGML_ASSERT(sd_fit_params(&params, &workload, &result) == SD_FIT_ERROR);
        sd_fit_result_free(&result);
        GGML_ASSERT(!sd_get_metadata_only_read());
        write_file(path, header);
        GGML_ASSERT(!read_file(path));
    }

}  // namespace safetensors_metadata_test

int main() {
    using namespace safetensors_metadata_test;
    const auto path = std::filesystem::temp_directory_path() / "sd-test-safetensors-metadata.safetensors";
    test_exception_cleanup(path);
    test_reader(path);
    test_invalid_headers(path);
    test_converted_types(path);
    test_shapes(path);
    test_shard_and_lora(path);
    test_lora_scalars(path);
    std::filesystem::remove(path);
    return 0;
}
