#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

void write_fixture(const std::filesystem::path& path, const std::string& marker) {
    const std::string header = make_header(marker);
    std::vector<int8_t> weights(4 * 256, 0);
    weights[0] = 2;
    const float scales[] = {0.5f, 0.5f, 0.5f, 0.5f};

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

}  // namespace

int main() {
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
    GGML_ASSERT(weight.comfy_int8_scale_nbytes == 4 * sizeof(float));
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
