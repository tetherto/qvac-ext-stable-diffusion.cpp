#include "model_io/gguf_io.h"
#include "model_io/gguf_reader_ext.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

template <typename T>
void write_value(std::ofstream& file, T value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_gguf(const std::filesystem::path& path,
                const std::vector<int32_t>& original_shape,
                ggml_type type = GGML_TYPE_F32,
                const std::vector<uint64_t>& physical_shape = {6, 2}) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    assert(file.is_open());

    file.write("GGUF", 4);
    write_value<uint32_t>(file, 3);
    write_value<uint64_t>(file, 1);
    write_value<uint64_t>(file, 1);

    const std::string metadata_key = "comfy.gguf.orig_shape.foo";
    write_value<uint64_t>(file, metadata_key.size());
    file.write(metadata_key.data(), static_cast<std::streamsize>(metadata_key.size()));
    write_value<uint32_t>(file, 9);  // ARRAY
    write_value<uint32_t>(file, 5);  // INT32
    write_value<uint64_t>(file, original_shape.size());
    for (int32_t dim : original_shape) {
        write_value<int32_t>(file, dim);
    }

    write_value<uint64_t>(file, 3);
    file.write("foo", 3);
    write_value<uint32_t>(file, physical_shape.size());
    for (uint64_t dim : physical_shape) {
        write_value<uint64_t>(file, dim);
    }
    write_value<uint32_t>(file, static_cast<uint32_t>(type));
    write_value<uint64_t>(file, 0);

    const auto position = static_cast<uint64_t>(file.tellp());
    const uint64_t aligned = (position + 31) & ~uint64_t(31);
    for (uint64_t i = position; i < aligned; ++i) {
        file.put('\0');
    }
    uint64_t element_count = 1;
    for (uint64_t dim : physical_shape) {
        element_count *= dim;
    }
    const size_t data_size =
        static_cast<size_t>(element_count / ggml_blck_size(type)) * ggml_type_size(type);
    for (size_t i = 0; i < data_size; ++i) {
        file.put('\0');
    }
}

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      "stable-diffusion-comfy-shape-test.gguf";

    write_gguf(path, {2, 6});

    GGUFReader fallback_reader;
    assert(fallback_reader.load(path.string()));
    const auto* fallback_shape = fallback_reader.comfy_original_shape("foo");
    assert(fallback_shape != nullptr);
    assert(*fallback_shape == std::vector<int64_t>({2, 6}));

    std::vector<TensorStorage> tensors;
    std::string error;
    assert(read_gguf_file(path.string(), tensors, &error));
    assert(tensors.size() == 1);
    assert(tensors[0].has_comfy_original_shape);
    assert(tensors[0].n_dims == 2);
    assert(tensors[0].ne[0] == 6);
    assert(tensors[0].ne[1] == 2);

    write_gguf(path,
               {256, 96},
               GGML_TYPE_Q4_K,
               {256, 96});
    tensors.clear();
    error.clear();
    assert(read_gguf_file(path.string(), tensors, &error));
    assert(tensors.size() == 1);
    assert(tensors[0].has_comfy_original_shape);
    assert(tensors[0].expected_type == GGML_TYPE_Q4_0);
    assert(tensors[0].ne[0] == 96);
    assert(tensors[0].ne[1] == 256);

    write_gguf(path, {-2, 6});
    tensors.clear();
    error.clear();
    assert(!read_gguf_file(path.string(), tensors, &error));
    assert(error.find("invalid comfy.gguf.orig_shape") != std::string::npos);

    write_gguf(path, {INT32_MAX, INT32_MAX});
    tensors.clear();
    error.clear();
    assert(!read_gguf_file(path.string(), tensors, &error));
    assert(error.find("invalid comfy.gguf.orig_shape") != std::string::npos);

    std::filesystem::remove(path);
    return 0;
}
