#include "safetensors_io.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "binary_io.h"
#include "core/util.h"
#include "json.hpp"

namespace fs = std::filesystem;

static constexpr size_t ST_HEADER_SIZE_LEN = 8;

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

static std::string resolve_index_shard_path(const std::string& index_path, const std::string& shard_path) {
    fs::path shard_fs_path(shard_path);
    if (shard_fs_path.is_absolute()) {
        return shard_fs_path.lexically_normal().string();
    }
    return (fs::path(index_path).parent_path() / shard_fs_path).lexically_normal().string();
}

bool is_safetensors_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        return false;
    }

    size_t header_size_ = model_io::read_u64(header_size_buf);
    if (header_size_ > file_size_ - ST_HEADER_SIZE_LEN || header_size_ <= 2) {
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        return false;
    }
    try {
        nlohmann::json header_ = nlohmann::json::parse(header_buf.data());
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

static ggml_type safetensors_dtype_to_ggml_type(const std::string& dtype) {
    ggml_type ttype = GGML_TYPE_COUNT;
    if (dtype == "F16") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "BF16") {
        ttype = GGML_TYPE_BF16;
    } else if (dtype == "F32") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F64") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F8_E4M3") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "F8_E5M2") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "I32") {
        ttype = GGML_TYPE_I32;
    } else if (dtype == "I64") {
        ttype = GGML_TYPE_I32;
    } else if (dtype == "I8") {
        ttype = GGML_TYPE_I8;
    }
    return ttype;
}

struct SafetensorsTensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end   = 0;
};

struct ComfyInt8Info {
    bool convrot            = false;
    uint32_t group_size     = 0;
    TensorStorageSidecar scale;
};

static bool read_safetensors_tensor_info(const nlohmann::json& value,
                                         const std::string& name,
                                         uint64_t data_size,
                                         SafetensorsTensorInfo* result,
                                         std::string* error) {
    try {
        if (!value.is_object() || !value.contains("dtype") || !value["dtype"].is_string() ||
            !value.contains("shape") || !value["shape"].is_array() ||
            !value.contains("data_offsets") || !value["data_offsets"].is_array() ||
            value["data_offsets"].size() != 2) {
            set_error(error, "invalid safetensors descriptor for tensor '" + name + "'");
            return false;
        }

        SafetensorsTensorInfo info;
        info.dtype = value["dtype"].get<std::string>();
        if (value["shape"].size() > SD_MAX_DIMS) {
            set_error(error, "too many dimensions for tensor '" + name + "'");
            return false;
        }
        uint64_t elements = 1;
        for (const auto& dimension : value["shape"]) {
            int64_t size = dimension.get<int64_t>();
            if (size <= 0 || elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                                              static_cast<uint64_t>(size)) {
                set_error(error, "invalid dimension for tensor '" + name + "'");
                return false;
            }
            elements *= static_cast<uint64_t>(size);
            info.shape.push_back(size);
        }
        info.begin = value["data_offsets"][0].get<uint64_t>();
        info.end   = value["data_offsets"][1].get<uint64_t>();
        if (info.begin > info.end || info.end > data_size) {
            set_error(error, "data offsets out of bounds for tensor '" + name + "'");
            return false;
        }
        *result = std::move(info);
        return true;
    } catch (const std::exception&) {
        set_error(error, "invalid safetensors descriptor for tensor '" + name + "'");
        return false;
    }
}

static bool is_power_of_four(uint64_t value) {
    if (value < 4) {
        return false;
    }
    while (value % 4 == 0) {
        value /= 4;
    }
    return value == 1;
}

static bool read_comfy_int8_metadata(std::ifstream& file,
                                     const std::map<std::string, SafetensorsTensorInfo>& tensors,
                                     uint64_t data_start,
                                     std::map<std::string, ComfyInt8Info>* result,
                                     std::set<std::string>* scale_tensor_names,
                                     std::string* error) {
    result->clear();
    scale_tensor_names->clear();
    constexpr const char* marker_suffix = ".comfy_quant";
    constexpr size_t marker_suffix_len  = 12;

    for (const auto& [marker_name, marker] : tensors) {
        if (!ends_with(marker_name, marker_suffix)) {
            continue;
        }
        if (marker.dtype != "U8") {
            set_error(error, "ComfyUI quantization marker '" + marker_name + "' must use U8 storage");
            return false;
        }
        if (marker.end - marker.begin > 4096) {
            set_error(error, "ComfyUI quantization marker '" + marker_name + "' is too large");
            return false;
        }
        uint64_t marker_elements = 1;
        for (int64_t dimension : marker.shape) {
            if (marker_elements > std::numeric_limits<uint64_t>::max() /
                                      static_cast<uint64_t>(dimension)) {
                set_error(error, "ComfyUI quantization marker '" + marker_name + "' shape overflows");
                return false;
            }
            marker_elements *= static_cast<uint64_t>(dimension);
        }
        if (marker_elements != marker.end - marker.begin) {
            set_error(error, "ComfyUI quantization marker '" + marker_name + "' has an invalid byte length");
            return false;
        }

        std::string marker_json(static_cast<size_t>(marker.end - marker.begin), '\0');
        file.clear();
        file.seekg(static_cast<std::streamoff>(data_start + marker.begin));
        file.read(marker_json.data(), static_cast<std::streamsize>(marker_json.size()));
        if (!file) {
            set_error(error, "failed to read ComfyUI quantization marker '" + marker_name + "'");
            return false;
        }

        // Reject malformed markers explicitly, including builds where JSON
        // parsing exceptions are disabled.
        nlohmann::json config = nlohmann::json::parse(marker_json, nullptr, false);
        if (config.is_discarded()) {
            set_error(error, "invalid JSON in ComfyUI quantization marker '" + marker_name + "'");
            return false;
        }
        if (!config.is_object() || !config.contains("format") || !config["format"].is_string()) {
            set_error(error, "invalid ComfyUI quantization marker '" + marker_name + "'");
            return false;
        }
        if (config["format"].get<std::string>() != "int8_tensorwise") {
            continue;
        }

        const std::string base = marker_name.substr(0, marker_name.size() - marker_suffix_len);
        const auto weight_it   = tensors.find(base + ".weight");
        const auto scale_it    = tensors.find(base + ".weight_scale");
        if (weight_it == tensors.end() || scale_it == tensors.end()) {
            set_error(error, "ComfyUI Int8 marker '" + marker_name + "' is missing its weight or weight_scale tensor");
            return false;
        }
        const SafetensorsTensorInfo& weight = weight_it->second;
        const SafetensorsTensorInfo& scale  = scale_it->second;
        if (weight.dtype != "I8" || weight.shape.size() != 2 || scale.dtype != "F32" ||
            scale.shape.size() != 2 || scale.shape[0] != weight.shape[0] || scale.shape[1] != 1) {
            set_error(error, "ComfyUI Int8 marker '" + marker_name + "' has incompatible weight and scale tensors");
            return false;
        }
        const uint64_t output_rows = static_cast<uint64_t>(weight.shape[0]);
        if (output_rows > std::numeric_limits<uint64_t>::max() / sizeof(float) ||
            scale.end - scale.begin != output_rows * sizeof(float)) {
            set_error(error, "ComfyUI Int8 marker '" + marker_name + "' has incompatible weight and scale tensors");
            return false;
        }

        ComfyInt8Info info;
        if (config.contains("convrot")) {
            if (!config["convrot"].is_boolean()) {
                set_error(error, "ComfyUI Int8 marker '" + marker_name + "' has a non-boolean convrot field");
                return false;
            }
            info.convrot = config["convrot"].get<bool>();
        }
        if (info.convrot) {
            if (!config.contains("convrot_groupsize") || !config["convrot_groupsize"].is_number_unsigned()) {
                set_error(error, "ComfyUI Int8 marker '" + marker_name + "' has no valid ConvRot group size");
                return false;
            }
            uint64_t group_size = config["convrot_groupsize"].get<uint64_t>();
            if (group_size != 256 || !is_power_of_four(group_size) ||
                static_cast<uint64_t>(weight.shape[1]) % group_size != 0) {
                set_error(error, "ComfyUI Int8 marker '" + marker_name + "' uses an unsupported ConvRot group size");
                return false;
            }
            info.group_size = static_cast<uint32_t>(group_size);
        } else if (config.contains("convrot_groupsize")) {
            set_error(error, "ComfyUI Int8 marker '" + marker_name + "' declares a group size without ConvRot");
            return false;
        }
        info.scale.name   = base + ".weight_scale";
        info.scale.type   = GGML_TYPE_F32;
        info.scale.n_dims = 2;
        // TensorStorage uses GGML's column-first shape convention.
        info.scale.ne[0]  = 1;
        info.scale.ne[1]  = weight.shape[0];
        info.scale.offset = data_start + scale.begin;
        info.scale.nbytes = scale.end - scale.begin;
        result->emplace(weight_it->first, info);
        scale_tensor_names->emplace(scale_it->first);
    }
    return true;
}

// https://huggingface.co/docs/safetensors/index
bool read_safetensors_file(const std::string& file_path,
                           std::vector<TensorStorage>& tensor_storages,
                           std::string* error,
                           std::map<std::string, std::string>* metadata) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "'");
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        set_error(error, "invalid safetensor file '" + file_path + "'");
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        set_error(error, "read safetensors header size failed: '" + file_path + "'");
        return false;
    }

    size_t header_size_ = model_io::read_u64(header_size_buf);
    if (header_size_ > file_size_ - ST_HEADER_SIZE_LEN) {
        set_error(error, "invalid safetensor file '" + file_path + "'");
        return false;
    }
    const size_t data_start = ST_HEADER_SIZE_LEN + header_size_;

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        set_error(error, "read safetensors header failed: '" + file_path + "'");
        return false;
    }

    nlohmann::json header_;
    try {
        header_ = nlohmann::json::parse(header_buf.data());
    } catch (const std::exception&) {
        set_error(error, "parsing safetensors header failed: '" + file_path + "'");
        return false;
    }

    if (metadata != nullptr) {
        metadata->clear();
        auto metadata_item = header_.find("__metadata__");
        if (metadata_item != header_.end() && metadata_item->is_object()) {
            for (const auto& item : metadata_item->items()) {
                if (item.value().is_string()) {
                    metadata->emplace(item.key(), item.value().get<std::string>());
                }
            }
        }
    }

    const uint64_t data_size = file_size_ - data_start;
    std::map<std::string, SafetensorsTensorInfo> tensor_infos;
    for (const auto& item : header_.items()) {
        if (item.key() == "__metadata__") {
            continue;
        }
        SafetensorsTensorInfo info;
        if (!read_safetensors_tensor_info(item.value(), item.key(), data_size, &info, error)) {
            return false;
        }
        tensor_infos.emplace(item.key(), std::move(info));
    }
    std::map<std::string, ComfyInt8Info> comfy_int8_tensors;
    std::set<std::string> comfy_int8_scale_tensors;
    if (!read_comfy_int8_metadata(file,
                                  tensor_infos,
                                  data_start,
                                  &comfy_int8_tensors,
                                  &comfy_int8_scale_tensors,
                                  error)) {
        return false;
    }

    tensor_storages.clear();
    for (const auto& [name, tensor_info] : tensor_infos) {
        // LOG_DEBUG("%s %s\n", name.c_str(), tensor_info.dump().c_str());

        const std::string& dtype = tensor_info.dtype;

        if (dtype == "U8" || comfy_int8_scale_tensors.find(name) != comfy_int8_scale_tensors.end()) {
            continue;
        }

        const uint64_t begin = tensor_info.begin;
        const uint64_t end   = tensor_info.end;

        ggml_type type = safetensors_dtype_to_ggml_type(dtype);
        if (type == GGML_TYPE_COUNT) {
            set_error(error, "unsupported dtype '" + dtype + "' (tensor '" + name + "')");
            return false;
        }

        int n_dims              = static_cast<int>(tensor_info.shape.size());
        int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            ne[i] = tensor_info.shape[i];
        }

        if (n_dims == 5) {
            if (ne[0] > std::numeric_limits<int64_t>::max() / ne[1]) {
                set_error(error, "tensor dimensions overflow for '" + name + "'");
                return false;
            }
            n_dims = 4;
            ne[0]  = ne[0] * ne[1];
            ne[1]  = ne[2];
            ne[2]  = ne[3];
            ne[3]  = ne[4];
        }

        // ggml_n_dims returns 1 for scalars
        if (n_dims == 0) {
            n_dims = 1;
        }

        TensorStorage tensor_storage(name, type, ne, n_dims, 0, data_start + begin);
        tensor_storage.reverse_ne();

        uint64_t tensor_data_size = end - begin;

        bool tensor_size_ok;
        if (dtype == "F8_E4M3") {
            tensor_storage.is_f8_e4m3 = true;
            // f8 -> f16
            tensor_size_ok = (tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F8_E5M2") {
            tensor_storage.is_f8_e5m2 = true;
            // f8 -> f16
            tensor_size_ok = (tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F64") {
            tensor_storage.is_f64 = true;
            // f64 -> f32
            tensor_size_ok = (tensor_storage.nbytes() * 2 == tensor_data_size);
        } else if (dtype == "I64") {
            tensor_storage.is_i64 = true;
            // i64 -> i32
            tensor_size_ok = (tensor_storage.nbytes() * 2 == tensor_data_size);
        } else {
            tensor_size_ok = (tensor_storage.nbytes() == tensor_data_size);
        }
        if (!tensor_size_ok) {
            set_error(error, "size mismatch for tensor '" + name + "' (" + dtype + ")");
            return false;
        }

        auto comfy_int8 = comfy_int8_tensors.find(name);
        if (dtype == "I8") {
            if (comfy_int8 == comfy_int8_tensors.end()) {
                set_error(error, "unsupported Int8 safetensors tensor '" + name + "' without a ComfyUI Int8 marker");
                return false;
            }
            tensor_storage.is_comfy_int8_tensorwise = true;
            tensor_storage.comfy_int8_convrot       = comfy_int8->second.convrot;
            tensor_storage.comfy_int8_group_size    = comfy_int8->second.group_size;
            tensor_storage.comfy_int8_scale         = comfy_int8->second.scale;
            // The runtime reconstructs an F16 matrix before backend upload.
            tensor_storage.expected_type = GGML_TYPE_F16;
        }

        tensor_storages.push_back(tensor_storage);

        // LOG_DEBUG("%s %s", tensor_storage.to_string().c_str(), dtype.c_str());
    }

    return true;
}

bool read_safetensors_index_file(const std::string& file_path,
                                 std::vector<std::string>& shard_paths,
                                 std::string* error) {
    shard_paths.clear();

    std::ifstream file(file_path);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "'");
        return false;
    }

    nlohmann::json index;
    try {
        index = nlohmann::json::parse(file);
    } catch (const std::exception&) {
        set_error(error, "parsing safetensors index failed: '" + file_path + "'");
        return false;
    }

    if (!index.is_object() || !index.contains("weight_map") || !index["weight_map"].is_object()) {
        set_error(error, "invalid safetensors index '" + file_path + "'");
        return false;
    }

    std::unordered_set<std::string> seen_shard_paths;
    for (const auto& item : index["weight_map"].items()) {
        if (!item.value().is_string()) {
            set_error(error, "invalid shard path for tensor '" + item.key() + "'");
            return false;
        }

        std::string shard_path = resolve_index_shard_path(file_path,
                                                          item.value().get<std::string>());
        if (seen_shard_paths.insert(shard_path).second) {
            shard_paths.push_back(std::move(shard_path));
        }
    }

    if (shard_paths.empty()) {
        set_error(error, "safetensors index has no tensors: '" + file_path + "'");
        return false;
    }

    return true;
}

static bool ggml_type_to_safetensors_dtype(ggml_type type, std::string* dtype) {
    switch (type) {
        case GGML_TYPE_F16:
            *dtype = "F16";
            return true;
        case GGML_TYPE_BF16:
            *dtype = "BF16";
            return true;
        case GGML_TYPE_F32:
            *dtype = "F32";
            return true;
        case GGML_TYPE_I32:
            *dtype = "I32";
            return true;
        default:
            return false;
    }
}

bool write_safetensors_file(const std::string& file_path,
                            const std::vector<TensorWriteInfo>& tensors,
                            std::string* error) {
    nlohmann::ordered_json header = nlohmann::ordered_json::object();

    uint64_t data_offset = 0;
    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor = write_tensor.tensor;
        if (tensor == nullptr) {
            set_error(error, "null tensor cannot be written to safetensors");
            return false;
        }

        const std::string name = ggml_get_name(tensor);
        std::string dtype;
        if (!ggml_type_to_safetensors_dtype(tensor->type, &dtype)) {
            set_error(error,
                      "unsupported safetensors dtype '" + std::string(ggml_type_name(tensor->type)) +
                          "' for tensor '" + name + "'");
            return false;
        }

        const uint64_t tensor_nbytes = ggml_nbytes(tensor);

        nlohmann::ordered_json json_tensor_info = nlohmann::ordered_json::object();
        json_tensor_info["dtype"]               = dtype;

        nlohmann::ordered_json shape = nlohmann::ordered_json::array();
        for (int i = 0; i < write_tensor.n_dims; ++i) {
            shape.push_back(write_tensor.ne[write_tensor.n_dims - 1 - i]);
        }
        json_tensor_info["shape"] = shape;

        nlohmann::ordered_json data_offsets = nlohmann::ordered_json::array();
        data_offsets.push_back(data_offset);
        data_offsets.push_back(data_offset + tensor_nbytes);
        json_tensor_info["data_offsets"] = data_offsets;

        header[name] = json_tensor_info;
        data_offset += tensor_nbytes;
    }

    const std::string header_str = header.dump();

    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "' for writing");
        return false;
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    model_io::write_u64(file, header_str.size());
    file.write(header_str.data(), header_str.size());
    if (!file) {
        set_error(error, "failed to write safetensors header to '" + file_path + "'");
        return false;
    }

    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor        = write_tensor.tensor;
        const std::string name     = ggml_get_name(tensor);
        const size_t tensor_nbytes = ggml_nbytes(tensor);
        file.write((const char*)tensor->data, tensor_nbytes);
        if (!file) {
            set_error(error,
                      "failed to write tensor '" + name + "' to '" + file_path + "'");
            return false;
        }
    }

    return true;
}

bool SafetensorsStreamingWriter::write_metadata(const std::string& file_path,
                                                const std::vector<TensorWritePlan>& tensors,
                                                std::string* error) {
    file_path_ = file_path;
    tensors_   = tensors;
    tensor_offsets_.clear();
    data_start_ = 0;
    file_size_  = 0;

    nlohmann::ordered_json header = nlohmann::ordered_json::object();
    uint64_t data_offset          = 0;
    tensor_offsets_.resize(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
        const TensorWritePlan& plan = tensors[i];
        std::string dtype;
        if (!ggml_type_to_safetensors_dtype(plan.type, &dtype)) {
            set_error(error,
                      "unsupported safetensors dtype '" + std::string(ggml_type_name(plan.type)) +
                          "' for tensor '" + plan.name + "'");
            return false;
        }

        nlohmann::ordered_json json_tensor_info = nlohmann::ordered_json::object();
        json_tensor_info["dtype"]               = dtype;

        nlohmann::ordered_json shape = nlohmann::ordered_json::array();
        for (int j = 0; j < plan.n_dims; ++j) {
            shape.push_back(plan.ne[plan.n_dims - 1 - j]);
        }
        json_tensor_info["shape"] = shape;

        nlohmann::ordered_json data_offsets = nlohmann::ordered_json::array();
        data_offsets.push_back(data_offset);
        data_offsets.push_back(data_offset + plan.nbytes());
        json_tensor_info["data_offsets"] = data_offsets;

        header[plan.name]  = json_tensor_info;
        tensor_offsets_[i] = data_offset;
        data_offset += plan.nbytes();
    }

    const std::string header_str = header.dump();
    data_start_                  = ST_HEADER_SIZE_LEN + header_str.size();

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    std::ofstream file(file_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "' for writing");
        return false;
    }

    uint8_t header_size[ST_HEADER_SIZE_LEN];
    for (int i = 0; i < static_cast<int>(ST_HEADER_SIZE_LEN); ++i) {
        header_size[i] = static_cast<uint8_t>((header_str.size() >> (8 * i)) & 0xFF);
    }
    file.write(reinterpret_cast<const char*>(header_size), sizeof(header_size));
    file.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    if (!file) {
        set_error(error, "failed to write safetensors header to '" + file_path + "'");
        return false;
    }

    file_size_ = data_start_ + data_offset;
    return true;
}

bool SafetensorsStreamingWriter::write_tensor(std::ostream& output,
                                              size_t tensor_index,
                                              const uint8_t* data,
                                              size_t size,
                                              std::string* error) const {
    if (tensor_index >= tensors_.size() || tensor_index >= tensor_offsets_.size()) {
        set_error(error, "invalid safetensors tensor index");
        return false;
    }
    const TensorWritePlan& plan = tensors_[tensor_index];
    if (size != plan.nbytes()) {
        set_error(error, "size mismatch while writing tensor '" + plan.name + "'");
        return false;
    }
    output.seekp(static_cast<std::streamoff>(data_start_ + tensor_offsets_[tensor_index]), std::ios::beg);
    if (!output) {
        set_error(error, "failed to seek output for tensor '" + plan.name + "'");
        return false;
    }
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    if (!output) {
        set_error(error, "failed to write tensor '" + plan.name + "' to '" + file_path_ + "'");
        return false;
    }
    return true;
}

uint64_t SafetensorsStreamingWriter::file_size() const {
    return file_size_;
}
