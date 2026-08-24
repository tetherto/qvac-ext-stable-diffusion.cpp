#ifndef __SD_MODEL_IO_GGUF_READER_EXT_H__
#define __SD_MODEL_IO_GGUF_READER_EXT_H__

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "core/util.h"
#include "ggml.h"

struct GGUFTensorInfo {
    std::string name;
    ggml_type type;
    std::vector<int64_t> shape;
    size_t offset;
};

enum class GGUFMetadataType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

class GGUFReader {
private:
    std::vector<GGUFTensorInfo> tensors_;
    std::map<std::string, std::vector<int64_t>> comfy_original_shapes_;
    size_t data_offset_;
    size_t alignment_ = 32;  // default alignment is 32

    template <typename T>
    bool safe_read(std::ifstream& fin, T& value) {
        fin.read(reinterpret_cast<char*>(&value), sizeof(T));
        return fin.good();
    }

    bool safe_read(std::ifstream& fin, char* buffer, size_t size) {
        fin.read(buffer, size);
        return fin.good();
    }

    bool safe_seek(std::ifstream& fin, std::streamoff offset, std::ios::seekdir dir) {
        fin.seekg(offset, dir);
        return fin.good();
    }

    bool skip_value(std::ifstream& fin, uint32_t type) {
        switch (static_cast<GGUFMetadataType>(type)) {
            case GGUFMetadataType::UINT8:
            case GGUFMetadataType::INT8:
            case GGUFMetadataType::BOOL:
                return safe_seek(fin, 1, std::ios::cur);

            case GGUFMetadataType::UINT16:
            case GGUFMetadataType::INT16:
                return safe_seek(fin, 2, std::ios::cur);

            case GGUFMetadataType::UINT32:
            case GGUFMetadataType::INT32:
            case GGUFMetadataType::FLOAT32:
                return safe_seek(fin, 4, std::ios::cur);

            case GGUFMetadataType::UINT64:
            case GGUFMetadataType::INT64:
            case GGUFMetadataType::FLOAT64:
                return safe_seek(fin, 8, std::ios::cur);

            case GGUFMetadataType::STRING: {
                uint64_t len = 0;
                if (!safe_read(fin, len) ||
                    len > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
                    return false;
                }
                return safe_seek(fin, static_cast<std::streamoff>(len), std::ios::cur);
            }

            case GGUFMetadataType::ARRAY: {
                uint32_t elem_type = 0;
                uint64_t len       = 0;
                if (!safe_read(fin, elem_type) || !safe_read(fin, len)) {
                    return false;
                }
                for (uint64_t i = 0; i < len; ++i) {
                    if (!skip_value(fin, elem_type)) {
                        return false;
                    }
                }
                return true;
            }

            default:
                return false;
        }
    }

    bool read_metadata(std::ifstream& fin) {
        uint64_t key_len = 0;
        if (!safe_read(fin, key_len))
            return false;

        if (key_len > 4096)
            return false;

        std::string key(key_len, '\0');
        if (!safe_read(fin, (char*)key.data(), key_len))
            return false;

        uint32_t type = 0;
        if (!safe_read(fin, type))
            return false;

        if (key == "general.alignment" && type == static_cast<uint32_t>(GGUFMetadataType::UINT32)) {
            uint32_t align_val = 0;
            if (!safe_read(fin, align_val))
                return false;

            if (align_val != 0 && (align_val & (align_val - 1)) == 0) {
                alignment_ = align_val;
                LOG_DEBUG("Found alignment: %zu", alignment_);
            } else {
                LOG_ERROR("Invalid alignment value %u, fallback to default %zu", align_val, alignment_);
            }
            return true;
        }

        if (type == static_cast<uint32_t>(GGUFMetadataType::ARRAY)) {
            uint32_t elem_type = 0;
            uint64_t len       = 0;
            if (!safe_read(fin, elem_type) || !safe_read(fin, len)) {
                return false;
            }

            constexpr const char* prefix = "comfy.gguf.orig_shape.";
            if (key.compare(0, std::char_traits<char>::length(prefix), prefix) == 0) {
                if (elem_type != static_cast<uint32_t>(GGUFMetadataType::INT32) ||
                    len == 0 || len > GGML_MAX_DIMS + 1) {
                    return false;
                }

                std::vector<int64_t> shape;
                shape.reserve(static_cast<size_t>(len));
                for (uint64_t i = 0; i < len; ++i) {
                    int32_t dim = 0;
                    if (!safe_read(fin, dim)) {
                        return false;
                    }
                    shape.push_back(dim);
                }
                comfy_original_shapes_[key.substr(std::char_traits<char>::length(prefix))] =
                    std::move(shape);
                return true;
            }

            for (uint64_t i = 0; i < len; ++i) {
                if (!skip_value(fin, elem_type)) {
                    return false;
                }
            }
            return true;
        }

        return skip_value(fin, type);
    }

    GGUFTensorInfo read_tensor_info(std::ifstream& fin) {
        GGUFTensorInfo info;

        uint64_t name_len;
        if (!safe_read(fin, name_len))
            throw std::runtime_error("read tensor name length failed");

        info.name.resize(name_len);
        if (!safe_read(fin, (char*)info.name.data(), name_len))
            throw std::runtime_error("read tensor name failed");

        uint32_t n_dims;
        if (!safe_read(fin, n_dims))
            throw std::runtime_error("read tensor dims failed");

        info.shape.resize(n_dims);
        for (uint32_t i = 0; i < n_dims; i++) {
            if (!safe_read(fin, info.shape[i]))
                throw std::runtime_error("read tensor shape failed");
        }

        if (n_dims > GGML_MAX_DIMS) {
            for (uint32_t i = GGML_MAX_DIMS; i < n_dims; i++) {
                if (info.shape[GGML_MAX_DIMS - 1] <= 0 ||
                    info.shape[i] <= 0 ||
                    info.shape[GGML_MAX_DIMS - 1] >
                        std::numeric_limits<int64_t>::max() / info.shape[i]) {
                    throw std::runtime_error("tensor shape overflows int64");
                }
                info.shape[GGML_MAX_DIMS - 1] *= info.shape[i];  // stack to last dim;
            }
            info.shape.resize(GGML_MAX_DIMS);
            n_dims = GGML_MAX_DIMS;
        }

        uint32_t type;
        if (!safe_read(fin, type))
            throw std::runtime_error("read tensor type failed");
        info.type = static_cast<ggml_type>(type);

        if (!safe_read(fin, info.offset))
            throw std::runtime_error("read tensor offset failed");

        return info;
    }

public:
    bool load(const std::string& file_path) {
        std::ifstream fin(file_path, std::ios::binary);
        if (!fin) {
            LOG_ERROR("failed to open '%s'", file_path.c_str());
            return false;
        }

        // --- Header ---
        char magic[4];
        if (!safe_read(fin, magic, 4) || strncmp(magic, "GGUF", 4) != 0) {
            LOG_ERROR("not a valid GGUF file");
            return false;
        }

        uint32_t version;
        if (!safe_read(fin, version))
            return false;

        uint64_t tensor_count, metadata_kv_count;
        if (!safe_read(fin, tensor_count))
            return false;
        if (!safe_read(fin, metadata_kv_count))
            return false;

        LOG_DEBUG("GGUF v%u, tensor_count=%llu, metadata_kv_count=%llu",
                  version, (unsigned long long)tensor_count, (unsigned long long)metadata_kv_count);

        // --- Read Metadata ---
        comfy_original_shapes_.clear();
        for (uint64_t i = 0; i < metadata_kv_count; i++) {
            if (!read_metadata(fin)) {
                LOG_ERROR("read meta data failed");
                return false;
            }
        }

        // --- Tensor Infos ---
        tensors_.clear();
        try {
            for (uint64_t i = 0; i < tensor_count; i++) {
                tensors_.push_back(read_tensor_info(fin));
            }
        } catch (const std::runtime_error& e) {
            LOG_ERROR("%s", e.what());
            return false;
        }

        data_offset_ = static_cast<size_t>(fin.tellg());
        if ((data_offset_ % alignment_) != 0) {
            data_offset_ = ((data_offset_ + alignment_ - 1) / alignment_) * alignment_;
        }
        fin.close();
        return true;
    }

    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    size_t data_offset() const { return data_offset_; }

    const std::vector<int64_t>* comfy_original_shape(const std::string& tensor_name) const {
        auto it = comfy_original_shapes_.find(tensor_name);
        return it == comfy_original_shapes_.end() ? nullptr : &it->second;
    }
};

#endif  // __SD_MODEL_IO_GGUF_READER_EXT_H__
