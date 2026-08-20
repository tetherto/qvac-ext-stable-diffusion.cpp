#include "gguf_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "core/util.h"
#include "gguf.h"
#include "gguf_reader_ext.h"

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

// ComfyUI-GGUF may reshape a quantized tensor to satisfy the quantizer's
// block-size requirement.  The original PyTorch dimensions are preserved in
// comfy.gguf.orig_shape.<tensor-name>; TensorStorage uses GGML's reversed
// dimension order, so restore that logical shape before model detection and
// loading while retaining the physical byte layout in the file.
static bool apply_comfy_original_shape(const gguf_context* ctx,
                                       const std::string& tensor_name,
                                       int64_t* ne,
                                       int* n_dims) {
    const std::string key = "comfy.gguf.orig_shape." + tensor_name;
    const int64_t key_id  = gguf_find_key(ctx, key.c_str());
    if (key_id < 0 ||
        gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_INT32) {
        return false;
    }

    const size_t shape_size = gguf_get_arr_n(ctx, key_id);
    if (shape_size == 0 || shape_size > GGML_MAX_DIMS) {
        return false;
    }

    const int32_t* shape = static_cast<const int32_t*>(gguf_get_arr_data(ctx, key_id));
    for (size_t i = 0; i < shape_size; ++i) {
        if (shape[i] <= 0) {
            return false;
        }
        ne[i] = shape[shape_size - 1 - i];
    }
    for (size_t i = shape_size; i < GGML_MAX_DIMS; ++i) {
        ne[i] = 1;
    }
    *n_dims = static_cast<int>(shape_size);
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error) {
    tensor_storages.clear();

    gguf_context* ctx_gguf_ = nullptr;
    ggml_context* ctx_meta_ = nullptr;

    ctx_gguf_ = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        GGUFReader gguf_reader;
        if (!gguf_reader.load(file_path)) {
            set_error(error, "failed to open '" + file_path + "' with GGUFReader");
            return false;
        }

        size_t data_offset = gguf_reader.data_offset();
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
            TensorStorage tensor_storage(
                gguf_tensor_info.name,
                gguf_tensor_info.type,
                gguf_tensor_info.shape.data(),
                static_cast<int>(gguf_tensor_info.shape.size()),
                0,
                data_offset + gguf_tensor_info.offset);

            tensor_storages.push_back(tensor_storage);
        }

        return true;
    }

    int n_tensors = static_cast<int>(gguf_get_n_tensors(ctx_gguf_));

    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name   = gguf_get_tensor_name(ctx_gguf_, i);
        ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset      = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        int64_t logical_ne[GGML_MAX_DIMS];
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            logical_ne[dim] = dummy->ne[dim];
        }
        int logical_n_dims       = ggml_n_dims(dummy);
        bool restored_comfy_shape = apply_comfy_original_shape(ctx_gguf_, name, logical_ne, &logical_n_dims);

        TensorStorage tensor_storage(name,
                                     dummy->type,
                                     logical_ne,
                                     logical_n_dims,
                                     0,
                                     offset);

        // ComfyUI packs quantized tensors using a physical row width that is
        // compatible with the source quant block, then records the original
        // logical shape in metadata. If the logical matrix row is not valid
        // for that source type, transcode it once to Q4_0 (32-value blocks)
        // instead of expanding the entire matrix to F32 at runtime.
        if (restored_comfy_shape &&
            ggml_is_quantized(dummy->type) &&
            logical_ne[0] % ggml_blck_size(dummy->type) != 0 &&
            logical_ne[0] % ggml_blck_size(GGML_TYPE_Q4_0) == 0) {
            tensor_storage.expected_type = GGML_TYPE_Q4_0;
        }

        if (ggml_nbytes(dummy) != tensor_storage.nbytes()) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            set_error(error, "size mismatch for tensor '" + name + "'");
            return false;
        }

        tensor_storages.push_back(tensor_storage);
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error) {
    gguf_context* gguf_ctx = gguf_init_empty();
    if (gguf_ctx == nullptr) {
        set_error(error, "gguf_init_empty failed");
        return false;
    }

    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor = write_tensor.tensor;
        if (tensor == nullptr) {
            set_error(error, "null tensor cannot be written to GGUF");
            gguf_free(gguf_ctx);
            return false;
        }
        gguf_add_tensor(gguf_ctx, tensor);
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    bool success = gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    if (!success) {
        set_error(error, "failed to write GGUF file '" + file_path + "'");
    }
    gguf_free(gguf_ctx);
    return success;
}

GGUFStreamingWriter::~GGUFStreamingWriter() {
    close();
}

bool GGUFStreamingWriter::write_metadata(const std::string& file_path,
                                         const std::vector<TensorWritePlan>& tensors,
                                         std::string* error) {
    close();
    tensors_   = tensors;
    file_size_ = 0;

    size_t meta_mem = 1 * 1024 * 1024 + tensors.size() * ggml_tensor_overhead();
    meta_ctx_       = ggml_init({meta_mem, nullptr, true});
    if (meta_ctx_ == nullptr) {
        set_error(error, "ggml_init failed for GGUF metadata");
        return false;
    }

    gguf_ctx_ = gguf_init_empty();
    if (gguf_ctx_ == nullptr) {
        set_error(error, "gguf_init_empty failed");
        close();
        return false;
    }

    for (const TensorWritePlan& plan : tensors) {
        ggml_tensor* tensor = ggml_new_tensor(meta_ctx_, plan.type, plan.n_dims, plan.ne);
        if (tensor == nullptr) {
            set_error(error, "ggml_new_tensor failed for tensor '" + plan.name + "'");
            close();
            return false;
        }
        ggml_set_name(tensor, plan.name.c_str());
        gguf_add_tensor(gguf_ctx_, tensor);
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    FILE* file = fopen(file_path.c_str(), "wb+");
    if (file == nullptr) {
        set_error(error, "failed to open output file '" + file_path + "'");
        close();
        return false;
    }

    // ggml exposes GGUF metadata writing through FILE* only. Keep FILE usage
    // isolated here; tensor data is written through std::fstream by the shared
    // streaming pipeline.
    if (!gguf_write_to_file_ptr(gguf_ctx_, file, true)) {
        fclose(file);
        set_error(error, "failed to write GGUF metadata to '" + file_path + "'");
        close();
        return false;
    }
    fclose(file);

    const uint64_t data_start = gguf_get_meta_size(gguf_ctx_);
    tensor_offsets_.resize(tensors.size());
    file_size_ = data_start;
    for (size_t i = 0; i < tensors.size(); i++) {
        tensor_offsets_[i] = data_start + gguf_get_tensor_offset(gguf_ctx_, static_cast<int64_t>(i));
        file_size_         = std::max(file_size_, tensor_offsets_[i] + tensors[i].nbytes());
    }
    return true;
}

bool GGUFStreamingWriter::write_tensor(std::ostream& output,
                                       size_t tensor_index,
                                       const uint8_t* data,
                                       size_t size,
                                       std::string* error) const {
    if (tensor_index >= tensors_.size() || tensor_index >= tensor_offsets_.size()) {
        set_error(error, "invalid GGUF tensor index");
        return false;
    }
    const TensorWritePlan& plan = tensors_[tensor_index];
    if (size != plan.nbytes()) {
        set_error(error, "size mismatch while writing tensor '" + plan.name + "'");
        return false;
    }
    output.seekp(static_cast<std::streamoff>(tensor_offsets_[tensor_index]), std::ios::beg);
    if (!output) {
        set_error(error, "failed to seek output for tensor '" + plan.name + "'");
        return false;
    }
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    if (!output) {
        set_error(error, "failed to write tensor '" + plan.name + "'");
        return false;
    }
    return true;
}

uint64_t GGUFStreamingWriter::file_size() const {
    return file_size_;
}

void GGUFStreamingWriter::close() {
    tensor_offsets_.clear();
    tensors_.clear();
    file_size_ = 0;
    if (gguf_ctx_ != nullptr) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
    if (meta_ctx_ != nullptr) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
}
