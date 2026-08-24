#include "gguf_io.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
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

enum class ComfyShapeResult {
    NOT_FOUND,
    RESTORED,
    INVALID,
};

static bool checked_element_count(const int64_t* ne,
                                  size_t n_dims,
                                  uint64_t* element_count) {
    uint64_t count = 1;
    for (size_t i = 0; i < n_dims; ++i) {
        if (ne[i] <= 0 || static_cast<uint64_t>(ne[i]) > std::numeric_limits<uint64_t>::max() / count) {
            return false;
        }
        count *= static_cast<uint64_t>(ne[i]);
    }
    *element_count = count;
    return true;
}

static ComfyShapeResult apply_comfy_shape_values(const int64_t* shape,
                                                 size_t shape_size,
                                                 uint64_t physical_element_count,
                                                 int64_t* ne,
                                                 int* n_dims) {
    if (shape_size == 0 || shape_size > GGML_MAX_DIMS + 1 ||
        !std::all_of(shape, shape + shape_size, [](int64_t dim) { return dim > 0; })) {
        return ComfyShapeResult::INVALID;
    }

    uint64_t logical_element_count = 1;
    for (size_t i = 0; i < shape_size; ++i) {
        const uint64_t dim = static_cast<uint64_t>(shape[i]);
        if (dim > std::numeric_limits<uint64_t>::max() / logical_element_count) {
            return ComfyShapeResult::INVALID;
        }
        logical_element_count *= dim;
    }
    if (logical_element_count != physical_element_count) {
        return ComfyShapeResult::INVALID;
    }

    const size_t collapsed_dims = shape_size > GGML_MAX_DIMS
                                      ? shape_size - GGML_MAX_DIMS + 1
                                      : 0;
    uint64_t collapsed_dimension = 1;
    for (size_t i = 0; i < collapsed_dims; ++i) {
        const uint64_t dim = static_cast<uint64_t>(shape[i]);
        if (dim > std::numeric_limits<uint64_t>::max() / collapsed_dimension) {
            return ComfyShapeResult::INVALID;
        }
        collapsed_dimension *= dim;
    }
    if (collapsed_dimension > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return ComfyShapeResult::INVALID;
    }
    std::fill(ne, ne + GGML_MAX_DIMS, 1);
    std::reverse_copy(shape + collapsed_dims, shape + shape_size, ne);
    if (collapsed_dims > 0) {
        ne[GGML_MAX_DIMS - 1] = static_cast<int64_t>(collapsed_dimension);
        *n_dims = GGML_MAX_DIMS;
    } else {
        *n_dims = static_cast<int>(shape_size);
    }
    return ComfyShapeResult::RESTORED;
}

// ComfyUI-GGUF may reshape a quantized tensor to satisfy the quantizer's
// block-size requirement.  The original PyTorch dimensions are preserved in
// comfy.gguf.orig_shape.<tensor-name>; TensorStorage uses GGML's reversed
// dimension order, so restore that logical shape before model detection and
// loading while retaining the physical byte layout in the file.
static ComfyShapeResult apply_comfy_original_shape(const gguf_context* ctx,
                                                   const std::string& tensor_name,
                                                   uint64_t physical_element_count,
                                                   int64_t* ne,
                                                   int* n_dims) {
    const std::string key = "comfy.gguf.orig_shape." + tensor_name;
    const int64_t key_id  = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        return ComfyShapeResult::NOT_FOUND;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_INT32) {
        return ComfyShapeResult::INVALID;
    }

    const size_t shape_size = gguf_get_arr_n(ctx, key_id);
    if (shape_size == 0 || shape_size > GGML_MAX_DIMS + 1) {
        return ComfyShapeResult::INVALID;
    }

    const int32_t* shape = static_cast<const int32_t*>(gguf_get_arr_data(ctx, key_id));
    if (shape == nullptr) {
        return ComfyShapeResult::INVALID;
    }
    std::vector<int64_t> shape_values(shape, shape + shape_size);
    return apply_comfy_shape_values(shape_values.data(),
                                    shape_values.size(),
                                    physical_element_count,
                                    ne,
                                    n_dims);
}

static ggml_type comfy_expected_type(ggml_type source_type, int64_t logical_row_width) {
    if (logical_row_width <= 0 ||
        logical_row_width % ggml_blck_size(GGML_TYPE_Q4_0) != 0 ||
        logical_row_width % ggml_blck_size(source_type) == 0) {
        return GGML_TYPE_COUNT;
    }

    switch (source_type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return GGML_TYPE_Q4_0;
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return GGML_TYPE_Q8_0;
        default:
            return GGML_TYPE_COUNT;
    }
}

static void apply_comfy_expected_type(TensorStorage& tensor_storage,
                                      bool restored_comfy_shape,
                                      size_t* remapped_tensors) {
    if (!restored_comfy_shape) {
        return;
    }
    tensor_storage.has_comfy_original_shape = true;
    const ggml_type expected_type =
        comfy_expected_type(tensor_storage.type, tensor_storage.ne[0]);
    if (expected_type != GGML_TYPE_COUNT) {
        tensor_storage.expected_type = expected_type;
        ++*remapped_tensors;
    } else if (ggml_is_quantized(tensor_storage.type) &&
               tensor_storage.ne[0] % ggml_blck_size(tensor_storage.type) != 0) {
        LOG_WARN("ComfyUI GGUF: no quantized remap for tensor %s (source type %s, logical row %" PRId64 "); falling back to F32",
                 tensor_storage.name.c_str(),
                 ggml_type_name(tensor_storage.type),
                 tensor_storage.ne[0]);
    }
}

static bool checked_physical_element_count(const int64_t* ne,
                                           size_t n_dims,
                                           uint64_t* element_count,
                                           std::string* error,
                                           const std::string& tensor_name) {
    if (checked_element_count(ne, n_dims, element_count)) {
        return true;
    }
    set_error(error, "invalid or overflowing physical shape for tensor '" + tensor_name + "'");
    return false;
}

static ComfyShapeResult apply_fallback_comfy_shape(const GGUFReader& reader,
                                                   const GGUFTensorInfo& tensor_info,
                                                   uint64_t physical_element_count,
                                                   int64_t* logical_ne,
                                                   int* logical_n_dims,
                                                   std::string* error) {
    const auto* shape = reader.comfy_original_shape(tensor_info.name);
    if (shape == nullptr) {
        return ComfyShapeResult::NOT_FOUND;
    }
    const ComfyShapeResult result = apply_comfy_shape_values(shape->data(),
                                                              shape->size(),
                                                              physical_element_count,
                                                              logical_ne,
                                                              logical_n_dims);
    if (result != ComfyShapeResult::RESTORED) {
        if (result == ComfyShapeResult::INVALID) {
            set_error(error, "invalid comfy.gguf.orig_shape for tensor '" + tensor_info.name + "'");
        }
    }
    return result;
}

static void initialize_logical_shape(const std::vector<int64_t>& physical_shape,
                                     int64_t* logical_ne,
                                     int* logical_n_dims) {
    std::fill(logical_ne, logical_ne + GGML_MAX_DIMS, 1);
    std::copy(physical_shape.begin(), physical_shape.end(), logical_ne);
    *logical_n_dims = static_cast<int>(physical_shape.size());
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

        size_t data_offset          = gguf_reader.data_offset();
        size_t remapped_tensors     = 0;
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
            uint64_t physical_element_count = 0;
            if (!checked_physical_element_count(gguf_tensor_info.shape.data(),
                                                gguf_tensor_info.shape.size(),
                                                &physical_element_count,
                                                error,
                                                gguf_tensor_info.name)) {
                return false;
            }

            int64_t logical_ne[GGML_MAX_DIMS];
            int logical_n_dims = 0;
            initialize_logical_shape(gguf_tensor_info.shape, logical_ne, &logical_n_dims);
            const ComfyShapeResult shape_result =
                apply_fallback_comfy_shape(gguf_reader,
                                           gguf_tensor_info,
                                           physical_element_count,
                                           logical_ne,
                                           &logical_n_dims,
                                           error);
            if (shape_result == ComfyShapeResult::INVALID) {
                return false;
            }

            TensorStorage tensor_storage(
                gguf_tensor_info.name,
                gguf_tensor_info.type,
                logical_ne,
                logical_n_dims,
                0,
                data_offset + gguf_tensor_info.offset);

            apply_comfy_expected_type(tensor_storage,
                                      shape_result == ComfyShapeResult::RESTORED,
                                      &remapped_tensors);
            tensor_storages.push_back(tensor_storage);
        }

        if (remapped_tensors > 0) {
            LOG_INFO("ComfyUI GGUF remapped %zu quantized tensors in fallback reader", remapped_tensors);
        }
        return true;
    }

    int n_tensors = static_cast<int>(gguf_get_n_tensors(ctx_gguf_));

    size_t data_offset      = gguf_get_data_offset(ctx_gguf_);
    size_t remapped_tensors = 0;
    for (int i = 0; i < n_tensors; i++) {
        std::string name   = gguf_get_tensor_name(ctx_gguf_, i);
        ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset      = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        uint64_t physical_element_count = 0;
        if (!checked_physical_element_count(dummy->ne,
                                            GGML_MAX_DIMS,
                                            &physical_element_count,
                                            error,
                                            name)) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            return false;
        }

        int64_t logical_ne[GGML_MAX_DIMS];
        std::copy(dummy->ne, dummy->ne + GGML_MAX_DIMS, logical_ne);
        int logical_n_dims = ggml_n_dims(dummy);
        const ComfyShapeResult shape_result =
            apply_comfy_original_shape(ctx_gguf_,
                                        name,
                                        physical_element_count,
                                        logical_ne,
                                        &logical_n_dims);
        if (shape_result == ComfyShapeResult::INVALID) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            set_error(error, "invalid comfy.gguf.orig_shape for tensor '" + name + "'");
            return false;
        }

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
        apply_comfy_expected_type(tensor_storage,
                                  shape_result == ComfyShapeResult::RESTORED,
                                  &remapped_tensors);

        if (ggml_nbytes(dummy) != tensor_storage.nbytes()) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            set_error(error, "size mismatch for tensor '" + name + "'");
            return false;
        }

        tensor_storages.push_back(tensor_storage);
    }

    if (remapped_tensors > 0) {
        LOG_INFO("ComfyUI GGUF remapped %zu quantized tensors", remapped_tensors);
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
