#include "core/util.h"
#include "json.hpp"
#include "model_io/binary_io.h"
#include "model_io/safetensors_io.h"
#include "upscaler.h"

#include <filesystem>
#include <fstream>
#include <limits>

namespace upscaler_fit_test {

    size_t write_model(const std::filesystem::path& path) {
        ggml_init_params init{};
        init.mem_size     = 2 * 1024 * 1024;
        init.no_alloc     = true;
        ggml_context* ctx = ggml_init(init);
        GGML_ASSERT(ctx != nullptr);
        ESRGANConfig config;
        config.num_block = 1;
        RRDBNet network(config);
        network.init(ctx, {}, "");
        std::map<std::string, ggml_tensor*> tensors;
        network.get_param_tensors(tensors);
        nlohmann::json header;
        size_t offset       = 0;
        size_t params_bytes = 0;
        for (const auto& entry : tensors) {
            std::vector<int64_t> shape;
            for (int i = ggml_n_dims(entry.second) - 1; i >= 0; --i) {
                shape.push_back(entry.second->ne[i]);
            }
            const size_t bytes  = ggml_nelements(entry.second) * sizeof(float);
            header[entry.first] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {offset, offset + bytes}}};
            offset += bytes;
            params_bytes += ends_with(entry.first, ".bias") ? bytes : bytes / 2;
        }
        const auto json = header.dump();
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        model_io::write_u64(file, json.size());
        file.write(json.data(), static_cast<std::streamsize>(json.size()));
        GGML_ASSERT(file.good());
        ggml_free(ctx);
        return params_bytes;
    }

    void set_host_memory(const char* gib) {
#if defined(_WIN32)
        GGML_ASSERT(_putenv_s("SD_FIT_DEBUG_HOST_MEMORY_GIB", gib) == 0);
#else
        GGML_ASSERT(setenv("SD_FIT_DEBUG_HOST_MEMORY_GIB", gib, 1) == 0);
#endif
    }

    void test_measurement(const std::string& path, size_t params_bytes) {
        sd_upscaler_fit_params_t params;
        sd_upscaler_fit_params_init(&params);
        params.esrgan_path = path.c_str();
        params.device      = SD_UPSCALER_DEVICE_CPU;
        params.width       = 16;
        params.height      = 12;
        params.tile_size   = 0;
        sd::fit_params::ModuleMemory full;
        GGML_ASSERT(sd_measure_upscaler(params, full));
        GGML_ASSERT(full.params_bytes == params_bytes);
        GGML_ASSERT(full.compute_bytes > 0 && full.host_bytes > 0);
        GGML_ASSERT(!GGMLRunner::measure_mode_enabled() && !sd_get_metadata_only_read());
        params.tile_size = 8;
        sd::fit_params::ModuleMemory tiled;
        GGML_ASSERT(sd_measure_upscaler(params, tiled));
        GGML_ASSERT(tiled.params_bytes == full.params_bytes);
        GGML_ASSERT(tiled.compute_bytes < full.compute_bytes);
        params.width  = 32;
        params.height = 24;
        sd::fit_params::ModuleMemory larger;
        GGML_ASSERT(sd_measure_upscaler(params, larger));
        GGML_ASSERT(larger.compute_bytes == tiled.compute_bytes);
        GGML_ASSERT(larger.host_bytes > tiled.host_bytes);
        params.width   = 16;
        params.height  = 12;
        params.repeats = 2;
        sd::fit_params::ModuleMemory repeated;
        GGML_ASSERT(sd_measure_upscaler(params, repeated));
        GGML_ASSERT(repeated.compute_bytes == tiled.compute_bytes);
        GGML_ASSERT(repeated.host_bytes > larger.host_bytes);
        params.direct = true;
        sd::fit_params::ModuleMemory direct;
        GGML_ASSERT(sd_measure_upscaler(params, direct));
        GGML_ASSERT(direct.params_bytes == repeated.params_bytes);
        GGML_ASSERT(direct.host_bytes == repeated.host_bytes);
        GGML_ASSERT(direct.compute_bytes <= repeated.compute_bytes);

        std::vector<TensorStorage> tensors;
        GGML_ASSERT(!read_safetensors_file(path, tensors));
        set_host_memory("64");
        sd_fit_result_t result{};
        GGML_ASSERT(sd_upscaler_fit_params(&params, &result) == SD_FIT_SUCCESS);
        GGML_ASSERT(!result.changed && result.report != nullptr);
        GGML_ASSERT(std::string(result.report).find("upscaler") != std::string::npos);
        GGML_ASSERT(std::string(result.report).find("host memory") != std::string::npos);
        sd_fit_result_free(&result);
        set_host_memory("0.5");
        GGML_ASSERT(sd_upscaler_fit_params(&params, &result) == SD_FIT_FAILURE);
        sd_fit_result_free(&result);
        set_host_memory("");

        params.repeats = 0;
        GGML_ASSERT(sd_upscaler_fit_params(&params, &result) == SD_FIT_ERROR);
        params.repeats = 100;
        GGML_ASSERT(sd_upscaler_fit_params(&params, &result) == SD_FIT_ERROR);
        params.repeats = 1;
        params.width   = std::numeric_limits<int>::max();
        GGML_ASSERT(sd_upscaler_fit_params(&params, &result) == SD_FIT_ERROR);
        params.width = -1;
        GGML_ASSERT(sd_upscaler_fit_params(&params, &result) == SD_FIT_ERROR);
        GGML_ASSERT(sd_upscaler_fit_params(nullptr, &result) == SD_FIT_ERROR);
        GGML_ASSERT(result.report == nullptr && !result.changed);
        GGML_ASSERT(sd_upscaler_fit_params(&params, nullptr) == SD_FIT_ERROR);
        params.width       = 16;
        params.esrgan_path = "missing-esrgan.safetensors";
        GGML_ASSERT(sd_upscaler_fit_params(&params, &result) == SD_FIT_ERROR);
        GGML_ASSERT(!GGMLRunner::measure_mode_enabled() && !sd_get_metadata_only_read());
    }

}  // namespace upscaler_fit_test

int main() {
    ggml_time_init();
    const auto path           = std::filesystem::temp_directory_path() / "sd-test-upscaler-fit.safetensors";
    const size_t params_bytes = upscaler_fit_test::write_model(path);
    upscaler_fit_test::test_measurement(path.string(), params_bytes);
    std::filesystem::remove(path);
    return 0;
}
