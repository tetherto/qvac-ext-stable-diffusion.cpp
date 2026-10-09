#include "abot_world.hpp"
#include "core/fit_params.h"
#include "json.hpp"
#include "model_io/binary_io.h"

#include <filesystem>
#include <fstream>

namespace abot_fit_test {

    void write_header(const std::filesystem::path& path, const nlohmann::json& header) {
        const auto text = header.dump();
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        model_io::write_u64(file, text.size());
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        GGML_ASSERT(file.good());
    }

    void write_model(const std::filesystem::path& path, const std::map<std::string, ggml_tensor*>& tensors) {
        nlohmann::json header;
        size_t offset = 0;
        for (const auto& entry : tensors) {
            std::vector<int64_t> shape;
            for (int i = ggml_n_dims(entry.second) - 1; i >= 0; --i) {
                shape.push_back(entry.second->ne[i]);
            }
            const size_t bytes  = ggml_nelements(entry.second) * sizeof(float);
            header[entry.first] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {offset, offset + bytes}}};
            offset += bytes;
        }
        write_header(path, header);
    }

    void set_host_memory(const char* gib) {
#if defined(_WIN32)
        GGML_ASSERT(_putenv_s("SD_FIT_DEBUG_HOST_MEMORY_GIB", gib) == 0);
#else
        GGML_ASSERT(setenv("SD_FIT_DEBUG_HOST_MEMORY_GIB", gib, 1) == 0);
#endif
    }

    size_t host_buffers(const sd_fit_result_t& result) {
        const char* marker = std::strstr(result.report, "session host buffers: ");
        GGML_ASSERT(marker);
        return static_cast<size_t>(std::strtoull(marker + std::strlen("session host buffers: "), nullptr, 10));
    }

    void test_fit(const std::string& dit, const std::string& tae, const std::string& scene) {
        sd_abot_session_params_v2_t params;
        sd_abot_session_params_v2_init(&params);
        params.dit_model_path = dit.c_str();
        params.taehv_path     = tae.c_str();
        params.scene_path     = scene.c_str();
        params.backend        = "cpu";
        sd_abot_fit_workload_t workload;
        sd_abot_fit_workload_init(&workload);
        GGML_ASSERT(workload.walk_steps == 100);
        sd_fit_result_t result{};
        set_host_memory("256");
        GGML_ASSERT(sd_abot_fit_params(&params, &workload, &result) == SD_FIT_SUCCESS);
        GGML_ASSERT(result.report && std::strstr(result.report, "100 walk steps"));
        GGML_ASSERT(std::strstr(result.report, "diffusion: params") && std::strstr(result.report, "vae: params"));
        GGML_ASSERT(!result.changed);
        sd_fit_result_free(&result);
        params.kv_cache = true;
        GGML_ASSERT(sd_abot_fit_params(&params, &workload, &result) == SD_FIT_SUCCESS);
        const size_t long_host = host_buffers(result);
        sd_fit_result_free(&result);
        workload.walk_steps = 1;
        GGML_ASSERT(sd_abot_fit_params(&params, &workload, &result) == SD_FIT_SUCCESS);
        GGML_ASSERT(long_host > host_buffers(result));
        sd_fit_result_free(&result);
        set_host_memory("0.6");
        GGML_ASSERT(sd_abot_fit_params(&params, &workload, &result) == SD_FIT_FAILURE);
        sd_fit_result_free(&result);
        workload.walk_steps = 0;
        GGML_ASSERT(sd_abot_fit_params(&params, &workload, &result) == SD_FIT_ERROR);
        workload.walk_steps = 1000001;
        GGML_ASSERT(sd_abot_fit_params(&params, &workload, &result) == SD_FIT_ERROR);
        GGML_ASSERT(!GGMLRunner::measure_mode_enabled() && !sd_get_metadata_only_read());
        workload.walk_steps        = 1;
        params.kv_cache            = false;
        params.num_frame_per_block = 64;
        set_host_memory("256");
        GGML_ASSERT(sd_abot_fit_params(&params, &workload, &result) == SD_FIT_SUCCESS);
        sd_fit_result_free(&result);
        set_host_memory("");
        ABOT::AbotScenePack strict;
        GGML_ASSERT(!strict.load(scene));
    }

    void test_decode_overlap(const std::string& dit, const std::string& tae, const std::string& scene) {
        SDMetadataOnlyReadScope metadata;
        for (const auto& block : std::vector<std::array<int, 3>>{{1, 1, 1}, {1, 2, 2}, {1, 4, 4}, {2, 2, 4}, {2, 3, 5}}) {
            SDBackendHandle backend(sd_backend_cpu_init());
            GGML_ASSERT(backend);
            ABOT::AbotWorldConfig config;
            config.num_frame_per_block = block[0];
            ABOT::AbotWalkSession session;
            GGML_ASSERT(session.load(backend.get(), backend.get(), backend.get(), backend.get(), dit, tae, scene, config, 42, 1));
            std::vector<GGMLRunner::graph_memory_measurement> records;
            GGMLRunner::set_measure_mode(true, &records);
            size_t host = 0;
            GGML_ASSERT(session.measure_memory(block[1], records, host));
            size_t estimated = 0;
            for (const auto& record : records) {
                if (record.module == SDBackendModule::VAE) {
                    estimated = std::max(estimated, record.compute_bytes);
                }
            }
            records.clear();
            GGML_ASSERT(session.tae->measure_decode(2, 2, block[2], 1));
            GGML_ASSERT(!records.empty() && estimated == records.back().compute_bytes);
            GGMLRunner::set_measure_mode(false);
        }
    }

    void test_real_model(const char* dit, const char* tae, const char* scene, const char* backend, const char* budget) {
        sd_abot_session_params_v2_t params;
        sd_abot_session_params_v2_init(&params);
        params.dit_model_path        = dit;
        params.taehv_path            = tae;
        params.scene_path            = scene;
        params.backend               = backend;
        params.offload_params_to_cpu = true;
        params.max_vram              = budget;
        params.stream_layers         = budget != nullptr;
        sd_abot_fit_workload_t workload;
        sd_abot_fit_workload_init(&workload);
        for (bool kv : {false, true}) {
            params.kv_cache = kv;
            sd_fit_result_t result{};
            const auto status = sd_abot_fit_params(&params, &workload, &result);
            GGML_ASSERT(status != SD_FIT_ERROR && result.report != nullptr);
            std::printf("%s\n", result.report);
            sd_fit_result_free(&result);
            if (budget != nullptr) {
                params.params_backend  = "diffusion=disk";
                const auto disk_status = sd_abot_fit_params(&params, &workload, &result);
                GGML_ASSERT(disk_status != SD_FIT_ERROR && result.report != nullptr);
                GGML_ASSERT(disk_status == status);
                std::printf("disk placement: %s\n", result.report);
                sd_fit_result_free(&result);
                params.params_backend = nullptr;
            }
        }
    }
}

int main(int argc, char** argv) {
    sd_set_log_callback([](sd_log_level_t, const char* text, void*) { std::fputs(text, stderr); }, nullptr);
    ggml_backend_load_all();
    if (argc == 5 || argc == 6) {
        abot_fit_test::test_real_model(argv[1], argv[2], argv[3], argv[4], argc == 6 ? argv[5] : nullptr);
        return 0;
    }
    GGML_ASSERT(argc == 1);
    SDBackendHandle backend(sd_backend_cpu_init());
    GGML_ASSERT(backend);
    const auto dir = std::filesystem::temp_directory_path() / "sd-abot-fit-test";
    std::filesystem::create_directories(dir);
    const auto dit = dir / "dit.safetensors", tae = dir / "tae.safetensors", scene = dir / "scene.safetensors";
    {
        ABOT::AbotWorldRunner runner(backend.get(), backend.get(), {}, "model.diffusion_model");
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors, "model.diffusion_model");
        GGML_ASSERT(!tensors.empty());
        abot_fit_test::write_model(dit, tensors);
    }
    {
        ABOT::AbotTinyVideoAutoEncoder runner(backend.get(), {}, "decoder", true, VERSION_ABOT_WORLD);
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors);
        GGML_ASSERT(!tensors.empty());
        abot_fit_test::write_model(tae, tensors);
    }
    nlohmann::json header;
    size_t offset = 0;
    for (const auto& entry : std::vector<std::pair<std::string, std::vector<int64_t>>>{
             {"prompt_embeds", {1, 512, 4096}}, {"first_frame_latents", {1, 1, 48, 2, 2}}, {"ref_latents", {1, 1, 48, 1, 32, 32}}, {"ref_mask", {1, 1}}}) {
        size_t bytes = sizeof(float);
        for (auto dim : entry.second) {
            bytes *= dim;
        }
        header[entry.first] = {{"dtype", "F32"}, {"shape", entry.second}, {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    abot_fit_test::write_header(scene, header);
    abot_fit_test::test_fit(dit.string(), tae.string(), scene.string());
    abot_fit_test::test_decode_overlap(dit.string(), tae.string(), scene.string());
    {
        SDMetadataOnlyReadScope metadata;
        ABOT::AbotScenePack pack;
        GGML_ASSERT(pack.load(scene.string()));
        GGML_ASSERT(pack.prompt_embeds.empty() && pack.first_frame_latents.empty() && pack.ref_latents.empty());
        for (const auto& invalid : std::vector<nlohmann::json>{
                 {{"dtype", "F16"}, {"shape", {1, 512, 4096}}, {"data_offsets", {0, 8388608}}},
                 {{"dtype", "F32"}, {"shape", {1, 512, 4096}}, {"data_offsets", {-1, 8388607}}},
                 {{"dtype", "F32"}, {"shape", {1, 512, 4096}}, {"data_offsets", {8, 8388608}}},
                 {{"dtype", "F32"}, {"shape", {1, 512.5, 4096}}, {"data_offsets", {0, 8388608}}}}) {
            auto bad             = header;
            bad["prompt_embeds"] = invalid;
            abot_fit_test::write_header(scene, bad);
            ABOT::AbotScenePack rejected;
            GGML_ASSERT(!rejected.load(scene.string()));
        }
    }
    std::filesystem::remove(dit);
    std::filesystem::remove(tae);
    std::filesystem::remove(scene);
    std::filesystem::remove(dir);
    return 0;
}
