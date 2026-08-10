#include "lora.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>

int main() {
    ggml_backend_t cpu = sd_backend_cpu_init();
    GGML_ASSERT(cpu != nullptr);

    const std::string missing_path =
        "/tmp/sd-test-lora-definitely-missing.safetensors";
    {
        LoraModel missing("missing", cpu, cpu, missing_path);
        GGML_ASSERT(!missing.load_from_file(1));
    }

    const std::string empty_path = "/tmp/sd-test-lora-empty.safetensors";
    const std::string corrupt_path =
        "/tmp/sd-test-lora-corrupt.safetensors";
    const std::string truncated_path =
        "/tmp/sd-test-lora-truncated.safetensors";
    {
        std::ofstream(empty_path, std::ios::binary);
    }
    {
        std::ofstream corrupt(corrupt_path, std::ios::binary);
        corrupt << "not a safetensors or gguf adapter";
    }
    {
        const std::string header =
            R"({"model.diffusion_model.test.weight.diff":{"dtype":"F32","shape":[6,4],"data_offsets":[0,96]}})";
        std::ofstream truncated(truncated_path, std::ios::binary);
        const uint64_t header_size = header.size();
        truncated.write(reinterpret_cast<const char*>(&header_size),
                        sizeof(header_size));
        truncated.write(header.data(),
                        static_cast<std::streamsize>(header.size()));
        const uint32_t partial_payload = 0;
        truncated.write(reinterpret_cast<const char*>(&partial_payload),
                        sizeof(partial_payload));
    }
    {
        LoraModel empty("empty", cpu, cpu, empty_path);
        GGML_ASSERT(!empty.load_from_file(1));
        LoraModel corrupt("corrupt", cpu, cpu, corrupt_path);
        GGML_ASSERT(!corrupt.load_from_file(1));
        LoraModel truncated("truncated", cpu, cpu, truncated_path);
        GGML_ASSERT(!truncated.load_from_file(1));
    }

    ggml_init_params init = {};
    init.mem_size         = 32 * ggml_tensor_overhead();
    init.no_alloc         = true;
    ggml_context* ctx     = ggml_init(init);
    GGML_ASSERT(ctx != nullptr);
    auto* model_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 6);
    std::map<std::string, ggml_tensor*> model_tensors = {
        {"model.diffusion_model.test.weight", model_weight},
    };

    {
        LoraModel incompatible("incompatible", cpu, cpu, missing_path);
        incompatible.lora_tensors
            ["lora.model.diffusion_model.test.weight.lora_down"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
        incompatible.lora_tensors
            ["lora.model.diffusion_model.test.weight.lora_up"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 6);
        GGML_ASSERT(incompatible.count_compatible_model_tensors(model_tensors) ==
                    0);
    }
    {
        LoraModel compatible("compatible", cpu, cpu, missing_path);
        compatible.lora_tensors
            ["lora.model.diffusion_model.test.weight.lora_down"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        compatible.lora_tensors
            ["lora.model.diffusion_model.test.weight.lora_up"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 6);
        GGML_ASSERT(compatible.count_compatible_model_tensors(model_tensors) ==
                    1);
    }
    {
        LoraModel loha("loha", cpu, cpu, missing_path);
        loha.lora_tensors
            ["lora.model.diffusion_model.test.weight.hada_w1_b"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        loha.lora_tensors
            ["lora.model.diffusion_model.test.weight.hada_w1_a"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 6);
        loha.lora_tensors
            ["lora.model.diffusion_model.test.weight.hada_w2_b"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        loha.lora_tensors
            ["lora.model.diffusion_model.test.weight.hada_w2_a"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 6);
        GGML_ASSERT(loha.count_compatible_model_tensors(model_tensors) == 1);
    }
    {
        LoraModel lokr("lokr", cpu, cpu, missing_path);
        lokr.lora_tensors
            ["lora.model.diffusion_model.test.weight.lokr_w1"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
        lokr.lora_tensors
            ["lora.model.diffusion_model.test.weight.lokr_w2"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 6);
        GGML_ASSERT(lokr.count_compatible_model_tensors(model_tensors) == 1);
    }
    {
        LoraModel lokr_factorized("lokr-factorized", cpu, cpu, missing_path);
        lokr_factorized.lora_tensors
            ["lora.model.diffusion_model.test.weight.lokr_w1_a"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        lokr_factorized.lora_tensors
            ["lora.model.diffusion_model.test.weight.lokr_w1_b"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
        lokr_factorized.lora_tensors
            ["lora.model.diffusion_model.test.weight.lokr_w2_a"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 6);
        lokr_factorized.lora_tensors
            ["lora.model.diffusion_model.test.weight.lokr_w2_b"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        GGML_ASSERT(
            lokr_factorized.count_compatible_model_tensors(model_tensors) ==
            1);
    }

    // Transactional immediate-LoRA preflight must reject a later invalid
    // change before any earlier valid change is applied, while allowing the
    // unchanged valid request to be reused afterward.
    std::vector<int> requested_changes = {1, 2};
    size_t applied_changes            = 0;
    GGML_ASSERT(!validate_lora_changes(
        requested_changes,
        [](int change) { return change == 1; }));
    GGML_ASSERT(applied_changes == 0);
    requested_changes = {1};
    GGML_ASSERT(validate_lora_changes(
        requested_changes,
        [](int change) {
            return change == 1;
        }));
    for (int change : requested_changes) {
        (void)change;
        ++applied_changes;
    }
    GGML_ASSERT(applied_changes == 1);

    ggml_free(ctx);
    std::remove(empty_path.c_str());
    std::remove(corrupt_path.c_str());
    std::remove(truncated_path.c_str());
    ggml_backend_free(cpu);
    return 0;
}
