#include "lora.hpp"

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
    {
        std::ofstream(empty_path, std::ios::binary);
    }
    {
        std::ofstream corrupt(corrupt_path, std::ios::binary);
        corrupt << "not a safetensors or gguf adapter";
    }
    {
        LoraModel empty("empty", cpu, cpu, empty_path);
        GGML_ASSERT(!empty.load_from_file(1));
        LoraModel corrupt("corrupt", cpu, cpu, corrupt_path);
        GGML_ASSERT(!corrupt.load_from_file(1));
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

    ggml_free(ctx);
    std::remove(empty_path.c_str());
    std::remove(corrupt_path.c_str());
    ggml_backend_free(cpu);
    return 0;
}
