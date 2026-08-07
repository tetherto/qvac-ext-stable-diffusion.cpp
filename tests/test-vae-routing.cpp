#include "ggml_extend.hpp"
#include "vae_fallback.hpp"

#include <cstddef>
#include <limits>
#include <string>

namespace {

    constexpr size_t MIB = 1024ull * 1024ull;

    struct TestRunner : public GGMLRunner {
        TestRunner(ggml_backend_t runtime, ggml_backend_t params)
            : GGMLRunner(runtime, params) {
        }

        std::string get_desc() override {
            return "test";
        }

        void add_mmap_backed_parameter() {
            ggml_tensor* tensor =
                ggml_new_tensor_2d(params_ctx, GGML_TYPE_F16, 128, 64);
            tensor->data = reinterpret_cast<void*>(1);
            rebuild_params_tensor_set();
        }
    };

}  // namespace

int main() {
    SDBackendManager compatibility_manager;
    std::string compatibility_error;
    GGML_ASSERT(compatibility_manager.init("cpu",
                                           "cpu",
                                           false,
                                           false,
                                           false,
                                           false,
                                           &compatibility_error));
    compatibility_manager.reset();

    ggml_backend_t runtime_cpu = sd_backend_cpu_init();
    ggml_backend_t params_cpu  = sd_backend_cpu_init();
    GGML_ASSERT(runtime_cpu != nullptr && params_cpu != nullptr);
    {
        TestRunner mmap_runner(runtime_cpu, params_cpu);
        mmap_runner.add_mmap_backed_parameter();
        GGML_ASSERT(mmap_runner.get_params_buffer_size() == 0);
        GGML_ASSERT(mmap_runner.get_pending_runtime_params_size() > 0);
    }
    ggml_backend_free(runtime_cpu);
    ggml_backend_free(params_cpu);

    SDBackendAssignment runtime;
    runtime.set_default("vulkan0");
    runtime.set_module(SDBackendModule::VAE, "metal");
    runtime.apply_keep_cpu_overrides(false, true, false);
    GGML_ASSERT(runtime.get(SDBackendModule::VAE) == "cpu");
    GGML_ASSERT(runtime.get(SDBackendModule::DIFFUSION) == "vulkan0");
    GGML_ASSERT(runtime.get(SDBackendModule::TE) == "vulkan0");

    SDBackendAssignment all_cpu_overrides;
    all_cpu_overrides.set_default("vulkan0");
    all_cpu_overrides.apply_keep_cpu_overrides(true, true, true);
    GGML_ASSERT(all_cpu_overrides.get(SDBackendModule::TE) == "cpu");
    GGML_ASSERT(all_cpu_overrides.get(SDBackendModule::VAE) == "cpu");
    GGML_ASSERT(all_cpu_overrides.get(SDBackendModule::CONTROL_NET) == "cpu");
    GGML_ASSERT(all_cpu_overrides.get(SDBackendModule::DIFFUSION) == "vulkan0");

    SDBackendAssignment fallback_params;
    fallback_params.set_module(SDBackendModule::DIFFUSION, "vulkan0");
    GGML_ASSERT(fallback_params.set_module_if_unassigned(SDBackendModule::VAE, "cpu"));
    GGML_ASSERT(fallback_params.get(SDBackendModule::VAE) == "cpu");
    GGML_ASSERT(fallback_params.get(SDBackendModule::DIFFUSION) == "vulkan0");

    SDBackendAssignment explicit_params;
    explicit_params.set_default("vulkan0");
    GGML_ASSERT(!explicit_params.set_module_if_unassigned(SDBackendModule::VAE, "cpu"));
    GGML_ASSERT(explicit_params.get(SDBackendModule::VAE) == "vulkan0");

    sd::VaeFallbackCapacity four_gib;
    four_gib.max_buffer_bytes  = 4096 * MIB;
    four_gib.free_memory_bytes = 4096 * MIB;
    four_gib.free_memory_ratio = 1.0f;

    constexpr size_t oversized_217_frame_graph = 14176870408ull;
    const auto oversized                       = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        true,
        false,
        true,
        false);
    GGML_ASSERT(oversized.use_cpu_fallback());
    GGML_ASSERT(oversized.reason == sd::VaeGraphRouteReason::EXCEEDS_CAPACITY);
    GGML_ASSERT(oversized.budget_bytes == 4096 * MIB);

    const auto within_capacity = sd::select_vae_graph_route(
        2ull * 1024ull * MIB,
        four_gib,
        true,
        false,
        true,
        false);
    GGML_ASSERT(!within_capacity.use_cpu_fallback());
    GGML_ASSERT(within_capacity.reason == sd::VaeGraphRouteReason::WITHIN_CAPACITY);

    const auto pending_params_exceed_free_memory =
        sd::select_vae_graph_route(
            700 * MIB,
            four_gib,
            true,
            false,
            true,
            false,
            3500 * MIB);
    GGML_ASSERT(pending_params_exceed_free_memory.use_cpu_fallback());
    GGML_ASSERT(pending_params_exceed_free_memory.compute_buffer_bytes ==
                700 * MIB);
    GGML_ASSERT(pending_params_exceed_free_memory.pending_runtime_param_bytes ==
                3500 * MIB);

    sd::VaeFallbackCapacity single_buffer_limited = four_gib;
    single_buffer_limited.free_memory_bytes       = 16ull * 1024ull * MIB;
    const auto single_buffer_exceeded             = sd::select_vae_graph_route(
        4096 * MIB + 1,
        single_buffer_limited,
        true,
        false,
        true,
        false,
        1);
    GGML_ASSERT(single_buffer_exceeded.use_cpu_fallback());

    const auto stateful_within_capacity = sd::select_vae_graph_route(
        2ull * 1024ull * MIB,
        four_gib,
        true,
        false,
        true,
        true);
    GGML_ASSERT(stateful_within_capacity.reason == sd::VaeGraphRouteReason::WITHIN_CAPACITY);

    const auto disabled = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        false,
        false,
        true,
        false);
    GGML_ASSERT(disabled.reason == sd::VaeGraphRouteReason::FALLBACK_DISABLED);

    const auto explicit_cpu = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        true,
        true,
        true,
        false);
    GGML_ASSERT(explicit_cpu.reason == sd::VaeGraphRouteReason::RUNTIME_ALREADY_CPU);
    GGML_ASSERT(!explicit_cpu.use_cpu_fallback());

    const auto no_cpu_params = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        true,
        false,
        false,
        false);
    GGML_ASSERT(no_cpu_params.reason == sd::VaeGraphRouteReason::CPU_PARAMS_UNAVAILABLE);

    const auto stateful = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        true,
        false,
        true,
        true);
    GGML_ASSERT(stateful.reason == sd::VaeGraphRouteReason::STATEFUL_GRAPH);

    for (int i = 0; i < 100; ++i) {
        const auto repeated = sd::select_vae_graph_route(
            oversized_217_frame_graph,
            four_gib,
            true,
            false,
            true,
            false);
        GGML_ASSERT(repeated.route == oversized.route);
        GGML_ASSERT(repeated.reason == oversized.reason);
        GGML_ASSERT(repeated.required_bytes == oversized.required_bytes);
        GGML_ASSERT(repeated.budget_bytes == oversized.budget_bytes);
    }

    sd::VaeFallbackCapacity free_memory_limited;
    free_memory_limited.max_buffer_bytes  = std::numeric_limits<size_t>::max();
    free_memory_limited.free_memory_bytes = 4000 * MIB;
    free_memory_limited.free_memory_ratio = 0.8f;
    const size_t scaled_budget            = sd::vae_fallback_budget(free_memory_limited);
    const size_t expected_budget          = 3200 * MIB;
    const size_t budget_delta             = scaled_budget > expected_budget
                                                ? scaled_budget - expected_budget
                                                : expected_budget - scaled_budget;
    // The API stores the ratio as float. AppleClang represents 0.8f slightly
    // above 0.8, which changes this multi-GiB product by 50 bytes.
    GGML_ASSERT(budget_delta <= 64);

    return 0;
}
