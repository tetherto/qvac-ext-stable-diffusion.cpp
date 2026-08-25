#include "core/ggml_extend.hpp"
#include "core/ggml_extend_backend.h"
#include "vae_fallback.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
    constexpr size_t MIB = 1024ull * 1024ull;

    class RecordingWeightManager : public RunnerWeightManager {
    public:
        std::vector<ggml_backend_t> assignments;
        size_t fail_on_assignment = 0;

        bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                    ggml_backend_t compute_backend) override {
            GGML_ASSERT(!tensors.empty());
            assignments.push_back(compute_backend);
            return fail_on_assignment == 0 || assignments.size() != fail_on_assignment;
        }

        bool prepare_params(const std::vector<ggml_tensor*>&) override { return true; }
        void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
        void release_params_backend_params(const std::vector<ggml_tensor*>&) override {}
    };

    class RoutingTestRunner : public GGMLRunner {
    public:
        RoutingTestRunner(ggml_backend_t backend,
                          const std::shared_ptr<RunnerWeightManager>& manager)
            : GGMLRunner(backend, manager) {}

        std::string get_desc() override { return "routing_test"; }

        ggml_tensor* make_param() {
            ggml_tensor* tensor = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, 1);
            ggml_set_name(tensor, "routing_test.weight");
            return tensor;
        }

        bool assign_graph_params(const std::vector<ggml_tensor*>& tensors,
                                 ggml_backend_t backend,
                                 const char* action) {
            return assign_graph_params_compute_backend(tensors, backend, action);
        }
    };
}

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

    ggml_backend_t original_backend = sd_backend_cpu_init();
    ggml_backend_t fallback_backend = sd_backend_cpu_init();
    GGML_ASSERT(original_backend != nullptr);
    GGML_ASSERT(fallback_backend != nullptr);
    {
        auto manager = std::make_shared<RecordingWeightManager>();
        RoutingTestRunner runner(original_backend, manager);
        std::vector<ggml_tensor*> graph_params = {runner.make_param()};

        GGML_ASSERT(runner.assign_graph_params(graph_params, fallback_backend, "assign"));
        GGML_ASSERT(runner.assign_graph_params(graph_params, original_backend, "restore"));
        GGML_ASSERT(manager->assignments.size() == 2);
        GGML_ASSERT(manager->assignments[0] == fallback_backend);
        GGML_ASSERT(manager->assignments[1] == original_backend);

        manager->assignments.clear();
        manager->fail_on_assignment = 2;
        GGML_ASSERT(runner.assign_graph_params(graph_params, fallback_backend, "assign"));
        GGML_ASSERT(!runner.assign_graph_params(graph_params, original_backend, "restore"));
        GGML_ASSERT(manager->assignments.size() == 2);
    }
    ggml_backend_free(fallback_backend);
    ggml_backend_free(original_backend);

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

    sd::VaeFallbackCapacity pending_boundary_capacity = four_gib;
    pending_boundary_capacity.free_memory_bytes = 16ull * 1024ull * MIB;
    const auto pending_params_at_single_buffer_limit =
        sd::select_vae_graph_route(
            1,
            pending_boundary_capacity,
            true,
            false,
            true,
            false,
            pending_boundary_capacity.max_buffer_bytes);
    GGML_ASSERT(!pending_params_at_single_buffer_limit.use_cpu_fallback());
    GGML_ASSERT(pending_params_at_single_buffer_limit.reason ==
                sd::VaeGraphRouteReason::WITHIN_CAPACITY);

    const auto pending_params_exceed_single_buffer_limit =
        sd::select_vae_graph_route(
            1,
            pending_boundary_capacity,
            true,
            false,
            true,
            false,
            pending_boundary_capacity.max_buffer_bytes + 1);
    GGML_ASSERT(pending_params_exceed_single_buffer_limit.use_cpu_fallback());
    GGML_ASSERT(pending_params_exceed_single_buffer_limit.reason ==
                sd::VaeGraphRouteReason::EXCEEDS_CAPACITY);

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
