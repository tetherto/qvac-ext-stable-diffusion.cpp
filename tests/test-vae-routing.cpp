#include "core/ggml_extend_backend.h"
#include "vae_fallback.hpp"

#include <cassert>
#include <cstddef>
#include <limits>

namespace {

    constexpr size_t MIB = 1024ull * 1024ull;

}  // namespace

int main() {
    SDBackendAssignment runtime;
    runtime.set_default("vulkan0");
    runtime.set_module(SDBackendModule::VAE, "metal");
    runtime.apply_keep_cpu_overrides(false, true, false);
    assert(runtime.get(SDBackendModule::VAE) == "cpu");
    assert(runtime.get(SDBackendModule::DIFFUSION) == "vulkan0");
    assert(runtime.get(SDBackendModule::TE) == "vulkan0");

    SDBackendAssignment all_cpu_overrides;
    all_cpu_overrides.set_default("vulkan0");
    all_cpu_overrides.apply_keep_cpu_overrides(true, true, true);
    assert(all_cpu_overrides.get(SDBackendModule::TE) == "cpu");
    assert(all_cpu_overrides.get(SDBackendModule::VAE) == "cpu");
    assert(all_cpu_overrides.get(SDBackendModule::CONTROL_NET) == "cpu");
    assert(all_cpu_overrides.get(SDBackendModule::DIFFUSION) == "vulkan0");

    SDBackendAssignment fallback_params;
    fallback_params.set_module(SDBackendModule::DIFFUSION, "vulkan0");
    assert(fallback_params.set_module_if_unassigned(SDBackendModule::VAE, "cpu"));
    assert(fallback_params.get(SDBackendModule::VAE) == "cpu");
    assert(fallback_params.get(SDBackendModule::DIFFUSION) == "vulkan0");

    SDBackendAssignment explicit_params;
    explicit_params.set_default("vulkan0");
    assert(!explicit_params.set_module_if_unassigned(SDBackendModule::VAE, "cpu"));
    assert(explicit_params.get(SDBackendModule::VAE) == "vulkan0");

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
    assert(oversized.use_cpu_fallback());
    assert(oversized.reason == sd::VaeGraphRouteReason::EXCEEDS_CAPACITY);
    assert(oversized.budget_bytes == 4096 * MIB);

    const auto within_capacity = sd::select_vae_graph_route(
        2ull * 1024ull * MIB,
        four_gib,
        true,
        false,
        true,
        false);
    assert(!within_capacity.use_cpu_fallback());
    assert(within_capacity.reason == sd::VaeGraphRouteReason::WITHIN_CAPACITY);

    const auto stateful_within_capacity = sd::select_vae_graph_route(
        2ull * 1024ull * MIB,
        four_gib,
        true,
        false,
        true,
        true);
    assert(stateful_within_capacity.reason == sd::VaeGraphRouteReason::WITHIN_CAPACITY);

    const auto disabled = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        false,
        false,
        true,
        false);
    assert(disabled.reason == sd::VaeGraphRouteReason::FALLBACK_DISABLED);

    const auto explicit_cpu = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        true,
        true,
        true,
        false);
    assert(explicit_cpu.reason == sd::VaeGraphRouteReason::RUNTIME_ALREADY_CPU);
    assert(!explicit_cpu.use_cpu_fallback());

    const auto no_cpu_params = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        true,
        false,
        false,
        false);
    assert(no_cpu_params.reason == sd::VaeGraphRouteReason::CPU_PARAMS_UNAVAILABLE);

    const auto stateful = sd::select_vae_graph_route(
        oversized_217_frame_graph,
        four_gib,
        true,
        false,
        true,
        true);
    assert(stateful.reason == sd::VaeGraphRouteReason::STATEFUL_GRAPH);

    for (int i = 0; i < 100; ++i) {
        const auto repeated = sd::select_vae_graph_route(
            oversized_217_frame_graph,
            four_gib,
            true,
            false,
            true,
            false);
        assert(repeated.route == oversized.route);
        assert(repeated.reason == oversized.reason);
        assert(repeated.required_bytes == oversized.required_bytes);
        assert(repeated.budget_bytes == oversized.budget_bytes);
    }

    sd::VaeFallbackCapacity free_memory_limited;
    free_memory_limited.max_buffer_bytes  = std::numeric_limits<size_t>::max();
    free_memory_limited.free_memory_bytes = 4000 * MIB;
    free_memory_limited.free_memory_ratio = 0.8f;
    const size_t scaled_budget = sd::vae_fallback_budget(free_memory_limited);
    const size_t expected_budget = 3200 * MIB;
    const size_t budget_delta = scaled_budget > expected_budget
                                    ? scaled_budget - expected_budget
                                    : expected_budget - scaled_budget;
    // The API stores the ratio as float. AppleClang represents 0.8f slightly
    // above 0.8, which changes this multi-GiB product by 50 bytes.
    assert(budget_delta <= 64);

    return 0;
}
