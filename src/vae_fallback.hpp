#ifndef __SD_VAE_FALLBACK_HPP__
#define __SD_VAE_FALLBACK_HPP__

#include <algorithm>
#include <cstddef>
#include <limits>

namespace sd {

    struct VaeFallbackCapacity {
        // SIZE_MAX means that GGML does not advertise a finite fixed logical-
        // buffer capacity for this backend.
        size_t max_buffer_bytes  = std::numeric_limits<size_t>::max();
        size_t free_memory_bytes = 0;
        float free_memory_ratio  = 0.9f;
    };

    enum class VaeGraphRoute {
        CONFIGURED_BACKEND,
        CPU_FALLBACK,
    };

    enum class VaeGraphRouteReason {
        FALLBACK_DISABLED,
        RUNTIME_ALREADY_CPU,
        CPU_PARAMS_UNAVAILABLE,
        STATEFUL_GRAPH,
        WITHIN_CAPACITY,
        EXCEEDS_CAPACITY,
    };

    struct VaeGraphRouteDecision {
        VaeGraphRoute route        = VaeGraphRoute::CONFIGURED_BACKEND;
        VaeGraphRouteReason reason = VaeGraphRouteReason::FALLBACK_DISABLED;
        size_t required_bytes      = 0;
        size_t budget_bytes        = std::numeric_limits<size_t>::max();

        bool use_cpu_fallback() const {
            return route == VaeGraphRoute::CPU_FALLBACK;
        }
    };

    inline size_t vae_fallback_scaled_budget(size_t bytes, float ratio) {
        const long double bounded_ratio = std::max<long double>(0.0L, std::min<long double>(1.0L, ratio));
        return static_cast<size_t>(static_cast<long double>(bytes) * bounded_ratio);
    }

    inline size_t vae_fallback_budget(const VaeFallbackCapacity& capacity) {
        size_t budget = capacity.max_buffer_bytes;
        if (capacity.free_memory_bytes > 0) {
            budget = std::min(budget,
                              vae_fallback_scaled_budget(capacity.free_memory_bytes,
                                                         capacity.free_memory_ratio));
        }
        return budget;
    }

    inline VaeGraphRouteDecision select_vae_graph_route(size_t required_bytes,
                                                        const VaeFallbackCapacity& capacity,
                                                        bool fallback_enabled,
                                                        bool runtime_is_cpu,
                                                        bool cpu_params_available,
                                                        bool has_stateful_cache) {
        VaeGraphRouteDecision decision;
        decision.required_bytes = required_bytes;
        decision.budget_bytes   = vae_fallback_budget(capacity);

        if (!fallback_enabled) {
            decision.reason = VaeGraphRouteReason::FALLBACK_DISABLED;
        } else if (runtime_is_cpu) {
            decision.reason = VaeGraphRouteReason::RUNTIME_ALREADY_CPU;
        } else if (!cpu_params_available) {
            decision.reason = VaeGraphRouteReason::CPU_PARAMS_UNAVAILABLE;
        } else if (required_bytes <= decision.budget_bytes) {
            decision.reason = VaeGraphRouteReason::WITHIN_CAPACITY;
        } else if (has_stateful_cache) {
            decision.reason = VaeGraphRouteReason::STATEFUL_GRAPH;
        } else {
            decision.route  = VaeGraphRoute::CPU_FALLBACK;
            decision.reason = VaeGraphRouteReason::EXCEEDS_CAPACITY;
        }
        return decision;
    }

}  // namespace sd

#endif
