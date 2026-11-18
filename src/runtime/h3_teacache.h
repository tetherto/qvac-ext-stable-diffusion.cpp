#ifndef __SD_RUNTIME_H3_TEACACHE_H__
#define __SD_RUNTIME_H3_TEACACHE_H__

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace sd_sample {

    // Initial quality screening covers these 124-frame latent layouts only.
    // Other shapes and step counts keep the retained dense path.
    inline bool h3_shape_supported(int64_t width, int64_t height, int64_t frames, int steps) {
        return steps == 50 && frames == 37 &&
               ((width == 54 && height == 30) || (width == 84 && height == 48));
    }

    enum class H3CachePolicy { OFF,
                               CONTROL,
                               TEA,
                               SOL_CONTROL,
                               SOL_TEA };

    struct H3CacheConfig {
        H3CachePolicy policy    = H3CachePolicy::OFF;
        float threshold         = 0.1f;
        float audio_threshold   = 0.05f;
        int warmup_steps        = 43;
        int cooldown_steps      = 1;
        int max_consecutive     = 1;
        int max_reuses          = 1;
        int dense_prefix_layers = 4;
        int dense_suffix_layers = 24;
        int num_forwards        = 0;

        bool full_stack() const {
            return policy == H3CachePolicy::SOL_TEA || policy == H3CachePolicy::SOL_CONTROL;
        }

        bool control() const {
            return policy == H3CachePolicy::CONTROL || policy == H3CachePolicy::SOL_CONTROL;
        }

        const char* policy_name() const {
            switch (policy) {
                case H3CachePolicy::CONTROL:
                    return "control";
                case H3CachePolicy::TEA:
                    return "tea";
                case H3CachePolicy::SOL_CONTROL:
                    return "sol-control";
                case H3CachePolicy::SOL_TEA:
                    return "sol-tea";
                default:
                    return "off";
            }
        }

        static H3CacheConfig from_environment() {
            H3CacheConfig result;
            const char* policy = std::getenv("SD_H3_CACHE_POLICY");
            if (policy && std::strcmp(policy, "control") == 0)
                result.policy = H3CachePolicy::CONTROL;
            if (policy && std::strcmp(policy, "tea") == 0)
                result.policy = H3CachePolicy::TEA;
            if (policy && std::strcmp(policy, "sol-control") == 0)
                result.policy = H3CachePolicy::SOL_CONTROL;
            if (policy && std::strcmp(policy, "sol-tea") == 0)
                result.policy = H3CachePolicy::SOL_TEA;
            const bool full_stack = result.full_stack();
            if (full_stack) {
                result.warmup_steps        = 5;
                result.cooldown_steps      = 1;
                result.max_consecutive     = 0;
                result.dense_prefix_layers = result.dense_suffix_layers = 0;
            }
            if (const char* threshold = std::getenv("SD_H3_CACHE_THRESHOLD")) {
                char* end         = nullptr;
                errno             = 0;
                const float value = std::strtof(threshold, &end);
                if (errno == ERANGE || end == threshold || *end || !std::isfinite(value) || value < 0.f || value > 1.f) {
                    result.policy = H3CachePolicy::OFF;
                } else {
                    result.threshold = value;
                }
            }
            const char* audio_threshold = std::getenv("SD_H3_CACHE_AUDIO_THRESHOLD");
            if (!full_stack && audio_threshold) {
                const char* threshold = audio_threshold;
                char* end             = nullptr;
                errno                 = 0;
                const float value     = std::strtof(threshold, &end);
                if (errno == ERANGE || end == threshold || *end || !std::isfinite(value) || value < 0.f || value > 1.f) {
                    result.policy = H3CachePolicy::OFF;
                } else {
                    result.audio_threshold = value;
                }
            }
            auto integer_option = [&](const char* name, int& target, int minimum, int maximum) {
                if (const char* count = std::getenv(name)) {
                    char* end        = nullptr;
                    errno            = 0;
                    const long value = std::strtol(count, &end, 10);
                    if (errno == ERANGE || end == count || *end || value < minimum || value > maximum) {
                        result.policy = H3CachePolicy::OFF;
                    } else {
                        target = static_cast<int>(value);
                    }
                }
            };
            integer_option("SD_H3_CACHE_WARMUP_STEPS", result.warmup_steps, full_stack ? 0 : 1, 200);
            integer_option("SD_H3_CACHE_COOLDOWN_STEPS", result.cooldown_steps, full_stack ? 0 : 1, 200);
            if (full_stack) {
                integer_option("SD_H3_CACHE_NUM_FORWARDS", result.num_forwards, 0, 200);
            } else {
                integer_option("SD_H3_CACHE_MAX_REUSES", result.max_reuses, 0, 200);
                integer_option("SD_H3_CACHE_DENSE_PREFIX_LAYERS", result.dense_prefix_layers, 1, 49);
                integer_option("SD_H3_CACHE_DENSE_SUFFIX_LAYERS", result.dense_suffix_layers, 1, 49);
            }
            integer_option("SD_H3_CACHE_MAX_CONSECUTIVE", result.max_consecutive, full_stack ? 0 : 1, full_stack ? 200 : 4);
            return result;
        }

        bool has_cached_layers(int64_t layers) const {
            if (full_stack())
                return layers > 0 && dense_prefix_layers == 0 && dense_suffix_layers == 0;
            return dense_prefix_layers > 0 && dense_suffix_layers > 0 &&
                   int64_t(dense_prefix_layers) + dense_suffix_layers < layers;
        }
    };

    struct H3CacheEntry {
        size_t slot              = 0;
        uint64_t signature       = 0;
        int last_step            = -1;
        int consecutive          = 0;
        int computed             = 0;
        int reused               = 0;
        bool has_probe           = false;
        bool has_residual        = false;
        bool invalidated         = false;
        double accumulator       = 0.0;
        double audio_accumulator = 0.0;
    };

    struct H3CacheDecision {
        bool reuse         = false;
        const char* reason = "initialize";
        std::array<double, 3> relative_l1{{0., 0., 0.}};
        double accumulator       = 0.;
        double audio_accumulator = 0.;
    };

    struct H3TeaCacheState {
        H3CacheConfig config;
        int total_steps    = 0;
        int computed_calls = 0;
        int reused_calls   = 0;
        std::unordered_map<const void*, H3CacheEntry> entries;
        // The request owns the backend buffers. Destruction releases them, so
        // a reused model instance cannot carry residuals into another request.
        std::shared_ptr<void> device_store;

        bool enabled() const { return config.policy != H3CachePolicy::OFF; }
        int forward_count() const { return config.num_forwards > 0 ? config.num_forwards : total_steps; }
        bool uses_probe() const {
            if (config.full_stack())
                return forward_count() <= total_steps &&
                       (config.control() || (config.threshold > 0.f &&
                                             config.warmup_steps < forward_count() - config.cooldown_steps));
            return config.policy == H3CachePolicy::CONTROL ||
                   (config.policy == H3CachePolicy::TEA && config.threshold > 0.f && config.audio_threshold > 0.f && config.max_reuses > 0 &&
                    config.warmup_steps < total_steps - config.cooldown_steps);
        }

        bool needs_step_probe(int step) const {
            if (config.full_stack())
                return uses_probe();
            return config.policy == H3CachePolicy::CONTROL ||
                   (uses_probe() && step >= config.warmup_steps - 1 &&
                    step < total_steps - config.cooldown_steps && reused_calls < config.max_reuses);
        }

        H3CacheDecision dense_refresh(H3CacheEntry& entry) {
            entry.has_probe = entry.has_residual = false;
            entry.accumulator = entry.audio_accumulator = 0.;
            entry.consecutive                           = 0;
            ++entry.computed;
            ++computed_calls;
            H3CacheDecision result;
            result.reason = "forced_full";
            return result;
        }

        H3CacheEntry& prepare(const void* condition, uint64_t signature, int step) {
            auto found = entries.find(condition);
            if (found == entries.end()) {
                H3CacheEntry fresh;
                fresh.slot = entries.size();
                found      = entries.emplace(condition, fresh).first;
            }
            H3CacheEntry& entry    = found->second;
            const bool had_history = entry.last_step >= 0;
            entry.invalidated      = false;
            if (entry.signature != signature || (entry.last_step >= 0 && step != entry.last_step + 1)) {
                const size_t slot = entry.slot;
                entry             = H3CacheEntry{};
                entry.slot        = slot;
                entry.signature   = signature;
                entry.invalidated = had_history;
            }
            entry.last_step = step;
            return entry;
        }

        bool eligible(const H3CacheEntry& entry, int step) const {
            if (config.full_stack())
                return config.policy == H3CachePolicy::SOL_TEA && step >= config.warmup_steps &&
                       step < forward_count() - config.cooldown_steps && entry.has_residual &&
                       (config.max_consecutive == 0 || entry.consecutive < config.max_consecutive);
            return config.policy == H3CachePolicy::TEA && total_steps >= 10 && step >= config.warmup_steps &&
                   step < total_steps - config.cooldown_steps &&
                   entry.has_residual && entry.consecutive < config.max_consecutive &&
                   reused_calls < config.max_reuses;
        }

        H3CacheDecision decide(H3CacheEntry& entry, int step, const std::array<float, 6>& sums) {
            if (config.full_stack())
                return decide_full_stack(entry, step, sums);
            H3CacheDecision result;
            bool finite = true;
            for (int i = 0; i < 3; ++i) {
                const double numerator = sums[2 * i], denominator = sums[2 * i + 1];
                finite = finite && std::isfinite(numerator) && std::isfinite(denominator) &&
                         numerator >= 0. && denominator > 1e-8;
                result.relative_l1[i] = numerator / std::max(denominator, 1e-8);
            }
            const double rate = *std::max_element(result.relative_l1.begin(), result.relative_l1.end());
            finite            = finite && std::isfinite(rate);
            if (entry.has_probe && finite) {
                entry.accumulator += rate;
                entry.audio_accumulator += result.relative_l1[1];
            }
            if (!entry.has_probe || !entry.has_residual)
                result.reason = "initialize";
            else if (config.policy == H3CachePolicy::CONTROL)
                result.reason = "control";
            else if (!eligible(entry, step))
                result.reason = "forced_full";
            else if (!finite || !std::isfinite(entry.accumulator) || !std::isfinite(entry.audio_accumulator))
                result.reason = "nonfinite_probe";
            else if (config.audio_threshold <= 0.f || entry.audio_accumulator >= config.audio_threshold)
                result.reason = "audio_threshold";
            else if (config.threshold > 0.f && entry.accumulator < config.threshold) {
                result.reuse  = true;
                result.reason = "below_threshold";
            } else
                result.reason = "threshold";
            result.accumulator       = entry.accumulator;
            result.audio_accumulator = entry.audio_accumulator;
            entry.has_probe          = finite;
            if (result.reuse) {
                ++entry.reused;
                ++reused_calls;
                ++entry.consecutive;
            } else {
                entry.accumulator       = 0.;
                entry.audio_accumulator = 0.;
                entry.consecutive       = 0;
                ++entry.computed;
                ++computed_calls;
            }
            return result;
        }

        H3CacheDecision decide_full_stack(H3CacheEntry& entry, int step, const std::array<float, 6>& sums) {
            H3CacheDecision result;
            for (int i = 0; i < 3; ++i)
                result.relative_l1[i] = double(sums[2 * i]) / std::max(double(sums[2 * i + 1]), 1e-8);
            const bool finite = std::isfinite(sums[4]) && std::isfinite(sums[5]) &&
                                sums[4] >= 0.f && sums[5] >= 0.f && std::isfinite(result.relative_l1[2]);
            if (config.control())
                result.reason = "control";
            else if (step < config.warmup_steps)
                result.reason = "warmup";
            else if (step >= forward_count() - config.cooldown_steps)
                result.reason = "cooldown";
            else if (!entry.has_probe || !entry.has_residual)
                result.reason = "initialize";
            else if (!finite)
                result.reason = "nonfinite_probe";
            else if (config.max_consecutive > 0 && entry.consecutive >= config.max_consecutive)
                result.reason = "consecutive_limit";
            else {
                entry.accumulator += result.relative_l1[2];
                if (!std::isfinite(entry.accumulator))
                    result.reason = "nonfinite_probe";
                else if (entry.accumulator < config.threshold) {
                    result.reuse  = true;
                    result.reason = "below_threshold";
                } else
                    result.reason = "threshold";
            }
            result.accumulator = entry.accumulator;
            entry.has_probe    = finite;
            if (result.reuse) {
                ++entry.reused;
                ++reused_calls;
                ++entry.consecutive;
            } else {
                entry.accumulator = entry.audio_accumulator = 0.;
                entry.consecutive                           = 0;
                ++entry.computed;
                ++computed_calls;
            }
            return result;
        }
    };

}  // namespace sd_sample
#endif
