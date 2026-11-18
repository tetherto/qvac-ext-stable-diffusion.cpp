#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include "runtime/h3_teacache.h"

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void environment(const char* name, const char* value = nullptr) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

int main() {
    try {
        using namespace sd_sample;
        environment("SD_H3_CACHE_POLICY");
        environment("SD_H3_CACHE_THRESHOLD");
        environment("SD_H3_CACHE_AUDIO_THRESHOLD");
        environment("SD_H3_CACHE_DENSE_PREFIX_LAYERS");
        environment("SD_H3_CACHE_DENSE_SUFFIX_LAYERS");
        environment("SD_H3_CACHE_MAX_REUSES");
        environment("SD_H3_CACHE_MAX_CONSECUTIVE");
        environment("SD_H3_CACHE_WARMUP_STEPS");
        environment("SD_H3_CACHE_COOLDOWN_STEPS");
        require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Default cache must be off");
        environment("SD_H3_CACHE_POLICY", "tea");
        for (const char* value : {"", "nan", "inf", "-1", "1.1", "0.01junk"}) {
            environment("SD_H3_CACHE_THRESHOLD", value);
            require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Bad threshold enabled reuse");
        }
        environment("SD_H3_CACHE_THRESHOLD", "0.01");
        for (const char* value : {"", "nan", "inf", "-1", "1.1", "0.01junk"}) {
            environment("SD_H3_CACHE_AUDIO_THRESHOLD", value);
            require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Bad audio threshold enabled reuse");
        }
        environment("SD_H3_CACHE_AUDIO_THRESHOLD");
        for (const char* name : {"SD_H3_CACHE_DENSE_PREFIX_LAYERS", "SD_H3_CACHE_DENSE_SUFFIX_LAYERS"}) {
            for (const char* value : {"", "0", "-1", "50", "bad"}) {
                environment(name, value);
                require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Invalid layer protection enabled reuse");
            }
            environment(name);
        }
        H3CacheConfig layer_config;
        require(layer_config.has_cached_layers(50), "Protected middle caching disabled");
        require(!layer_config.has_cached_layers(20) && !layer_config.has_cached_layers(19), "Empty/negative cached interval accepted");
        require(layer_config.max_reuses == 1, "Default reuse budget is not conservative");
        for (const char* value : {"", "-1", "201", "one", "9999999999999999999999999999"}) {
            environment("SD_H3_CACHE_MAX_REUSES", value);
            require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Bad limit enabled reuse");
        }
        environment("SD_H3_CACHE_MAX_REUSES");
        for (const char* value : {"", "0", "5", "bad", "9999999999999999999999999999"}) {
            environment("SD_H3_CACHE_MAX_CONSECUTIVE", value);
            require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Invalid consecutive limit enabled reuse");
        }
        environment("SD_H3_CACHE_MAX_CONSECUTIVE");
        require(H3CacheConfig::from_environment().policy == H3CachePolicy::TEA, "Valid opt-in configuration rejected");
        for (const char* name : {"SD_H3_CACHE_WARMUP_STEPS", "SD_H3_CACHE_COOLDOWN_STEPS"}) {
            for (const char* value : {"", "0", "201", "bad"}) {
                environment(name, value);
                require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Unprotected/invalid dense window accepted");
            }
            environment(name);
        }
        environment("SD_H3_CACHE_WARMUP_STEPS", "15");
        require(H3CacheConfig::from_environment().warmup_steps == 15, "Explicit late-only window rejected");
        environment("SD_H3_CACHE_WARMUP_STEPS");
        require(h3_shape_supported(54, 30, 37, 50) && h3_shape_supported(84, 48, 37, 50), "Candidate geometry disabled");
        for (int steps : {0, 20, 49, 51})
            require(!h3_shape_supported(54, 30, 37, steps), "Unscreened step count enabled approximation");
        require(!h3_shape_supported(32, 20, 10, 50) && !h3_shape_supported(54, 30, 36, 50), "Unscreened shape enabled approximation");
        H3TeaCacheState inactive;
        inactive.config.policy       = H3CachePolicy::TEA;
        inactive.total_steps         = 20;
        inactive.config.warmup_steps = 37;
        require(!inactive.uses_probe(), "Impossible cache window still allocates/probes");
        inactive.total_steps = 50;
        require(inactive.uses_probe(), "Eligible request disabled probing");
        inactive.config.threshold = 0.f;
        require(!inactive.uses_probe(), "Zero-threshold disabled cache still probes");
        inactive.config.threshold       = 0.1f;
        inactive.config.audio_threshold = 0.f;
        require(!inactive.uses_probe(), "Zero audio threshold still probes");
        inactive.config.policy = H3CachePolicy::CONTROL;
        require(inactive.uses_probe(), "Diagnostic control must still exercise probing");
        H3TeaCacheState deferred;
        deferred.total_steps   = 50;
        deferred.config.policy = H3CachePolicy::TEA;
        require(!deferred.needs_step_probe(0) && !deferred.needs_step_probe(41), "Warmup performs unused probes");
        require(deferred.needs_step_probe(42) && deferred.needs_step_probe(43), "Reuse lacks immediately preceding probe/residual");
        require(!deferred.needs_step_probe(49), "Cooldown performs unused probes");
        deferred.reused_calls = deferred.config.max_reuses;
        require(!deferred.needs_step_probe(44), "Exhausted budget performs unused probes");
        H3CacheEntry old_entry;
        old_entry.has_probe = old_entry.has_residual = true;
        old_entry.accumulator = old_entry.audio_accumulator = 0.5;
        old_entry.consecutive                               = 1;
        deferred.dense_refresh(old_entry);
        require(!old_entry.has_probe && !old_entry.has_residual && old_entry.accumulator == 0. &&
                    old_entry.audio_accumulator == 0. && old_entry.consecutive == 0 && deferred.computed_calls == 1,
                "Unprobed dense refresh kept stale reusable state");
        deferred.config.policy = H3CachePolicy::CONTROL;
        require(deferred.needs_step_probe(0) && deferred.needs_step_probe(49), "Exact control must exercise full split path");
        int a, b;
        const std::array<float, 6> small{{.001f, 1.f, .001f, 1.f, .001f, 1.f}};
        for (float threshold : {0.f, 0.01f}) {
            H3TeaCacheState bounded;
            bounded.total_steps            = 50;
            bounded.config.policy          = H3CachePolicy::TEA;
            bounded.config.threshold       = threshold;
            bounded.config.max_reuses      = 6;
            bounded.config.warmup_steps    = 5;
            bounded.config.max_consecutive = 2;
            std::vector<int> reused;
            for (int step = 0; step < 50; ++step) {
                auto& entry   = bounded.prepare(&a, 42, step);
                auto decision = bounded.decide(entry, step, small);
                if (decision.reuse)
                    reused.push_back(step);
                else
                    entry.has_residual = true;
            }
            require(threshold == 0.f ? reused.empty() : reused == std::vector<int>({5, 6, 8, 9, 11, 12}),
                    "Zero threshold or configurable consecutive/request reuse limit failed");
        }
        for (auto policy : {H3CachePolicy::CONTROL, H3CachePolicy::TEA}) {
            H3TeaCacheState state;
            state.total_steps         = 20;
            state.config.policy       = policy;
            state.config.warmup_steps = 5;
            std::vector<int> reused;
            for (int step = 0; step < 20; ++step) {
                auto& entry   = state.prepare(&a, 42, step);
                auto decision = state.decide(entry, step, small);
                if (decision.reuse)
                    reused.push_back(step);
                else
                    entry.has_residual = true;
            }
            require(policy == H3CachePolicy::CONTROL ? reused.empty() : reused == std::vector<int>({5}), "Warmup/cooldown/reuse bounds");
            require(state.computed_calls + state.reused_calls == 20, "Actual branch-call accounting");
        }
        H3TeaCacheState state;
        state.total_steps         = 20;
        state.config.policy       = H3CachePolicy::TEA;
        state.config.warmup_steps = 4;
        auto& first               = state.prepare(&a, 42, 3);
        first.has_probe = first.has_residual = true;
        auto& second                         = state.prepare(&b, 42, 3);
        require(!second.has_residual && second.slot != first.slot, "Condition branches share state");
        require(!state.decide(second, 4, small).reuse, "Cold branch reused another branch residual");
        auto& valid = state.prepare(&a, 42, 4);
        auto audio  = small;
        audio[2]    = .2f;
        require(!state.decide(valid, 4, audio).reuse, "Video-dominated packing hid audio change");
        H3TeaCacheState audio_guard;
        audio_guard.total_steps            = 50;
        audio_guard.config.policy          = H3CachePolicy::TEA;
        audio_guard.config.warmup_steps    = 1;
        audio_guard.config.threshold       = 0.1f;
        audio_guard.config.audio_threshold = 0.02f;
        audio_guard.config.max_reuses      = 3;
        audio_guard.config.max_consecutive = 3;
        auto& audio_entry                  = audio_guard.prepare(&a, 91, 1);
        audio_entry.has_probe = audio_entry.has_residual = true;
        auto guarded_stats                               = small;
        guarded_stats[2]                                 = 0.03f;
        auto veto                                        = audio_guard.decide(audio_entry, 1, guarded_stats);
        require(!veto.reuse && std::string(veto.reason) == "audio_threshold", "Joint threshold hid separate audio veto");
        guarded_stats[2] = 0.015f;
        require(audio_guard.decide(audio_guard.prepare(&a, 91, 2), 2, guarded_stats).reuse, "Safe audio probe rejected");
        veto = audio_guard.decide(audio_guard.prepare(&a, 91, 3), 3, guarded_stats);
        require(!veto.reuse && std::string(veto.reason) == "audio_threshold", "Audio change did not accumulate across reuses");
        require(audio_entry.audio_accumulator == 0., "Dense refresh did not reset audio error budget");
        auto& changed = state.prepare(&a, 43, 5);
        require(changed.invalidated && !changed.has_probe && !changed.has_residual, "Shape/condition signature not invalidated");
        changed.has_probe = changed.has_residual = true;
        auto& repeated                           = state.prepare(&a, 43, 5);
        require(repeated.invalidated && !repeated.has_residual, "Repeated/nonmonotonic scheduler step reused state");
        repeated.has_probe = repeated.has_residual = true;
        auto& gap                                  = state.prepare(&a, 43, 7);
        require(gap.invalidated && !gap.has_residual, "Non-adjacent scheduler step reused a stale residual");
        for (float bad : {0.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            auto& entry     = state.prepare(&a, 44, 4);
            entry.has_probe = entry.has_residual = true;
            auto stats                           = small;
            stats[3]                             = bad;
            require(!state.decide(entry, 4, stats).reuse, "Invalid/zero probe norm reused residual");
        }
        H3TeaCacheState fresh;
        require(fresh.entries.empty() && !fresh.device_store, "New request retained a cache");
        environment("SD_H3_CACHE_POLICY", "sol-tea");
        environment("SD_H3_CACHE_THRESHOLD", "0.1");
        environment("SD_H3_CACHE_NUM_FORWARDS", "49");
        environment("SD_H3_CACHE_MAX_REUSES", "1");
        environment("SD_H3_CACHE_MAX_CONSECUTIVE");
        environment("SD_H3_CACHE_AUDIO_THRESHOLD", "0");
        H3TeaCacheState sol;
        sol.config      = H3CacheConfig::from_environment();
        sol.total_steps = 50;
        require(sol.config.full_stack() && sol.config.has_cached_layers(50) && sol.config.warmup_steps == 5 &&
                    sol.config.cooldown_steps == 1 && sol.config.dense_prefix_layers == 0 && sol.config.dense_suffix_layers == 0,
                "SOL preset must cache the whole block stack after five computed forwards");
        require(sol.uses_probe() && sol.needs_step_probe(0) && sol.needs_step_probe(49), "SOL probe must track every forward");
        auto joint_small = small;
        joint_small[2]   = 0.5f;
        std::vector<int> sol_reused;
        for (int step = 0; step < 50; ++step) {
            auto& entry = sol.prepare(&a, 300, step);
            if (sol.decide(entry, step, joint_small).reuse)
                sol_reused.push_back(step);
            else
                entry.has_residual = true;
        }
        std::vector<int> sol_expected;
        for (int step = 5; step < 48; ++step)
            sol_expected.push_back(step);
        require(sol_reused == sol_expected && sol.computed_calls == 7 && sol.reused_calls == 43,
                "SOL incorrectly retained bounded-policy limits or an audio-only veto");
        require(sol.config.max_consecutive == 0, "SOL default must retain unlimited consecutive reuse");
        sol             = H3TeaCacheState{};
        sol.config      = H3CacheConfig::from_environment();
        sol.total_steps = 50;
        auto changing   = joint_small;
        changing[4]     = 0.04f;
        std::vector<int> computed;
        for (int step = 0; step <= 10; ++step) {
            auto& entry = sol.prepare(&a, 301, step);
            if (!sol.decide(entry, step, changing).reuse) {
                computed.push_back(step);
                entry.has_residual = true;
            }
        }
        require(computed == std::vector<int>({0, 1, 2, 3, 4, 7, 10}), "SOL accumulated-change refresh schedule differs");
        auto invalid    = changing;
        invalid[4]      = std::numeric_limits<float>::quiet_NaN();
        auto& bad_probe = sol.prepare(&a, 301, 11);
        require(!sol.decide(bad_probe, 11, invalid).reuse && !bad_probe.has_probe, "SOL reused a nonfinite probe");
        auto& recover = sol.prepare(&a, 301, 12);
        require(!sol.decide(recover, 12, changing).reuse, "SOL failed to reinitialize after invalid history");
        auto& new_condition = sol.prepare(&a, 302, 13);
        require(!sol.decide(new_condition, 13, changing).reuse, "SOL reused changed conditioning");
        for (const char* value : {"", "-1", "201", "bad", "9999999999999999999999999999"}) {
            environment("SD_H3_CACHE_MAX_CONSECUTIVE", value);
            require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Invalid SOL consecutive limit enabled caching");
        }
        environment("SD_H3_CACHE_MAX_CONSECUTIVE", "1");
        environment("SD_H3_CACHE_WARMUP_STEPS", "17");
        environment("SD_H3_CACHE_THRESHOLD", "0.05");
        sol             = H3TeaCacheState{};
        sol.config      = H3CacheConfig::from_environment();
        sol.total_steps = 50;
        require(sol.config.max_consecutive == 1, "Explicit SOL consecutive limit ignored");
        std::vector<int> limited_reused;
        auto limited_change = joint_small;
        limited_change[4]   = 0.02f;
        for (int step = 0; step < 50; ++step) {
            auto& entry             = sol.prepare(&a, 304, step);
            const bool must_refresh = entry.consecutive == 1 && step < 48;
            if (must_refresh)
                require(!sol.eligible(entry, step), "SOL eligibility ignored consecutive limit");
            const auto decision = sol.decide(entry, step, limited_change);
            if (decision.reuse) {
                limited_reused.push_back(step);
            } else {
                entry.has_residual = true;
                require(entry.accumulator == 0. && entry.consecutive == 0, "SOL refresh retained stale accumulator/count");
            }
            if (must_refresh)
                require(!decision.reuse && std::strcmp(decision.reason, "consecutive_limit") == 0,
                        "SOL did not compute immediately after one reused step");
        }
        std::vector<int> limited_expected;
        for (int step = 17; step < 48; step += 2)
            limited_expected.push_back(step);
        require(limited_reused == limited_expected && sol.reused_calls == 16 && sol.computed_calls == 34,
                "SOL limited reuse window/accounting differs");
        sol                  = H3TeaCacheState{};
        sol.config           = H3CacheConfig::from_environment();
        sol.config.threshold = 0.01f;
        sol.total_steps      = 50;
        for (int step = 0; step < 50; ++step) {
            auto& entry = sol.prepare(&a, 305, step);
            require(!sol.decide(entry, step, limited_change).reuse, "SOL cap bypassed tighter change threshold");
            entry.has_residual = true;
        }
        environment("SD_H3_CACHE_POLICY", "sol-control");
        sol             = H3TeaCacheState{};
        sol.config      = H3CacheConfig::from_environment();
        sol.total_steps = 50;
        for (int step = 0; step < 50; ++step) {
            auto& entry = sol.prepare(&a, 303, step);
            require(!sol.decide(entry, step, small).reuse, "SOL diagnostic control reused residual");
            entry.has_residual = true;
        }
        require(sol.computed_calls == 50 && sol.reused_calls == 0, "SOL control accounting");
        environment("SD_H3_CACHE_NUM_FORWARDS", "51");
        sol.config = H3CacheConfig::from_environment();
        require(!sol.uses_probe(), "SOL forward count exceeds actual scheduler calls");
        environment("SD_H3_CACHE_NUM_FORWARDS", "bad");
        require(H3CacheConfig::from_environment().policy == H3CachePolicy::OFF, "Invalid SOL forward count enabled caching");
        std::puts("PASS default-off, config validation, branch isolation, signature/scheduler invalidation, bounded reuse, audio protection, finite probes, fresh request");
        std::puts("PASS SOL full-stack policy, uncapped/consecutive reuse, joint probe, accumulated refresh, cooldown, control and invalidation");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
