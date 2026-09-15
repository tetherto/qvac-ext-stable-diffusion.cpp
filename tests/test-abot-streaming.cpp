#include <cmath>
#include <filesystem>
#include <iostream>

#include "core/ggml_extend.hpp"
#include "gguf.h"
#include "model_manager.h"

// Exercise the production executor on CPU so history/cut regressions are also
// covered on CI workers without GPUs or multi-gigabyte ABot model downloads.
struct AbotHistoryTestRunner : GGMLRunner {
    using GGMLRunner::GGMLRunner;

    std::string get_desc() override { return "ABot history streaming test"; }

    std::optional<sd::Tensor<float>> run(bool segmented, bool merge, bool append, int step) {
        auto input     = sd::Tensor<float>::from_vector(std::vector<float>(8, 0.25f + step));
        auto get_graph = [&]() {
            auto* gf = new_graph_custom(128);
            auto* x  = make_input(input);
            for (int layer = 0; layer < 3; ++layer) {
                const std::string name = "kv." + std::to_string(layer);
                x                      = ggml_scale(compute_ctx, x, 0.5f);
                if (auto* history = get_cache_tensor_by_name(name)) {
                    // History has a different shape from the final result and
                    // is an external input reused after segment scratch resets.
                    x = ggml_add(compute_ctx, x, ggml_concat(compute_ctx, history, history, 0));
                }
                sd::ggml_graph_cut::mark_graph_cut(x, name, "x");
                if (append) {
                    auto* capture = ggml_cont(compute_ctx, ggml_view_1d(compute_ctx, x, 4, 0));
                    sd::ggml_graph_cut::mark_graph_cut(capture, name, "capture");
                    ggml_set_output(capture);
                    if (segmented)
                        cache_persistent(name, capture);
                    else
                        cache(name, capture);
                }
            }
            ggml_build_forward_expand(gf, ggml_scale(compute_ctx, x, 2.f));
            return gf;
        };
        if (!segmented)
            return compute<float>(get_graph, 1, false);

        reset_compute_ctx();
        auto* gf  = get_compute_graph(get_graph);
        auto plan = sd::ggml_graph_cut::build_plan(runtime_backend, gf, {}, get_desc().c_str());
        if (merge) {
            plan = sd::ggml_graph_cut::apply_max_vram_budget(gf, plan, 1024 * 1024, runtime_backend, {}, nullptr);
        }
        if (!plan.valid || plan.segments.size() != (merge ? 1 : 4))
            return std::nullopt;
        return compute_graph_cut_segments<float>(gf, plan, 1, true);
    }
};

static bool test_history(ggml_backend_t backend, bool merge) {
    AbotHistoryTestRunner baseline(backend);
    AbotHistoryTestRunner streaming(backend);
    // Capture, reuse without mutation, append/overwrite, then reuse again.
    // Repeated updates also exercise stable cache addresses after cut scratch
    // has been freed and rebuilt for a subsequent walk block.
    for (int step = 0; step < 8; ++step) {
        const bool append = step % 2 == 0;
        auto expected     = baseline.run(false, false, append, step);
        auto actual       = streaming.run(true, merge, append, step);
        if (!expected || !actual || expected->numel() != 8 || actual->numel() != 8) {
            std::cerr << "Missing or incorrect video result at step " << step << '\n';
            return false;
        }
        for (int i = 0; i < 8; ++i) {
            if (!std::isfinite(actual->data()[i]) || std::fabs(expected->data()[i] - actual->data()[i]) > 1e-6f) {
                std::cerr << "History mismatch at step " << step << ", merge=" << merge << '\n';
                return false;
            }
        }
    }
    return true;
}

struct AbotDiskTestRunner : GGMLRunner {
    std::map<std::string, ggml_tensor*> weights;

    explicit AbotDiskTestRunner(ggml_backend_t backend) : GGMLRunner(backend) {
        for (int i = 0; i < 2; ++i) {
            const auto name = "model.diffusion_model.blocks." + std::to_string(i) + ".weight";
            auto* tensor    = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, 8);
            ggml_set_name(tensor, name.c_str());
            weights[name] = tensor;
        }
    }

    std::string get_desc() override { return "ABot disk residency test"; }

    std::optional<sd::Tensor<float>> run(std::shared_ptr<ModelManager> manager, bool segmented) {
        weight_manager = manager;
        auto input     = sd::Tensor<float>::from_vector(std::vector<float>(8, 1.f));
        auto get_graph = [&]() {
            auto* gf = new_graph_custom(32);
            auto* x  = make_input(input);
            for (const auto& weight : weights) {
                x = ggml_mul(compute_ctx, x, weight.second);
                sd::ggml_graph_cut::mark_graph_cut(x, weight.first, "x");
            }
            ggml_build_forward_expand(gf, ggml_scale(compute_ctx, x, 2.f));
            return gf;
        };
        if (!segmented)
            return compute<float>(get_graph, 1, false);
        reset_compute_ctx();
        auto* gf = get_compute_graph(get_graph);
        rebuild_params_tensor_set();
        auto plan = sd::ggml_graph_cut::build_plan(runtime_backend, gf, params_tensor_set_, nullptr);
        return compute_graph_cut_segments<float>(gf, plan, 1, true);
    }
};

static bool test_disk_residency(ggml_backend_t backend, bool disk, bool segmented) {
    AbotDiskTestRunner runner(backend);
    // Declare manager after runner: its teardown still needs the tensors.
    auto manager    = std::make_shared<ModelManager>();
    const auto path = std::filesystem::temp_directory_path() / "sd-abot-streaming-weights.gguf";
    auto* gguf      = gguf_init_empty();
    std::vector<float> values(8, 0.5f);
    for (const auto& weight : runner.weights) {
        weight.second->data = values.data();
        gguf_add_tensor(gguf, weight.second);
    }
    const bool written = gguf_write_to_file(gguf, path.string().c_str(), false);
    gguf_free(gguf);
    for (const auto& weight : runner.weights)
        weight.second->data = nullptr;
    const auto mode = disk ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend;
    bool passed     = written && manager->loader().init_from_file(path.string()) &&
                      manager->register_param_tensors("test", runner.weights, mode, backend, backend) &&
                      manager->validate_registered_tensors();
    for (int step = 0; passed && step < 3; ++step) {
        auto result = runner.run(manager, segmented);
        passed      = result && result->numel() == 8 && std::fabs(result->data()[0] - 0.5f) < 1e-6f;
        for (const auto& weight : runner.weights) {
            passed = passed && ((weight.second->buffer == nullptr) == disk);
        }
    }
    if (!passed)
        std::cerr << "Parameter residency failed: disk=" << disk << ", segmented=" << segmented << '\n';
    manager.reset();
    std::filesystem::remove(path);
    return passed;
}

static bool test_plan_cache(ggml_backend_t backend) {
    sd::ggml_graph_cut::PlanCache cache;
    int previous_variant = -1;
    for (int variant : {0, 0, 1, 1, 2, 2, 3, 3, 4, 4}) {
        ggml_init_params init = {1024 * 1024, nullptr, true};
        auto* ctx             = ggml_init(init);
        auto* graph           = ggml_new_graph_custom(ctx, 32, false);
        auto* input           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, variant == 4 ? 16 : 8);
        auto* a               = ggml_scale(ctx, input, 0.5f);
        auto* b               = ggml_sqr(ctx, variant == 2 ? a : input);
        sd::ggml_graph_cut::mark_graph_cut(a, variant == 3 ? "capture" : "pre", "x");
        auto* output = variant == 1 ? ggml_add(ctx, b, a) : ggml_add(ctx, a, b);
        ggml_set_name(output, "ggml_runner_final_result_tensor");
        ggml_build_forward_expand(graph, output);
        // Same counts and input shapes, different traversal and cut indices:
        // the denoise -> KV-capture transition must not reuse the old plan.
        // Also cover changed edges, cut names and dimensions, and rebuild
        // identical graphs at new addresses to verify valid cache reuse.
        if (previous_variant >= 0) {
            const bool expected_match = variant == previous_variant;
            if (sd::ggml_graph_cut::plan_matches_graph(graph, cache.graph_cut_plan) != expected_match ||
                sd::ggml_graph_cut::plan_matches_graph(graph, cache.budgeted_graph_cut_plan) != expected_match) {
                std::cerr << "Incorrect graph-cut cache match for variant " << variant << '\n';
                ggml_free(ctx);
                return false;
            }
        }
        auto plan = sd::ggml_graph_cut::resolve_plan(backend, graph, &cache, 1024 * 1024, {}, nullptr);
        if (!plan.valid || !sd::ggml_graph_cut::plan_matches_graph(graph, plan) ||
            sd::ggml_graph_cut::output_tensor(graph, cache.graph_cut_plan.segments.front(), 0) != a) {
            std::cerr << "Graph-cut cache failed to rebuild or reuse a matching plan\n";
            ggml_free(ctx);
            return false;
        }
        previous_variant = variant;
        ggml_free(ctx);
    }
    return true;
}

int main() {
    sd_abot_session_params_t params;
    sd_abot_session_params_init(&params);
    if (params.params_backend || params.max_vram || params.stream_layers || params.offload_params_to_cpu || params.kv_cache) {
        std::cerr << "ABot memory controls must remain opt-in\n";
        return 1;
    }
    auto* backend = sd_backend_cpu_init();
    if (!backend)
        return 1;
    bool passed = test_history(backend, false) && test_history(backend, true);
    passed      = test_plan_cache(backend) && passed;
    for (bool disk : {false, true}) {
        for (bool segmented : {false, true}) {
            passed = test_disk_residency(backend, disk, segmented) && passed;
        }
    }
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
