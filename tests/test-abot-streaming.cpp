#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>

#include "core/ggml_extend.hpp"
#include "gguf.h"
#include "model_manager.h"

// Exercise the production executor on CPU so history/cut regressions are also
// covered on CI workers without GPUs or multi-gigabyte ABot model downloads.
struct AbotHistoryTestRunner : GGMLRunner {
    using GGMLRunner::GGMLRunner;

    std::string get_desc() override { return "ABot history streaming test"; }

    size_t persistent_pool_count() const { return persistent_cache_pools_.size(); }

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
    if (streaming.persistent_pool_count() != 1) {
        std::cerr << "Persistent history tensors were not pooled\n";
        return false;
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

static bool test_disk_residency(ggml_backend_t backend, ggml_backend_t params_backend, bool disk, bool segmented) {
    AbotDiskTestRunner runner(backend);
    // Declare manager after runner: its teardown still needs the tensors.
    auto manager         = std::make_shared<ModelManager>();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("sd-abot-streaming-" + std::to_string(std::random_device{}()) +
                            "-" + std::to_string(std::random_device{}()));
    if (!std::filesystem::create_directory(directory))
        return false;
    const auto path = directory / "weights.gguf";
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
                      manager->register_param_tensors("test", runner.weights, mode, backend, params_backend) &&
                      manager->validate_registered_tensors();
    for (int step = 0; passed && step < 3; ++step) {
        auto result = runner.run(manager, segmented);
        passed      = result && result->numel() == 8;
        if (passed) {
            for (int i = 0; i < 8; ++i)
                passed = std::isfinite(result->data()[i]) && std::fabs(result->data()[i] - 0.5f) < 1e-6f && passed;
        }
        for (const auto& weight : runner.weights) {
            passed = passed && ((weight.second->buffer == nullptr) == disk);
        }
    }
    if (!passed)
        std::cerr << "Parameter residency failed: disk=" << disk << ", segmented=" << segmented << '\n';
    manager.reset();
    std::filesystem::remove(path);
    std::filesystem::remove(directory);
    return passed;
}

static bool test_view_output(ggml_backend_t backend) {
    ggml_init_params init = {1024 * 1024, nullptr, true};
    auto* ctx             = ggml_init(init);
    auto* graph           = ggml_new_graph_custom(ctx, 32, false);
    auto* input           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    auto* root            = ggml_scale(ctx, input, 2.f);
    auto* view            = ggml_reshape_2d(ctx, root, 4, 2);
    auto* consumed        = ggml_scale(ctx, view, 3.f);
    // A merged segment consumes a cut view before copying it to the cache.
    // The later allocation must not reuse its still-live backing buffer.
    auto* tail = ggml_cont(ctx, consumed);
    ggml_build_forward_expand(graph, tail);
    sd::ggml_graph_cut::Segment segment;
    sd::ggml_graph_cut::Segment::InputRef ref;
    ref.type       = sd::ggml_graph_cut::Segment::INPUT_EXTERNAL;
    ref.leaf_index = 0;
    segment.input_refs.push_back(ref);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        segment.internal_node_indices.push_back(i);
        if (ggml_graph_node(graph, i) == view || ggml_graph_node(graph, i) == tail)
            segment.output_node_indices.push_back(i);
    }
    const auto root_flags = root->flags;
    const auto view_flags = view->flags;
    sd::ggml_graph_cut::measure_segment_compute_buffer(backend, graph, segment, nullptr);
    bool passed               = root->flags == root_flags && view->flags == view_flags;
    ggml_context* segment_ctx = nullptr;
    auto* cut_graph           = sd::ggml_graph_cut::build_segment_graph(graph, segment, &segment_ctx);
    auto* allocator           = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    passed                    = ggml_gallocr_alloc_graph(allocator, cut_graph) && passed;
    std::vector<float> values(8, 0.5f);
    if (passed) {
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        passed = ggml_backend_graph_compute(backend, cut_graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(view, values.data(), 0, values.size() * sizeof(float));
        for (float value : values)
            passed = std::fabs(value - 1.f) < 1e-6f && passed;
    }
    ggml_gallocr_free(allocator);
    ggml_free(segment_ctx);
    ggml_free(ctx);
    if (!passed)
        std::cerr << "Merged segment overwrote a cut view's backing buffer\n";
    return passed;
}

struct AbotPlanCacheTestRunner : GGMLRunner {
    using GGMLRunner::GGMLRunner;

    std::string get_desc() override { return "ABot phase plan cache test"; }

    bool resolve_phase(size_t phase) {
        reset_compute_ctx();
        auto get_graph = [&]() {
            auto* graph = new_graph_custom(32);
            auto* input = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 8);
            auto* a     = ggml_scale(compute_ctx, input, 0.5f);
            auto* b     = phase == 1 ? ggml_sqr(compute_ctx, a) : ggml_sqr(compute_ctx, input);
            sd::ggml_graph_cut::mark_graph_cut(a, phase == 2 ? "append" : "denoise", "x");
            auto* output = phase == 0 ? ggml_add(compute_ctx, a, b) : ggml_add(compute_ctx, b, a);
            ggml_build_forward_expand(graph, output);
            return graph;
        };
        auto* graph = get_compute_graph(get_graph);
        rebuild_params_tensor_set();
        set_graph_cut_plan_cache_key(phase);
        GraphCutPlan plan;
        return resolve_graph_cut_plan(graph, &plan) && plan.valid &&
               sd::ggml_graph_cut::plan_matches_graph(graph, plan);
    }

    size_t phase_cache_count() const { return graph_cut_plan_caches_.size(); }
};

static bool test_phase_plan_caches(ggml_backend_t backend) {
    AbotPlanCacheTestRunner runner(backend);
    for (int iteration = 0; iteration < 3; ++iteration) {
        for (size_t phase = 0; phase < 3; ++phase) {
            if (!runner.resolve_phase(phase)) {
                std::cerr << "Failed to resolve ABot graph phase " << phase << '\n';
                return false;
            }
        }
    }
    if (runner.phase_cache_count() != 3) {
        std::cerr << "ABot graph phases did not retain separate plan caches\n";
        return false;
    }
    return true;
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

int main(int argc, char** argv) {
    struct {
        sd_abot_session_params_t params;
        uint64_t canary;
    } legacy = {{}, UINT64_C(0x8a7b6c5d4e3f2011)};
    sd_abot_session_params_init(&legacy.params);
    if (legacy.canary != UINT64_C(0x8a7b6c5d4e3f2011) ||
        legacy.params.n_threads != -1 || legacy.params.seed != 42) {
        std::cerr << "Legacy ABot parameter initialization changed its ABI\n";
        return 1;
    }
    sd_abot_session_params_v2_t params;
    sd_abot_session_params_v2_init(&params);
    if (params.params_backend || params.max_vram || params.stream_layers || params.offload_params_to_cpu || params.kv_cache) {
        std::cerr << "ABot memory controls must remain opt-in\n";
        return 1;
    }
    SDBackendManager backends;
    std::string error;
    if (argc > 2 || !backends.init(argc == 2 ? argv[1] : "cpu", "diffusion=cpu", nullptr, false, &error)) {
        std::cerr << "Expected an available backend name: " << error << '\n';
        return 1;
    }
    auto disk_vae_params                = params;
    disk_vae_params.dit_model_path      = "unused-dit";
    disk_vae_params.taehv_path          = "unused-taehv";
    disk_vae_params.scene_path          = "unused-scene";
    disk_vae_params.backend             = "cpu";
    disk_vae_params.params_backend      = "vae=disk";
    if (sd_abot_session_new_v2(&disk_vae_params) != nullptr) {
        std::cerr << "ABot accepted unsupported vae=disk residency\n";
        return 1;
    }
    auto* backend        = backends.runtime_backend(SDBackendModule::DIFFUSION);
    auto* params_backend = backends.params_backend(SDBackendModule::DIFFUSION);
    std::cout << "Testing streaming on " << ggml_backend_name(backend) << '\n';
    bool passed = test_history(backend, false) && test_history(backend, true);
    passed      = test_plan_cache(backend) && test_phase_plan_caches(backend) && passed;
    passed      = test_view_output(backend) && passed;
    for (bool disk : {false, true}) {
        for (bool segmented : {false, true}) {
            passed = test_disk_residency(backend, params_backend, disk, segmented) && passed;
        }
    }
    return passed ? 0 : 1;
}
