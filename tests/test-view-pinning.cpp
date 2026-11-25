#include <ggml-backend.h>
#include <ggml.h>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../ggml/src/ggml-backend-impl.h"
#include "core/ggml_extend_backend.h"
#include "core/view_pinning.h"

namespace {
    constexpr size_t GRAPH_CONTEXT_BYTES = 1024 * 1024;
    constexpr size_t GRAPH_NODES         = 64;
    constexpr int SCHEDULER_BACKENDS     = 3;
    constexpr int64_t WIDTH              = 4;
    constexpr int64_t HEIGHT             = 4;
    constexpr int64_t CHANNELS           = 8;
    constexpr int GROUPS                 = 2;
    constexpr float EPSILON              = 1e-6f;
    constexpr float INPUT_VALUE          = 0.5f;
    constexpr float PRELUDE_SCALE        = 2.0f;

    struct NormBlock {
        ggml_context* ctx      = ggml_init({GRAPH_CONTEXT_BYTES, nullptr, true});
        ggml_tensor* input     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, WIDTH, HEIGHT, CHANNELS, 1);
        ggml_tensor* weight    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, CHANNELS, 1);
        ggml_tensor* reshaped  = ggml_reshape_1d(ctx, weight, CHANNELS);
        ggml_tensor* root      = ggml_group_norm(ctx, input, GROUPS, EPSILON);
        ggml_tensor* scaled    = ggml_mul_inplace(ctx, root, weight);
        ggml_tensor* shifted   = ggml_add_inplace(ctx, scaled, weight);
        ggml_tensor* out_place = ggml_mul(ctx, root, weight);
        ~NormBlock() { ggml_free(ctx); }
    };

    struct SplitBlock {
        ggml_context* ctx    = ggml_init({GRAPH_CONTEXT_BYTES, nullptr, true});
        ggml_tensor* input   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, WIDTH, HEIGHT, CHANNELS, 1);
        ggml_tensor* weight  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, CHANNELS, 1);
        ggml_tensor* prelude = ggml_scale(ctx, input, PRELUDE_SCALE);
        ggml_tensor* root    = ggml_group_norm(ctx, prelude, GROUPS, EPSILON);
        ggml_tensor* scaled  = ggml_mul_inplace(ctx, root, weight);
        ggml_tensor* shifted = ggml_add_inplace(ctx, scaled, weight);
        ~SplitBlock() { ggml_free(ctx); }
    };

    struct Backends {
        SDBackendHandle first{sd_backend_cpu_init()};
        SDBackendHandle second{sd_backend_cpu_init()};
        SDBackendHandle fallback{sd_backend_cpu_init()};
    };

    using Assignments = std::unordered_map<const ggml_tensor*, ggml_backend_t>;

    const char* accelerator_buffer_name(ggml_backend_buffer_type_t) {
        return "TestAccelerator";
    }

    ggml_backend_buffer_t allocate_accelerator_buffer(ggml_backend_buffer_type_t, size_t size) {
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    size_t accelerator_buffer_alignment(ggml_backend_buffer_type_t) {
        return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    }

    bool accelerator_buffer_is_host(ggml_backend_buffer_type_t) {
        return true;
    }

    ggml_backend_buffer_type make_accelerator_buffer_type(ggml_backend_t backend) {
        return {{accelerator_buffer_name, allocate_accelerator_buffer, accelerator_buffer_alignment, nullptr, nullptr, accelerator_buffer_is_host},
                ggml_backend_get_device(backend),
                nullptr};
    }

    void require(bool ok, const char* message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
    }

    bool supports_every_node(ggml_backend_t, const ggml_tensor*) {
        return true;
    }

    NodeSupportCheck rejects_group_norm_on(ggml_backend_t rejecting) {
        return [rejecting](ggml_backend_t backend, const ggml_tensor* node) {
            return backend != rejecting || node->op != GGML_OP_GROUP_NORM;
        };
    }

    NodeSupportCheck rejects_group_norm_on_both(ggml_backend_t first, ggml_backend_t second) {
        return [first, second](ggml_backend_t backend, const ggml_tensor* node) {
            return (backend != first && backend != second) || node->op != GGML_OP_GROUP_NORM;
        };
    }

    NodeAssignment record_into(Assignments& assignments) {
        return [&assignments](ggml_tensor* node, ggml_backend_t backend) { assignments[node] = backend; };
    }

    ggml_backend_t assigned(const Assignments& assignments, const ggml_tensor* node) {
        auto found = assignments.find(node);
        return found == assignments.end() ? nullptr : found->second;
    }

    void test_view_roots(const NormBlock& block) {
        require(find_view_root(block.root) == block.root, "A node without a view source is its own root");
        require(find_view_root(block.scaled) == block.root, "An in-place op resolves to the node it modifies");
        require(find_view_root(block.shifted) == block.root, "Chained in-place ops resolve to the first node");
    }

    void test_computed_views(const NormBlock& block) {
        require(is_computed_view(block.scaled), "An in-place op on a computed node is a computed view");
        require(is_computed_view(block.shifted), "A chained in-place op is a computed view");
        require(!is_computed_view(block.root), "A computed node is not a view");
        require(!is_computed_view(block.out_place), "An out-of-place op is not a view");
        require(!is_computed_view(block.reshaped), "A reshaped weight is not a computed view");
    }

    void test_in_place_op_follows_accelerator_root(const NormBlock& block, const Backends& backends) {
        Assignments assignments;
        NodePinner pinner(backends.fallback.get(), supports_every_node, record_into(assignments));
        pinner.pin(block.root, backends.first.get());
        pinner.pin(block.scaled, backends.second.get());
        pinner.pin(block.shifted, backends.second.get());
        require(assigned(assignments, block.root) == backends.first.get(), "A supported node is pinned to the current backend");
        require(assigned(assignments, block.scaled) == backends.first.get(), "An in-place op must stay with the root's backend");
        require(assigned(assignments, block.shifted) == backends.first.get(), "A chained in-place op must stay with the root's backend");
    }

    void test_rejected_root_is_assigned_to_fallback(const NormBlock& block, const Backends& backends) {
        Assignments assignments;
        NodePinner pinner(backends.fallback.get(), rejects_group_norm_on(backends.first.get()), record_into(assignments));
        pinner.pin(block.root, backends.first.get());
        require(assigned(assignments, block.root) == nullptr, "A rejected node is left alone until an in-place op needs it");
        pinner.pin(block.scaled, backends.first.get());
        pinner.pin(block.shifted, backends.first.get());
        require(assigned(assignments, block.root) == backends.fallback.get(),
                "A root the second accelerator could take must be assigned to the fallback explicitly");
        require(assigned(assignments, block.scaled) == backends.fallback.get(), "An in-place op must run where its root was assigned");
        require(assigned(assignments, block.shifted) == backends.fallback.get(), "A chained in-place op must run where its root was assigned");
    }

    void test_other_nodes_keep_current_backend(const NormBlock& block, const Backends& backends) {
        Assignments assignments;
        NodePinner pinner(backends.fallback.get(), rejects_group_norm_on(backends.first.get()), record_into(assignments));
        pinner.pin(block.root, backends.first.get());
        pinner.pin(block.out_place, backends.first.get());
        require(assigned(assignments, block.out_place) == backends.first.get(), "An out-of-place op keeps the current backend");
        require(assigned(assignments, block.root) == nullptr, "An out-of-place consumer must not move its source");
    }

    void test_missing_fallback_keeps_current_backend(const NormBlock& block, const Backends& backends) {
        Assignments assignments;
        NodePinner pinner(nullptr, rejects_group_norm_on(backends.first.get()), record_into(assignments));
        pinner.pin(block.root, backends.first.get());
        pinner.pin(block.scaled, backends.first.get());
        require(assigned(assignments, block.root) == nullptr, "Without a fallback backend the root cannot be reassigned");
        require(assigned(assignments, block.scaled) == backends.first.get(), "Without a fallback backend the current backend is kept");
    }

    void test_unplaceable_root_leaves_in_place_op_to_scheduler(const NormBlock& block, const Backends& backends) {
        Assignments assignments;
        NodePinner pinner(backends.fallback.get(),
                          rejects_group_norm_on_both(backends.first.get(), backends.fallback.get()),
                          record_into(assignments));
        pinner.pin(block.root, backends.first.get());
        pinner.pin(block.scaled, backends.first.get());
        require(assigned(assignments, block.root) == nullptr, "A root the fallback rejects stays with the scheduler");
        require(assigned(assignments, block.scaled) == nullptr, "An in-place op with an unknown owner stays with the scheduler");
    }

    void test_allocated_root_stays_with_scheduler(const Backends& backends) {
        NormBlock block;
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(block.ctx, backends.second.get());
        require(buffer != nullptr && block.root->buffer != nullptr, "Allocation failed");
        Assignments assignments;
        NodePinner pinner(backends.fallback.get(), supports_every_node, record_into(assignments));
        pinner.pin(block.scaled, backends.first.get());
        ggml_backend_buffer_free(buffer);
        require(assigned(assignments, block.root) == nullptr, "A root allocated elsewhere must not be reassigned");
        require(assigned(assignments, block.scaled) == nullptr, "An in-place op on an allocated root stays with the scheduler");
    }

    void pin_split_graph(ggml_cgraph* graph, NodePinner& pinner, const SplitBlock& block, const Backends& backends) {
        for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            ggml_tensor* node = ggml_graph_node(graph, index);
            pinner.pin(node, node == block.prelude ? backends.second.get() : backends.first.get());
        }
    }

    void fill_tensor(ggml_tensor* tensor) {
        std::vector<float> values(ggml_nelements(tensor), INPUT_VALUE);
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }

    bool compute_graph(ggml_backend_sched_t sched, ggml_cgraph* graph, const SplitBlock& block) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return false;
        }
        fill_tensor(block.input);
        fill_tensor(block.weight);
        return ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    }

    void test_scheduler_keeps_in_place_op_with_rejected_root(const Backends& backends) {
        SplitBlock block;
        ggml_backend_buffer_type first_type                   = make_accelerator_buffer_type(backends.first.get());
        ggml_backend_buffer_type second_type                  = make_accelerator_buffer_type(backends.second.get());
        ggml_backend_t order[SCHEDULER_BACKENDS]              = {backends.first.get(), backends.second.get(), backends.fallback.get()};
        ggml_backend_buffer_type_t types[SCHEDULER_BACKENDS] = {&first_type, &second_type, ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_t sched                            = ggml_backend_sched_new(order, types, SCHEDULER_BACKENDS, GRAPH_NODES, false, true);
        ggml_cgraph* graph                                    = ggml_new_graph_custom(block.ctx, GRAPH_NODES, false);
        ggml_build_forward_expand(graph, block.shifted);
        NodePinner pinner(backends.fallback.get(), rejects_group_norm_on(backends.first.get()),
                          [sched](ggml_tensor* node, ggml_backend_t backend) { ggml_backend_sched_set_tensor_backend(sched, node, backend); });
        pin_split_graph(graph, pinner, block, backends);
        const bool computed            = compute_graph(sched, graph, block);
        ggml_backend_t prelude_backend = ggml_backend_sched_get_tensor_backend(sched, block.prelude);
        ggml_backend_t root_backend    = ggml_backend_sched_get_tensor_backend(sched, block.root);
        ggml_backend_t scale_backend   = ggml_backend_sched_get_tensor_backend(sched, block.scaled);
        ggml_backend_t shift_backend   = ggml_backend_sched_get_tensor_backend(sched, block.shifted);
        ggml_backend_sched_free(sched);
        require(computed, "The scheduler must allocate and compute the pinned graph");
        require(prelude_backend == backends.second.get(), "The node before the root runs on the second accelerator");
        require(root_backend == backends.fallback.get(), "The scheduler must not move the rejected root to the second accelerator");
        require(scale_backend == root_backend && shift_backend == root_backend,
                "The scheduler must run in-place ops on the backend that owns their root");
    }
}

int main() {
    ggml_backend_load_all();
    Backends backends;
    try {
        require(backends.first && backends.second && backends.fallback, "CPU backend unavailable");
        NormBlock block;
        test_view_roots(block);
        test_computed_views(block);
        test_in_place_op_follows_accelerator_root(block, backends);
        test_rejected_root_is_assigned_to_fallback(block, backends);
        test_other_nodes_keep_current_backend(block, backends);
        test_missing_fallback_keeps_current_backend(block, backends);
        test_unplaceable_root_leaves_in_place_op_to_scheduler(block, backends);
        test_allocated_root_stays_with_scheduler(backends);
        test_scheduler_keeps_in_place_op_with_rejected_root(backends);
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS in-place ops are pinned to the backend of the node they modify\n";
    return 0;
}
