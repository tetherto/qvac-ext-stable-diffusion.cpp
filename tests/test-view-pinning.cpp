#include <ggml-backend.h>
#include <ggml.h>
#include <iostream>
#include <stdexcept>

#include "core/ggml_extend_backend.h"
#include "core/view_pinning.h"

namespace {
    constexpr size_t GRAPH_CONTEXT_BYTES = 1024 * 1024;
    constexpr int64_t WIDTH              = 4;
    constexpr int64_t HEIGHT             = 4;
    constexpr int64_t CHANNELS           = 8;
    constexpr int GROUPS                 = 2;
    constexpr float EPSILON              = 1e-6f;

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

    void require(bool ok, const char* message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
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

    void test_in_place_op_follows_fallback_root(const NormBlock& block, ggml_backend_t accelerator, ggml_backend_t cpu) {
        PinnedNodeBackends pinned{{block.root, cpu}};
        require(select_pin_backend(block.scaled, accelerator, cpu, pinned) == cpu,
                "An in-place op must run where its root was pinned");
        require(select_pin_backend(block.shifted, accelerator, cpu, pinned) == cpu,
                "A chained in-place op must run where its root was pinned");
    }

    void test_in_place_op_follows_accelerator_root(const NormBlock& block, ggml_backend_t accelerator, ggml_backend_t cpu) {
        PinnedNodeBackends pinned{{block.root, accelerator}};
        require(select_pin_backend(block.scaled, cpu, cpu, pinned) == accelerator,
                "An in-place op must stay with a root pinned to the accelerator");
    }

    void test_unpinned_root_uses_fallback(const NormBlock& block, ggml_backend_t accelerator, ggml_backend_t cpu) {
        PinnedNodeBackends pinned;
        require(select_pin_backend(block.scaled, accelerator, cpu, pinned) == cpu,
                "An in-place op on an unpinned root must use the fallback backend");
        require(select_pin_backend(block.scaled, accelerator, nullptr, pinned) == accelerator,
                "Without a fallback backend the current backend is kept");
    }

    void test_other_nodes_keep_current_backend(const NormBlock& block, ggml_backend_t accelerator, ggml_backend_t cpu) {
        PinnedNodeBackends pinned{{block.root, cpu}};
        require(select_pin_backend(block.out_place, accelerator, cpu, pinned) == accelerator,
                "An out-of-place op keeps the current backend");
        require(select_pin_backend(block.root, accelerator, cpu, pinned) == accelerator,
                "A computed node keeps the current backend");
    }
}

int main() {
    ggml_backend_load_all();
    SDBackendHandle accelerator(sd_backend_cpu_init());
    SDBackendHandle cpu(sd_backend_cpu_init());
    try {
        require(accelerator.get() != nullptr && cpu.get() != nullptr, "CPU backend unavailable");
        NormBlock block;
        test_view_roots(block);
        test_computed_views(block);
        test_in_place_op_follows_fallback_root(block, accelerator.get(), cpu.get());
        test_in_place_op_follows_accelerator_root(block, accelerator.get(), cpu.get());
        test_unpinned_root_uses_fallback(block, accelerator.get(), cpu.get());
        test_other_nodes_keep_current_backend(block, accelerator.get(), cpu.get());
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS in-place ops are pinned to the backend of the node they modify\n";
    return 0;
}
