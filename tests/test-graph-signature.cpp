#include <ggml.h>
#include <iostream>
#include <stdexcept>

#include "core/graph_signature.h"

namespace {
    constexpr size_t GRAPH_CONTEXT_BYTES = 1024 * 1024;
    constexpr int64_t WIDTH              = 8;
    constexpr int64_t CHANNELS           = 4;
    constexpr float SCALE                = 2.0f;
    constexpr float OTHER_SCALE          = 3.0f;

    struct Block {
        ggml_context* ctx = ggml_init({GRAPH_CONTEXT_BYTES, nullptr, true});
        ggml_tensor* input;
        ggml_tensor* weight;
        ggml_cgraph* graph;

        Block(int64_t width, float scale, bool add_weight_first) {
            input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, CHANNELS);
            weight       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, CHANNELS);
            auto scaled  = ggml_scale(ctx, input, scale);
            auto shifted = add_weight_first ? ggml_add(ctx, weight, scaled) : ggml_add(ctx, scaled, weight);
            graph        = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, ggml_mul(ctx, shifted, scaled));
        }
        ~Block() { ggml_free(ctx); }
    };

    bool nothing_is_external(const ggml_tensor*) {
        return false;
    }

    sd::GraphSignature sign(const Block& block) {
        return sd::graph_signature(block.graph, nothing_is_external);
    }

    void expect(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void test_same_graphs_share_a_signature() {
        Block first(WIDTH, SCALE, false);
        Block second(WIDTH, SCALE, false);
        expect(sign(first) == sign(second), "graphs built the same way must share a signature");
    }

    void test_shape_changes_the_signature() {
        Block narrow(WIDTH, SCALE, false);
        Block wide(WIDTH * 2, SCALE, false);
        expect(sign(narrow) != sign(wide), "a different shape must change the signature");
    }

    void test_op_params_change_the_signature() {
        Block first(WIDTH, SCALE, false);
        Block second(WIDTH, OTHER_SCALE, false);
        expect(sign(first) != sign(second), "different op params must change the signature");
    }

    void test_wiring_changes_the_signature() {
        Block first(WIDTH, SCALE, false);
        Block swapped(WIDTH, SCALE, true);
        expect(sign(first) != sign(swapped), "different sources must change the signature");
    }

    void test_external_tensors_change_the_signature() {
        Block block(WIDTH, SCALE, false);
        const auto weight_is_external = [&](const ggml_tensor* tensor) { return tensor == block.weight; };
        expect(sign(block) != sd::graph_signature(block.graph, weight_is_external),
               "marking a tensor as external must change the signature");
    }
}

int main() {
    try {
        test_same_graphs_share_a_signature();
        test_shape_changes_the_signature();
        test_op_params_change_the_signature();
        test_wiring_changes_the_signature();
        test_external_tensors_change_the_signature();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "graph signature tests passed" << std::endl;
    return 0;
}
