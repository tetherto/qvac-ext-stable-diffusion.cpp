#include <ggml-backend.h>
#include <ggml.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"

namespace {
    constexpr size_t CONTEXT_BYTES   = 8 * 1024 * 1024;
    constexpr size_t GRAPH_NODES     = 64;
    constexpr int GROUPS             = 32;
    constexpr float EPSILON          = 1e-6f;
    constexpr float TOLERANCE        = 1e-4f;
    constexpr int VALUE_STEP         = 37;
    constexpr int VALUE_RANGE        = 251;
    constexpr float VALUE_SCALE      = 17.0f;
    constexpr int64_t UNEVEN_CHANNEL = 48;

    struct Shape {
        int64_t width;
        int64_t height;
        int64_t channels;
        int64_t batch;
    };

    const std::vector<Shape> SHAPES = {{8, 8, 64, 1}, {4, 6, 64, 2}, {5, 3, 96, 3}};

    using NormBuilder = std::function<ggml_tensor*(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*)>;

    struct Storage {
        ggml_context* ctx            = ggml_init({CONTEXT_BYTES, nullptr, true});
        ggml_backend_buffer_t buffer = nullptr;
        ~Storage() {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
            ggml_free(ctx);
        }
    };

    void require(bool ok, const char* message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
    }

    bool accepts_every_op(const ggml_tensor*) {
        return true;
    }

    bool rejects_every_op(const ggml_tensor*) {
        return false;
    }

    bool rejects_group_norm(const ggml_tensor* node) {
        return node->op != GGML_OP_GROUP_NORM;
    }

    std::vector<float> make_values(size_t count, int seed) {
        std::vector<float> values(count);
        for (size_t index = 0; index < count; ++index) {
            values[index] = float(int((index * VALUE_STEP + seed) % VALUE_RANGE) - VALUE_RANGE / 2) / VALUE_SCALE;
        }
        return values;
    }

    void fill_tensor(ggml_tensor* tensor, int seed) {
        std::vector<float> values = make_values(ggml_nelements(tensor), seed);
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }

    std::vector<float> read_tensor(const ggml_tensor* tensor) {
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        return values;
    }

    bool graph_has_op(const ggml_cgraph* graph, ggml_op op) {
        for (int index = 0; index < ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)); ++index) {
            if (ggml_graph_node(const_cast<ggml_cgraph*>(graph), index)->op == op) {
                return true;
            }
        }
        return false;
    }

    ggml_cgraph* build_graph(ggml_context* ctx, ggml_tensor* result) {
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, GRAPH_NODES, false);
        ggml_build_forward_expand(graph, result);
        return graph;
    }

    std::vector<float> compute_norm(ggml_backend_t backend, const Shape& shape, const NormBuilder& build) {
        Storage storage;
        auto* input        = ggml_new_tensor_4d(storage.ctx, GGML_TYPE_F32, shape.width, shape.height, shape.channels, shape.batch);
        auto* weight       = ggml_new_tensor_1d(storage.ctx, GGML_TYPE_F32, shape.channels);
        auto* bias         = ggml_new_tensor_1d(storage.ctx, GGML_TYPE_F32, shape.channels);
        ggml_tensor* out   = build(storage.ctx, input, weight, bias);
        ggml_cgraph* graph = build_graph(storage.ctx, out);
        storage.buffer     = ggml_backend_alloc_ctx_tensors(storage.ctx, backend);
        require(storage.buffer != nullptr, "Allocation failed");
        fill_tensor(input, 0);
        fill_tensor(weight, 1);
        fill_tensor(bias, 2);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Graph compute failed");
        return read_tensor(out);
    }

    void require_close(const std::vector<float>& expected, const std::vector<float>& actual, const char* message) {
        require(expected.size() == actual.size(), message);
        for (size_t index = 0; index < expected.size(); ++index) {
            require(std::isfinite(actual[index]) && std::abs(expected[index] - actual[index]) <= TOLERANCE, message);
        }
    }

    ggml_tensor* build_reference(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        return ggml_ext_group_norm(ctx, x, w, b, GROUPS, EPSILON);
    }

    ggml_tensor* build_bare_reference(ggml_context* ctx, ggml_tensor* x, ggml_tensor*, ggml_tensor*) {
        return ggml_ext_group_norm(ctx, x, nullptr, nullptr, GROUPS, EPSILON);
    }

    ggml_tensor* build_bare_row_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor*, ggml_tensor*) {
        return ggml_ext_group_norm_as_row_norm(ctx, x, GROUPS, EPSILON);
    }

    ggml_tensor* build_without_group_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        return ggml_ext_group_norm(ctx, x, w, b, GROUPS, EPSILON, rejects_group_norm);
    }

    void test_row_norm_matches_group_norm(ggml_backend_t backend, const Shape& shape) {
        require_close(compute_norm(backend, shape, build_bare_reference),
                      compute_norm(backend, shape, build_bare_row_norm),
                      "Row norm differs from group norm");
        require_close(compute_norm(backend, shape, build_reference),
                      compute_norm(backend, shape, build_without_group_norm),
                      "Affine group norm differs when the backend lacks GROUP_NORM");
    }

    void test_all_shapes_match(ggml_backend_t backend) {
        for (const Shape& shape : SHAPES) {
            test_row_norm_matches_group_norm(backend, shape);
        }
    }

    ggml_cgraph* build_selection_graph(Storage& storage, int64_t channels, const ggml_ext_op_support_t& supports) {
        auto* input = ggml_new_tensor_4d(storage.ctx, GGML_TYPE_F32, SHAPES[0].width, SHAPES[0].height, channels, SHAPES[0].batch);
        return build_graph(storage.ctx, ggml_ext_group_norm(storage.ctx, input, nullptr, nullptr, GROUPS, EPSILON, supports));
    }

    void require_form(int64_t channels, const ggml_ext_op_support_t& supports, ggml_op expected, ggml_op unexpected, const char* message) {
        Storage storage;
        ggml_cgraph* graph = build_selection_graph(storage, channels, supports);
        require(graph_has_op(graph, expected) && !graph_has_op(graph, unexpected), message);
    }

    void test_form_selection(ggml_backend_t backend) {
        const int64_t channels = SHAPES[0].channels;
        require(!ggml_ext_backend_op_support(nullptr), "A null backend must give no support check");
        require_form(channels, nullptr, GGML_OP_GROUP_NORM, GGML_OP_NORM, "No support check must keep GROUP_NORM");
        require_form(channels, ggml_ext_backend_op_support(backend), GGML_OP_GROUP_NORM, GGML_OP_NORM, "A backend with GROUP_NORM must keep it");
        require_form(channels, accepts_every_op, GGML_OP_GROUP_NORM, GGML_OP_NORM, "A supported GROUP_NORM must be kept");
        require_form(channels, rejects_group_norm, GGML_OP_NORM, GGML_OP_GROUP_NORM, "A backend without GROUP_NORM must get the row norm");
        require_form(channels, rejects_every_op, GGML_OP_GROUP_NORM, GGML_OP_NORM, "A backend without NORM must keep GROUP_NORM");
        require_form(UNEVEN_CHANNEL, rejects_group_norm, GGML_OP_GROUP_NORM, GGML_OP_NORM, "Groups that split channels must keep GROUP_NORM");
    }

    void require_own_output(ggml_tensor* affine, const char* message) {
        require(affine->op == GGML_OP_ADD && affine->view_src == nullptr, message);
        require(affine->src[0]->op == GGML_OP_MUL && affine->src[0]->view_src == nullptr, message);
    }

    void test_scale_and_shift_are_not_views() {
        Storage storage;
        const Shape& shape = SHAPES[0];
        auto* input        = ggml_new_tensor_4d(storage.ctx, GGML_TYPE_F32, shape.width, shape.height, shape.channels, shape.batch);
        auto* weight       = ggml_new_tensor_1d(storage.ctx, GGML_TYPE_F32, shape.channels);
        auto* bias         = ggml_new_tensor_1d(storage.ctx, GGML_TYPE_F32, shape.channels);
        auto* row_weight   = ggml_new_tensor_1d(storage.ctx, GGML_TYPE_F32, shape.width);
        auto* row_bias     = ggml_new_tensor_1d(storage.ctx, GGML_TYPE_F32, shape.width);
        require_own_output(ggml_ext_group_norm(storage.ctx, input, weight, bias, GROUPS, EPSILON, rejects_group_norm),
                           "Group norm scale and shift must own their output so that a backend can fuse them");
        require_own_output(ggml_ext_layer_norm(storage.ctx, input, row_weight, row_bias, EPSILON),
                           "Layer norm scale and shift must own their output so that a backend can fuse them");
    }
}

int main() {
    ggml_backend_load_all();
    SDBackendHandle backend(sd_backend_cpu_init());
    try {
        require(backend.get() != nullptr, "CPU backend unavailable");
        test_all_shapes_match(backend.get());
        test_form_selection(backend.get());
        test_scale_and_shift_are_not_views();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS group norm falls back to an equivalent row norm only when the backend lacks GROUP_NORM\n";
    return 0;
}
