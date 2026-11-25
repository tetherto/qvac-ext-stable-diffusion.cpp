#include <ggml-backend.h>
#include <ggml.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"

namespace {
    constexpr size_t CONTEXT_BYTES = 8 * 1024 * 1024;
    constexpr size_t GRAPH_NODES   = 64;
    constexpr float TOLERANCE      = 1e-4f;
    constexpr int VALUE_STEP       = 37;
    constexpr int VALUE_RANGE      = 251;
    constexpr float VALUE_SCALE    = 17.0f;
    constexpr int64_t CHANNELS_IN  = 5;
    constexpr int64_t CHANNELS_OUT = 7;
    constexpr int UNIT_STRIDE      = 1;
    constexpr int STRIDE_PARAM     = 0;

    struct Geometry {
        int64_t width;
        int64_t height;
        int64_t batch;
        int kernel;
        int s0;
        int s1;
        int padding;
    };

    const std::vector<Geometry> STRIDED = {{8, 8, 1, 3, 2, 2, 1}, {9, 6, 2, 3, 2, 2, 0}, {8, 6, 2, 1, 2, 2, 0}, {12, 7, 1, 3, 3, 2, 1}};
    const Geometry DENSE                = {8, 8, 1, 3, 1, 1, 1};
    const std::vector<Geometry> POINTWISE = {{8, 6, 1, 1, 1, 1, 0}, {5, 7, 3, 1, 1, 1, 0}};

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

    struct Inputs {
        ggml_tensor* x;
        ggml_tensor* w;
        ggml_tensor* b;
    };

    using ConvBuilder = ggml_tensor* (*)(ggml_context*, const Inputs&, const Geometry&);

    void require(bool ok, const char* message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
    }

    bool is_strided_conv(const ggml_tensor* node) {
        return node->op == GGML_OP_CONV_2D && node->op_params[STRIDE_PARAM] > UNIT_STRIDE;
    }

    bool accepts_every_op(const ggml_tensor*) {
        return true;
    }

    bool rejects_every_op(const ggml_tensor*) {
        return false;
    }

    bool rejects_strided_conv(const ggml_tensor* node) {
        return !is_strided_conv(node);
    }

    bool rejects_strided_conv_and_pool(const ggml_tensor* node) {
        return !is_strided_conv(node) && node->op != GGML_OP_POOL_2D;
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

    bool graph_has_op(ggml_cgraph* graph, ggml_op op) {
        for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            if (ggml_graph_node(graph, index)->op == op) {
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

    Inputs make_inputs(ggml_context* ctx, const Geometry& geometry) {
        return {ggml_new_tensor_4d(ctx, GGML_TYPE_F32, geometry.width, geometry.height, CHANNELS_IN, geometry.batch),
                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, geometry.kernel, geometry.kernel, CHANNELS_IN, CHANNELS_OUT),
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, CHANNELS_OUT)};
    }

    ggml_tensor* build_conv(ggml_context* ctx, const Inputs& in, const Geometry& geometry, const ggml_ext_op_support_t& supports) {
        return ggml_ext_conv_2d(ctx, in.x, in.w, in.b, geometry.s0, geometry.s1, geometry.padding, geometry.padding,
                                1, 1, true, false, false, 1.f, supports);
    }

    ggml_tensor* build_reference(ggml_context* ctx, const Inputs& in, const Geometry& geometry) {
        return build_conv(ctx, in, geometry, nullptr);
    }

    ggml_tensor* build_pointwise_as_linear(ggml_context* ctx, const Inputs& in, const Geometry&) {
        return ggml_ext_conv_2d_pointwise_as_linear(ctx, in.x, in.w, in.b);
    }

    ggml_tensor* build_identity(ggml_context* ctx, const Inputs& in, const Geometry&) {
        return ggml_cont(ctx, in.x);
    }

    ggml_tensor* build_token_round_trip(ggml_context* ctx, const Inputs& in, const Geometry& geometry) {
        return ggml_ext_tokens_to_image(ctx, ggml_ext_image_to_tokens(ctx, in.x), geometry.width, geometry.height);
    }

    ggml_tensor* build_without_strided_conv(ggml_context* ctx, const Inputs& in, const Geometry& geometry) {
        return build_conv(ctx, in, geometry, rejects_strided_conv);
    }

    std::vector<float> compute_conv(ggml_backend_t backend, const Geometry& geometry, ConvBuilder build) {
        Storage storage;
        Inputs in          = make_inputs(storage.ctx, geometry);
        ggml_tensor* out   = build(storage.ctx, in, geometry);
        ggml_cgraph* graph = build_graph(storage.ctx, out);
        storage.buffer     = ggml_backend_alloc_ctx_tensors(storage.ctx, backend);
        require(storage.buffer != nullptr, "Allocation failed");
        fill_tensor(in.x, 0);
        fill_tensor(in.w, 1);
        fill_tensor(in.b, 2);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Graph compute failed");
        return read_tensor(out);
    }

    void require_close(const std::vector<float>& expected, const std::vector<float>& actual, const char* message) {
        require(expected.size() == actual.size(), message);
        for (size_t index = 0; index < expected.size(); ++index) {
            require(std::isfinite(actual[index]) && std::abs(expected[index] - actual[index]) <= TOLERANCE, message);
        }
    }

    void test_subsampled_matches_strided(ggml_backend_t backend) {
        for (const Geometry& geometry : STRIDED) {
            require_close(compute_conv(backend, geometry, build_reference),
                          compute_conv(backend, geometry, build_without_strided_conv),
                          "Subsampled dense convolution differs from the strided convolution");
        }
    }

    void test_linear_matches_pointwise(ggml_backend_t backend) {
        for (const Geometry& geometry : POINTWISE) {
            require_close(compute_conv(backend, geometry, build_reference),
                          compute_conv(backend, geometry, build_pointwise_as_linear),
                          "Linear form differs from the pointwise convolution");
        }
    }

    void test_token_layout_round_trips(ggml_backend_t backend) {
        for (const Geometry& geometry : POINTWISE) {
            require_close(compute_conv(backend, geometry, build_identity),
                          compute_conv(backend, geometry, build_token_round_trip),
                          "Image to tokens and back must return the image");
        }
    }

    void test_pointwise_detection(ggml_backend_t backend) {
        Storage storage;
        Inputs pointwise = make_inputs(storage.ctx, POINTWISE[0]);
        Inputs dense     = make_inputs(storage.ctx, DENSE);
        require(ggml_ext_conv_2d_is_pointwise(pointwise.w, 1, 1, 0, 0), "A 1x1 unit-stride convolution is pointwise");
        require(!ggml_ext_conv_2d_is_pointwise(pointwise.w, 2, 2, 0, 0), "A strided 1x1 convolution is not pointwise");
        require(!ggml_ext_conv_2d_is_pointwise(pointwise.w, 1, 1, 1, 1), "A padded 1x1 convolution is not pointwise");
        require(!ggml_ext_conv_2d_is_pointwise(dense.w, 1, 1, 0, 0), "A 3x3 convolution is not pointwise");
        require(!sd_backend_prefers_linear_pointwise_conv(backend), "The CPU backend keeps pointwise convolutions");
        require(!sd_backend_prefers_linear_pointwise_conv(nullptr), "A null backend keeps pointwise convolutions");
    }

    bool selects_subsampling(const Geometry& geometry, const ggml_ext_op_support_t& supports) {
        Storage storage;
        Inputs in = make_inputs(storage.ctx, geometry);
        return graph_has_op(build_graph(storage.ctx, build_conv(storage.ctx, in, geometry, supports)), GGML_OP_POOL_2D);
    }

    void test_form_selection(ggml_backend_t backend) {
        const Geometry& strided = STRIDED[0];
        require(!selects_subsampling(strided, nullptr), "No support check must keep the strided convolution");
        require(!selects_subsampling(strided, ggml_ext_backend_op_support(backend)), "A backend with strided convolution must keep it");
        require(!selects_subsampling(strided, accepts_every_op), "A supported strided convolution must be kept");
        require(selects_subsampling(strided, rejects_strided_conv), "A backend without strided convolution must subsample a dense one");
        require(!selects_subsampling(strided, rejects_strided_conv_and_pool), "A backend without pooling must keep the strided convolution");
        require(!selects_subsampling(strided, rejects_every_op), "A backend without convolution must keep the strided convolution");
        require(!selects_subsampling(DENSE, rejects_every_op), "A dense convolution must never be subsampled");
    }
}

int main() {
    ggml_backend_load_all();
    SDBackendHandle backend(sd_backend_cpu_init());
    try {
        require(backend.get() != nullptr, "CPU backend unavailable");
        test_subsampled_matches_strided(backend.get());
        test_form_selection(backend.get());
        test_linear_matches_pointwise(backend.get());
        test_token_layout_round_trips(backend.get());
        test_pointwise_detection(backend.get());
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS strided convolution falls back to a subsampled dense one only when the backend lacks it\n";
    return 0;
}
