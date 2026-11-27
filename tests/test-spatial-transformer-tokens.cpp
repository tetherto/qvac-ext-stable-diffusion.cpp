#include <ggml-backend.h>
#include <ggml.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/ggml_runner.h"
#include "model/common/block.hpp"

namespace spatial_transformer_tokens_test {
    constexpr size_t PARAMS_CONTEXT_BYTES  = 8 * 1024 * 1024;
    constexpr size_t COMPUTE_CONTEXT_BYTES = 32 * 1024 * 1024;
    constexpr size_t GRAPH_NODES           = 1024;
    constexpr int64_t WIDTH                = 6;
    constexpr int64_t HEIGHT               = 4;
    constexpr int64_t BATCH                = 2;
    constexpr int64_t CHANNELS             = 32;
    constexpr int64_t HEADS                = 2;
    constexpr int64_t HEAD_DIM             = 16;
    constexpr int64_t DEPTH                = 1;
    constexpr int64_t CONTEXT_DIM          = 16;
    constexpr int64_t CONTEXT_TOKENS       = 5;
    constexpr int VALUE_STEP               = 37;
    constexpr int VALUE_RANGE              = 251;
    constexpr float WEIGHT_SCALE           = 500.0f;
    constexpr float INPUT_SCALE            = 17.0f;
    constexpr float TOLERANCE              = 2e-3f;
    constexpr int WEIGHT_SEED              = 3;
    constexpr int INPUT_SEED               = 0;
    constexpr int CONTEXT_SEED             = 1;
    const char* const PREFIX               = "transformer";
    const char* const PROJ_IN_WEIGHT       = "transformer.proj_in.weight";

    struct Storage {
        ggml_context* ctx            = nullptr;
        ggml_backend_buffer_t buffer = nullptr;

        explicit Storage(size_t bytes)
            : ctx(ggml_init({bytes, nullptr, true})) {}
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

    std::vector<float> make_values(size_t count, int seed, float scale) {
        std::vector<float> values(count);
        for (size_t index = 0; index < count; ++index) {
            values[index] = float(int((index * VALUE_STEP + seed) % VALUE_RANGE) - VALUE_RANGE / 2) / scale;
        }
        return values;
    }

    void fill_tensor(ggml_tensor* tensor, int seed, float scale) {
        std::vector<float> values = make_values(ggml_nelements(tensor), seed, scale);
        if (tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> halves(values.size());
            ggml_fp32_to_fp16_row(values.data(), halves.data(), static_cast<int64_t>(values.size()));
            ggml_backend_tensor_set(tensor, halves.data(), 0, ggml_nbytes(tensor));
            return;
        }
        require(tensor->type == GGML_TYPE_F32, "Unexpected parameter type");
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }

    void fill_params(const std::map<std::string, ggml_tensor*>& params) {
        int seed = WEIGHT_SEED;
        for (const auto& entry : params) {
            fill_tensor(entry.second, seed++, WEIGHT_SCALE);
        }
    }

    bool graph_has_op(ggml_cgraph* graph, ggml_op op) {
        for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            if (ggml_graph_node(graph, index)->op == op) {
                return true;
            }
        }
        return false;
    }

    struct BlockRun {
        std::vector<float> output;
        bool has_convolution;
    };

    struct Fixture {
        SDBackendHandle backend{sd_backend_cpu_init()};
        Storage params{PARAMS_CONTEXT_BYTES};
        SpatialTransformer block{CHANNELS, HEADS, HEAD_DIM, DEPTH, CONTEXT_DIM, false};
        std::map<std::string, ggml_tensor*> tensors;

        Fixture() {
            require(backend.get() != nullptr, "CPU backend unavailable");
            block.init(params.ctx, {}, PREFIX);
            block.get_param_tensors(tensors, PREFIX);
            params.buffer = ggml_backend_alloc_ctx_tensors(params.ctx, backend.get());
            require(params.buffer != nullptr, "Parameter allocation failed");
            fill_params(tensors);
        }

        BlockRun run(bool pointwise_conv_as_linear) {
            Storage compute(COMPUTE_CONTEXT_BYTES);
            GGMLRunnerContext runner;
            runner.backend                  = backend.get();
            runner.ggml_ctx                 = compute.ctx;
            runner.conv2d_direct_enabled    = true;
            runner.pointwise_conv_as_linear = pointwise_conv_as_linear;

            ggml_tensor* x       = ggml_new_tensor_4d(compute.ctx, GGML_TYPE_F32, WIDTH, HEIGHT, CHANNELS, BATCH);
            ggml_tensor* context = ggml_new_tensor_3d(compute.ctx, GGML_TYPE_F32, CONTEXT_DIM, CONTEXT_TOKENS, BATCH);
            ggml_tensor* out     = block.forward(&runner, x, context);
            ggml_cgraph* graph   = ggml_new_graph_custom(compute.ctx, GRAPH_NODES, false);
            ggml_build_forward_expand(graph, out);
            compute.buffer = ggml_backend_alloc_ctx_tensors(compute.ctx, backend.get());
            require(compute.buffer != nullptr, "Compute allocation failed");
            fill_tensor(x, INPUT_SEED, INPUT_SCALE);
            fill_tensor(context, CONTEXT_SEED, INPUT_SCALE);
            require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "Graph compute failed");

            BlockRun result{std::vector<float>(ggml_nelements(out)), graph_has_op(graph, GGML_OP_CONV_2D)};
            ggml_backend_tensor_get(out, result.output.data(), 0, ggml_nbytes(out));
            return result;
        }
    };

    void require_close(const std::vector<float>& expected, const std::vector<float>& actual, const char* message) {
        require(expected.size() == actual.size(), message);
        for (size_t index = 0; index < expected.size(); ++index) {
            const float allowed = TOLERANCE * (1.0f + std::abs(expected[index]));
            require(std::isfinite(actual[index]) && std::abs(expected[index] - actual[index]) <= allowed, message);
        }
    }

    void test_token_projection_matches_convolution() {
        Fixture fixture;
        require(fixture.tensors.at(PROJ_IN_WEIGHT)->type == GGML_TYPE_F16, "The projection must keep its production F16 weights");

        const BlockRun convolution = fixture.run(false);
        const BlockRun tokens      = fixture.run(true);
        require(convolution.has_convolution, "The reference block must project with convolutions");
        require(!tokens.has_convolution, "The token branch must replace both projections");
        require_close(convolution.output, tokens.output, "The token branch differs from the convolution path");
    }

    int run() {
        ggml_backend_load_all();
        try {
            test_token_projection_matches_convolution();
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << error.what() << "\n";
            return 1;
        }
        std::cout << "PASS the spatial transformer gives the same output with its projections in the token layout\n";
        return 0;
    }
}

int main() {
    return spatial_transformer_tokens_test::run();
}
