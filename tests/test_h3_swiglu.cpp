#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool run_case(int64_t features, int64_t columns) {
    ggml_context* ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    if (!ctx) {
        return false;
    }

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, features * 2, columns);
    ggml_tensor* first = ggml_cont(ctx, ggml_view_2d(ctx, input, features, columns, input->nb[1], 0));
    ggml_tensor* second = ggml_cont(ctx, ggml_view_2d(ctx, input, features, columns, input->nb[1], features * input->nb[0]));
    ggml_tensor* old_path = ggml_mul(ctx, ggml_silu(ctx, first), second);
    ggml_tensor* fused    = ggml_swiglu(ctx, input);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, old_path);
    ggml_build_forward_expand(graph, fused);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> values(static_cast<size_t>(features * 2 * columns));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((static_cast<int>(i * 17 % 257) - 128)) / 19.0f;
    }
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    bool passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> expected(static_cast<size_t>(features * columns));
    std::vector<float> actual(static_cast<size_t>(features * columns));
    if (passed) {
        ggml_backend_tensor_get(old_path, expected.data(), 0, ggml_nbytes(old_path));
        ggml_backend_tensor_get(fused, actual.data(), 0, ggml_nbytes(fused));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 2e-6f * (1.0f + std::fabs(expected[i]))) {
                std::fprintf(stderr, "SwiGLU mismatch at %zu: %.9g vs %.9g\n", i, actual[i], expected[i]);
                passed = false;
                break;
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return passed;
}

int main() {
    const bool passed = run_case(37, 1) && run_case(14336, 23) && run_case(14336, 257);
    if (passed) {
        std::puts("H3 fused SwiGLU test passed");
    }
    return passed ? 0 : 1;
}
