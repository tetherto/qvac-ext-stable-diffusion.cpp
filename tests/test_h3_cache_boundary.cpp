#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <array>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    const bool cuda        = argc == 2 && std::strcmp(argv[1], "--cuda") == 0;
    ggml_backend_t backend = nullptr;
    if (cuda) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#else
        return 77;
#endif
    } else {
        backend = ggml_backend_cpu_init();
    }
    if (!backend)
        return 1;
    auto ctx = ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true});
    if (!ctx)
        return 1;
    const std::array<float, 8> input{{1e8f, -1e8f, 16777216.f, -16777216.f, 10.f, -10.f, 0.25f, -0.25f}};
    const std::array<float, 8> output{{1.f, -1.f, 0.5f, -0.5f, 0.125f, -0.125f, 10.f, -10.f}};
    auto old_input  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input.size());
    auto old_output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input.size());
    auto current    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input.size());
    auto legacy     = ggml_add(ctx, current, ggml_sub(ctx, old_output, old_input));
    auto boundary   = ggml_add(ctx, old_output, ggml_sub(ctx, current, old_input));
    auto graph      = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, legacy);
    ggml_build_forward_expand(graph, boundary);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer)
        return 1;
    ggml_backend_tensor_set(old_input, input.data(), 0, sizeof(input));
    ggml_backend_tensor_set(old_output, output.data(), 0, sizeof(output));
    ggml_backend_tensor_set(current, input.data(), 0, sizeof(input));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        return 1;
    std::array<float, 8> legacy_result{}, boundary_result{};
    ggml_backend_tensor_get(legacy, legacy_result.data(), 0, sizeof(legacy_result));
    ggml_backend_tensor_get(boundary, boundary_result.data(), 0, sizeof(boundary_result));
    const bool regression_reproduced = std::memcmp(legacy_result.data(), output.data(), sizeof(output)) != 0;
    const bool exact                 = std::memcmp(boundary_result.data(), output.data(), sizeof(output)) == 0;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::printf("%s fresh cache cancellation regression reproduced=%d boundary_exact=%d\n", exact && regression_reproduced ? "PASS" : "FAIL", regression_reproduced, exact);
    return exact && regression_reproduced ? 0 : 1;
}
