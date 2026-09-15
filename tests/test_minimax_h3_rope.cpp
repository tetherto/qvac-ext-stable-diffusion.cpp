#include "model/diffusion/minimax_h3.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum class RopePattern { IDENTITY, QUARTER_TURN, GENERAL };

struct RopeCase {
    int64_t dimensions;
    int64_t rotated;
    int64_t heads;
    int64_t tokens;
    int64_t batches;
    bool strided;
    RopePattern pattern;
};

static size_t input_index(const RopeCase& shape, int64_t d, int64_t h, int64_t l, int64_t n) {
    const int64_t width = shape.dimensions + (shape.strided ? 8 : 0);
    return static_cast<size_t>(d + (shape.strided ? 3 : 0) + width * (h + shape.heads * (l + shape.tokens * n)));
}

static bool supports_fused_rope(ggml_backend_t backend, const RopeCase& shape) {
    ggml_init_params params = {1024 * 1024, nullptr, true};
    ggml_context* ctx       = ggml_init(params);
    if (!ctx) {
        return false;
    }
    auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape.rotated, shape.heads, shape.tokens, shape.batches);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, shape.rotated / 2, shape.tokens);
    const bool supported = ggml_backend_supports_op(backend, ggml_rope_flux(ctx, x, pe));
    ggml_free(ctx);
    return supported;
}

static bool run_path(ggml_backend_t backend,
                     const RopeCase& shape,
                     const std::vector<float>& input,
                     const std::vector<float>& positions,
                     bool use_fused,
                     bool expect_fused,
                     bool benchmark,
                     std::vector<float>* actual) {
    ggml_init_params params = {8 * 1024 * 1024, nullptr, true};
    ggml_context* ctx       = ggml_init(params);
    if (!ctx) {
        return false;
    }
    const int64_t width = shape.dimensions + (shape.strided ? 8 : 0);
    auto storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, shape.heads, shape.tokens, shape.batches);
    auto x = shape.strided ? ggml_view_4d(ctx, storage, shape.dimensions, shape.heads, shape.tokens, shape.batches,
                                         storage->nb[1], storage->nb[2], storage->nb[3], 3 * sizeof(float))
                           : storage;
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, shape.rotated / 2, shape.tokens);
    auto result = MiniMaxH3::apply_partial_rope(ctx, x, pe, use_fused ? backend : nullptr);
    auto graph = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(graph, result);
    int fused_nodes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        fused_nodes += ggml_graph_node(graph, i)->op == GGML_OP_ROPE_FLUX;
    }
    if ((fused_nodes != 0) != expect_fused) {
        std::fprintf(stderr, "unexpected RoPE graph: fused nodes=%d expected=%d\n", fused_nodes, expect_fused);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "could not allocate RoPE test tensors\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(storage, input.data(), 0, ggml_nbytes(storage));
    ggml_backend_tensor_set(pe, positions.data(), 0, ggml_nbytes(pe));
    bool passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (passed && benchmark) {
        constexpr int runs = 3;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < runs; ++i) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                passed = false;
                break;
            }
        }
        ggml_backend_synchronize(backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / runs;
        std::printf("H3 partial RoPE backend=%s path=%s D=%lld R=%lld H=%lld L=%lld N=%lld %.3f ms/run\n",
                    ggml_backend_name(backend), expect_fused ? "fused" : "fallback",
                    static_cast<long long>(shape.dimensions), static_cast<long long>(shape.rotated),
                    static_cast<long long>(shape.heads), static_cast<long long>(shape.tokens),
                    static_cast<long long>(shape.batches), ms);
    }
    if (passed) {
        actual->resize(ggml_nelements(result));
        ggml_backend_tensor_get(result, actual->data(), 0, ggml_nbytes(result));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

static bool run_case(ggml_backend_t backend, const RopeCase& shape, bool benchmark = false) {
    const int64_t width = shape.dimensions + (shape.strided ? 8 : 0);
    std::vector<float> input(width * shape.heads * shape.tokens * shape.batches, 5000.0f);
    std::vector<float> positions(2 * shape.rotated * shape.tokens);
    for (int64_t n = 0; n < shape.batches; ++n) {
        for (int64_t l = 0; l < shape.tokens; ++l) {
            for (int64_t h = 0; h < shape.heads; ++h) {
                for (int64_t d = 0; d < shape.dimensions; ++d) {
                    input[input_index(shape, d, h, l, n)] = d == shape.rotated ? -0.0f :
                        static_cast<float>((d * 11 + h * 7 + l * 13 + n * 17) % 251 - 125) * 0.015625f;
                }
            }
        }
    }
    for (int64_t l = 0; l < shape.tokens; ++l) {
        for (int64_t p = 0; p < shape.rotated / 2; ++p) {
            float* matrix = positions.data() + 4 * (p + (shape.rotated / 2) * l);
            if (shape.pattern == RopePattern::IDENTITY) {
                matrix[0] = matrix[3] = 1.0f;
                matrix[1] = matrix[2] = 0.0f;
            } else if (shape.pattern == RopePattern::QUARTER_TURN) {
                matrix[0] = matrix[3] = 0.0f;
                matrix[1] = -1.0f;
                matrix[2] = 1.0f;
            } else {
                // Non-symmetric matrices expose row/column swaps that identity
                // rotations, or accidentally paired inverse rotations, hide.
                const double angle = (p * 7 + l * 3) * 0.031;
                matrix[0] = static_cast<float>(0.81 + 0.17 * std::sin(angle));
                matrix[1] = static_cast<float>(-0.31 + 0.11 * std::cos(angle));
                matrix[2] = static_cast<float>(0.57 + 0.13 * std::sin(angle + 0.2));
                matrix[3] = static_cast<float>(0.73 - 0.19 * std::cos(angle + 0.3));
            }
        }
    }

    const bool fused_supported = supports_fused_rope(backend, shape) && std::getenv("GGML_ROPE_FLUX_DISABLE") == nullptr;
    std::vector<float> fallback;
    std::vector<float> selected;
    if (!run_path(backend, shape, input, positions, false, false, benchmark, &fallback) ||
        !run_path(backend, shape, input, positions, true, fused_supported, benchmark, &selected)) {
        return false;
    }
    double max_error = 0;
    for (int64_t n = 0; n < shape.batches; ++n) {
        for (int64_t h = 0; h < shape.heads; ++h) {
            for (int64_t l = 0; l < shape.tokens; ++l) {
                for (int64_t d = 0; d < shape.dimensions; ++d) {
                    const size_t index = d + shape.dimensions * (l + shape.tokens * (h + shape.heads * n));
                    const float original = input[input_index(shape, d, h, l, n)];
                    double expected = original;
                    if (d < shape.rotated) {
                        const int64_t pair = d % (shape.rotated / 2);
                        const int64_t component = d / (shape.rotated / 2);
                        const float* matrix = positions.data() + 4 * (pair + (shape.rotated / 2) * l);
                        expected = static_cast<double>(input[input_index(shape, pair, h, l, n)]) * matrix[component * 2] +
                                   static_cast<double>(input[input_index(shape, pair + shape.rotated / 2, h, l, n)]) * matrix[component * 2 + 1];
                    }
                    for (const auto* output : {&fallback, &selected}) {
                        const float value = (*output)[index];
                        const double error = std::abs(static_cast<double>(value) - expected);
                        max_error = std::max(max_error, error);
                        if (!std::isfinite(value) || error > 1e-5 * (1.0 + std::abs(expected)) ||
                            (d >= shape.rotated && std::memcmp(&value, &original, sizeof(float)) != 0)) {
                            std::fprintf(stderr, "H3 RoPE mismatch path=%s R=%lld strided=%d index=%zu: got %.9g expected %.9g\n",
                                         output == &fallback ? "fallback" : "selected", static_cast<long long>(shape.rotated),
                                         shape.strided, index, value, expected);
                            return false;
                        }
                    }
                }
            }
        }
    }
    std::printf("H3 RoPE passed backend=%s R=%lld pattern=%d strided=%d fused=%d max_abs=%.3g\n",
                ggml_backend_name(backend), static_cast<long long>(shape.rotated), static_cast<int>(shape.pattern),
                shape.strided, fused_supported, max_error);
    return true;
}

// Direct operator coverage complements H3's split-half adapter: arbitrary
// element/row strides, offset views, head/batch mapping, and null-PE permutation.
static bool run_direct_flux_case(ggml_backend_t backend, bool strided, bool null_pe) {
    constexpr int D = 14, H = 3, L = 7, B = 2;
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return false;
    auto storage = strided ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, D + 1, H + 1, (L + 1) * B)
                           : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, L, B);
    auto x = storage;
    size_t x_offset = 0;
    if (strided) {
        x_offset = sizeof(float);
        x = ggml_view_4d(ctx, storage, D, H, L, B, storage->nb[2], storage->nb[3],
                         storage->nb[3] * (L + 1), x_offset);
        x->nb[0] = 2 * sizeof(float);
    }
    auto pe_storage = strided ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 3, D / 2 + 1, L + 1)
                              : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, D / 2, L);
    auto pe = pe_storage;
    size_t pe_offset = 0;
    if (strided) {
        pe_offset = sizeof(float);
        pe = ggml_view_4d(ctx, pe_storage, 2, 2, D / 2, L, pe_storage->nb[1], pe_storage->nb[2],
                          pe_storage->nb[3], pe_offset);
        pe->nb[0] = 2 * sizeof(float);
    }
    auto result = ggml_rope_flux(ctx, x, null_pe ? nullptr : pe);
    if (!ggml_backend_supports_op(backend, result)) {
        std::printf("Direct ROPE_FLUX unsupported backend=%s strided=%d null_pe=%d (skipped)\n",
                    ggml_backend_name(backend), strided, null_pe);
        ggml_free(ctx);
        return true;
    }
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> input(ggml_nelements(storage), 5000.0f);
    std::vector<float> positions(ggml_nelements(pe_storage), 7000.0f);
    auto xi = [&](int d, int h, int l, int b) {
        return (x_offset + d * x->nb[0] + h * x->nb[1] + l * x->nb[2] + b * x->nb[3]) / sizeof(float);
    };
    auto pi = [&](int c, int out, int p, int l) {
        return (pe_offset + c * pe->nb[0] + out * pe->nb[1] + p * pe->nb[2] + l * pe->nb[3]) / sizeof(float);
    };
    for (int b = 0; b < B; ++b) for (int l = 0; l < L; ++l) for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) input[xi(d,h,l,b)] = ((d*11+h*7+l*13+b*17)%67-33)*0.03125f;
    for (int l = 0; l < L; ++l) for (int p = 0; p < D / 2; ++p)
        for (int out = 0; out < 2; ++out) for (int c = 0; c < 2; ++c)
            positions[pi(c,out,p,l)] = (c*3-out*5+p*7+l*11-23)*0.015625f;
    ggml_backend_tensor_set(storage, input.data(), 0, ggml_nbytes(storage));
    ggml_backend_tensor_set(pe_storage, positions.data(), 0, ggml_nbytes(pe_storage));
    bool passed = true;
    std::vector<float> actual(ggml_nelements(result));
    for (int repeat = 0; repeat < 2 && passed; ++repeat) {
        passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (!passed) break;
        ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
        for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) for (int l = 0; l < L; ++l)
            for (int d = 0; d < D; ++d) {
                const int p = d / 2, out = d % 2;
                const float expected = null_pe ? input[xi(d,h,l,b)] :
                    input[xi(2*p,h,l,b)] * positions[pi(0,out,p,l)] +
                    input[xi(2*p+1,h,l,b)] * positions[pi(1,out,p,l)];
                const float got = actual[d + D * (l + L * (h + H*b))];
                if (!std::isfinite(got) || std::fabs(got-expected) > 1e-6f * (1+std::fabs(expected))) {
                    std::fprintf(stderr, "Direct ROPE_FLUX mismatch strided=%d null_pe=%d d=%d h=%d l=%d b=%d got=%g expected=%g\n",
                                 strided, null_pe, d,h,l,b,got,expected);
                    passed = false;
                }
            }
    }
    std::printf("Direct ROPE_FLUX backend=%s strided=%d null_pe=%d %s\n", ggml_backend_name(backend),
                strided, null_pe, passed ? "passed" : "FAILED");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

int main(int argc, char** argv) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name(argc > 1 ? argv[1] : "CPU", nullptr);
    if (!backend) {
        std::fprintf(stderr, "requested RoPE backend is not available\n");
        return 1;
    }
    bool passed = true;
    for (bool strided : {false, true}) for (bool null_pe : {false, true})
        passed = run_direct_flux_case(backend, strided, null_pe) && passed;
    for (int64_t rotated : {96, 128, 32}) {
        for (auto pattern : {RopePattern::IDENTITY, RopePattern::QUARTER_TURN, RopePattern::GENERAL}) {
            for (bool strided : {false, true}) {
                passed = run_case(backend, {128, rotated, 3, 7, 2, strided, pattern}) && passed;
            }
        }
    }
    if (std::getenv("SD_MINIMAX_H3_ROPE_BENCH")) {
        passed = run_case(backend, {128, 96, 56, 15422, 1, true, RopePattern::GENERAL}, true) && passed;
    }
    ggml_backend_free(backend);
    if (passed) {
        std::puts("MiniMax H3 partial RoPE tests passed");
    }
    return passed ? 0 : 1;
}
