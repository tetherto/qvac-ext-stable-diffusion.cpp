#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/diffusion/minimax_h3.hpp"

using Context   = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer    = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

static void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

static Context make_context() {
    Context context(ggml_init({ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false), nullptr, true}), ggml_free);
    require(context != nullptr, "context allocation failed");
    return context;
}

static void upload(ggml_tensor* tensor, const std::vector<float>& values) {
    require(tensor->type == GGML_TYPE_F32 && ggml_nelements(tensor) == static_cast<int64_t>(values.size()), "input shape mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static std::vector<unsigned char> download_bytes(ggml_tensor* tensor) {
    require(ggml_is_contiguous(tensor), "retained output is not contiguous");
    std::vector<unsigned char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    require(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16, "unexpected retained type");
    const size_t size = ggml_type_size(tensor->type);
    for (size_t offset = 0; offset < bytes.size(); offset += size) {
        float value;
        if (tensor->type == GGML_TYPE_F32)
            std::memcpy(&value, bytes.data() + offset, size);
        else {
            ggml_fp16_t half;
            std::memcpy(&half, bytes.data() + offset, size);
            value = ggml_fp16_to_fp32(half);
        }
        require(std::isfinite(value), "non-finite retained value");
    }
    return bytes;
}

static void compare_bytes(const std::vector<unsigned char>& actual, const std::vector<unsigned char>& expected, const std::string& label) {
    require(actual == expected, label + ": byte equality failed");
}

static std::vector<float> make_values(int64_t count, int seed, float scale = 1.f) {
    std::vector<float> values(count);
    for (int64_t i = 0; i < count; ++i)
        values[i] = scale * (static_cast<float>((i * 17 + seed * 29) % 251) - 125.f) / 128.f;
    return values;
}

static std::vector<float> position_values(int64_t tokens, int rotation, int seed, bool arbitrary = false) {
    if (arbitrary)
        return make_values(2 * rotation * tokens, seed, 0.5f);
    std::vector<std::vector<float>> ids;
    for (int64_t t = 0; t < tokens; ++t)
        ids.push_back({0.2f * t + 0.01f * seed, -0.3f * t + 0.02f * seed, 0.1f * t * t - 0.03f * seed});
    return Rope::embed_nd(ids, 1, 10000.f, rotation == 96 ? std::vector<int>{32, 32, 32} : std::vector<int>{32, 32, 64});
}

struct Graph {
    Context context    = make_context();
    ggml_cgraph* graph = ggml_new_graph_custom(context.get(), 4096, false);
    Allocator allocator;

    explicit Graph(ggml_backend_t backend)
        : allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {}

    ggml_tensor* input(int64_t n0, int64_t n1, int64_t n2, int64_t n3 = 1) {
        auto tensor = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, n0, n1, n2, n3);
        ggml_set_input(tensor);
        return tensor;
    }

    void retain(ggml_tensor* tensor) {
        ggml_set_output(tensor);
        if (tensor->view_src != nullptr)
            ggml_set_output(tensor->view_src);
        ggml_build_forward_expand(graph, tensor);
    }

    void allocate(ggml_backend_t backend) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), std::string("unsupported operation: ") + ggml_op_name(ggml_graph_node(graph, i)->op));
        require(ggml_gallocr_alloc_graph(allocator.get(), graph), "graph allocation failed");
    }

    void compute(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
    }
};

enum class InputLayout { Packed,
                         Padded,
                         ChannelStride };

struct InputView {
    ggml_tensor* base;
    ggml_tensor* tensor;
};

static InputView head_input(Graph& graph, int64_t heads, int64_t tokens, int64_t batch, InputLayout layout) {
    if (layout == InputLayout::Packed) {
        auto input = graph.input(128, heads, tokens, batch);
        return {input, input};
    }
    if (layout == InputLayout::Padded) {
        auto base   = graph.input(139, heads + 2, tokens + 1, batch);
        auto tensor = ggml_view_4d(graph.context.get(), base, 128, heads, tokens, batch, base->nb[1], base->nb[2], base->nb[3], 3 * sizeof(float) + base->nb[1]);
        return {base, tensor};
    }
    auto base   = graph.input(2 * heads + 7, 133, tokens + 1, batch);
    auto tensor = ggml_view_4d(graph.context.get(), base, heads, 128, tokens, batch, base->nb[1], base->nb[2], base->nb[3], sizeof(float) + 3 * base->nb[1]);
    return {base, ggml_permute(graph.context.get(), tensor, 1, 0, 2, 3)};
}

static InputView positional_input(Graph& graph, int64_t tokens, int rotation, InputLayout layout) {
    if (layout == InputLayout::Packed) {
        auto input = graph.input(2, 2, rotation / 2, tokens);
        return {input, input};
    }
    auto base   = graph.input(5, 4, rotation / 2 + 3, tokens + 1);
    auto tensor = ggml_view_4d(graph.context.get(), base, 2, 2, rotation / 2, tokens, base->nb[1], base->nb[2], base->nb[3], sizeof(float) + base->nb[1] + base->nb[2]);
    if (layout == InputLayout::ChannelStride)
        tensor = ggml_permute(graph.context.get(), tensor, 1, 0, 2, 3);
    return {base, tensor};
}

static std::vector<unsigned char> upload_view(const InputView& input, const std::vector<float>& values) {
    require(ggml_nelements(input.tensor) == static_cast<int64_t>(values.size()), "logical upload shape mismatch");
    require(input.tensor == input.base || input.tensor->view_src == input.base, "view does not reference its upload base");
    std::vector<float> physical(ggml_nelements(input.base), 123.75f);
    size_t index = 0;
    for (int64_t i3 = 0; i3 < input.tensor->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < input.tensor->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < input.tensor->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < input.tensor->ne[0]; ++i0) {
                    const size_t offset = input.tensor->view_offs + i0 * input.tensor->nb[0] + i1 * input.tensor->nb[1] + i2 * input.tensor->nb[2] + i3 * input.tensor->nb[3];
                    require(offset % sizeof(float) == 0 && offset / sizeof(float) < physical.size(), "view upload exceeds backing storage");
                    physical[offset / sizeof(float)] = values[index++];
                }
    upload(input.base, physical);
    std::vector<unsigned char> bytes(physical.size() * sizeof(float));
    std::memcpy(bytes.data(), physical.data(), bytes.size());
    return bytes;
}

static std::vector<ggml_tensor*> reachable(ggml_tensor* output) {
    auto context = make_context();
    auto graph   = ggml_new_graph_custom(context.get(), 4096, false);
    ggml_build_forward_expand(graph, output);
    std::vector<ggml_tensor*> nodes;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        nodes.push_back(ggml_graph_node(graph, i));
    return nodes;
}

static int count(ggml_tensor* output, ggml_op op) {
    int result = 0;
    for (auto node : reachable(output))
        result += node->op == op;
    return result;
}

// Freeze the pre-optimization helper independently of its new slice policy.
static ggml_tensor* legacy_partial_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe) {
    const int64_t rotation = pe->ne[2] * 2;
    auto rotated           = Rope::apply_rope(ctx, ggml_ext_slice(ctx, x, 0, 0, rotation), pe, false);
    if (rotation == x->ne[0])
        return rotated;
    auto tail = MiniMaxH3::attention_layout(ctx, ggml_ext_slice(ctx, x, 0, rotation, x->ne[0]));
    return ggml_concat(ctx, rotated, tail, 0);
}

static void scalar_rotation(ggml_tensor* output, const std::vector<float>& input, const std::vector<float>& pe, int64_t heads, int64_t tokens, int64_t batch, int rotation) {
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    const int half  = rotation / 2;
    float max_error = 0.f;
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t t = 0; t < tokens; ++t)
                for (int lane = 0; lane < 128; ++lane) {
                    const size_t source      = 128 * (h + heads * (t + tokens * b));
                    const size_t destination = lane + 128 * (t + tokens * (h + heads * b));
                    float expected           = input[source + lane];
                    if (lane < rotation) {
                        const int pair      = lane % half;
                        const size_t matrix = 4 * (pair + half * t) + (lane < half ? 0 : 2);
                        expected            = input[source + pair] * pe[matrix] + input[source + pair + half] * pe[matrix + 1];
                    }
                    const float error = std::fabs(actual[destination] - expected);
                    require(std::isfinite(actual[destination]) && error <= 2e-6f * (1.f + std::fabs(expected)), "scalar noninterleaved RoPE mismatch");
                    if (lane >= rotation)
                        require(std::memcmp(&actual[destination], &expected, sizeof(float)) == 0, "unrotated tail changed");
                    max_error = std::max(max_error, error);
                }
    std::printf("scalar rotation=%d heads=%lld tokens=%lld batch=%lld max_abs=%.9g\n", rotation, static_cast<long long>(heads), static_cast<long long>(tokens), static_cast<long long>(batch), max_error);
}

static void test_helper(ggml_backend_t backend, int64_t tokens, int64_t batch, int rotation, bool arbitrary, InputLayout layout) {
    std::printf("helper tokens=%lld batch=%lld rotation=%d arbitrary=%d layout=%d\n", static_cast<long long>(tokens), static_cast<long long>(batch), rotation, arbitrary, static_cast<int>(layout));
    std::fflush(stdout);
    Graph graph(backend);
    constexpr int heads = 56;
    auto input          = head_input(graph, heads, tokens, batch, layout);
    auto pe             = positional_input(graph, tokens, rotation, layout);
    // The old channel slice helper assumes contiguous channels.
    auto legacy_input = layout == InputLayout::ChannelStride ? ggml_cont(graph.context.get(), input.tensor) : input.tensor;
    auto reference    = legacy_partial_rope(graph.context.get(), legacy_input, pe.tensor);
    auto copied       = MiniMaxH3::apply_partial_rope(graph.context.get(), legacy_input, pe.tensor);
    auto optimized    = MiniMaxH3::apply_partial_rope(graph.context.get(), input.tensor, pe.tensor, true);
    require(count(reference, GGML_OP_CONT) == count(copied, GGML_OP_CONT), "legacy helper structure changed");
    require(count(reference, GGML_OP_CONT) - count(optimized, GGML_OP_CONT) == (rotation == 128 ? 0 : 2), "unexpected broadcast helper materializations");
    require(count(reference, GGML_OP_REPEAT) == 2 && count(optimized, GGML_OP_REPEAT) == 0, "optimized helper must broadcast coefficients without REPEAT");
    require(count(optimized, GGML_OP_MUL) == 4 && count(optimized, GGML_OP_ADD) == 2, "optimized helper must retain all four PE coefficients");
    for (auto output : {reference, copied, optimized})
        graph.retain(output);
    if (layout != InputLayout::Packed) {
        graph.retain(input.base);
        graph.retain(pe.base);
    }
    graph.allocate(backend);
    std::vector<unsigned char> first;
    for (int seed : {11, 23, 11}) {
        auto values             = make_values(ggml_nelements(input.tensor), seed);
        auto positions          = position_values(tokens, rotation, seed, arbitrary);
        const auto input_before = upload_view(input, values);
        const auto pe_before    = upload_view(pe, positions);
        graph.compute(backend);
        const auto actual = download_bytes(optimized);
        compare_bytes(actual, download_bytes(reference), "broadcast helper versus frozen legacy");
        compare_bytes(actual, download_bytes(copied), "broadcast helper versus default policy");
        scalar_rotation(optimized, values, positions, heads, tokens, batch, rotation);
        if (layout != InputLayout::Packed) {
            compare_bytes(download_bytes(input.base), input_before, "retained input and padding must not change");
            compare_bytes(download_bytes(pe.base), pe_before, "retained PE and padding must not change");
        }
        if (first.empty())
            first = actual;
        else if (seed == 11)
            compare_bytes(actual, first, "helper repeated A/B/A");
        else
            require(actual != first, "changing helper inputs had no effect");
    }
}

struct AttentionParameters {
    Context context = make_context();
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    MiniMaxH3::Attention attention{288, 3, 128, 1e-5f};

    AttentionParameters(ggml_backend_t backend, bool half) {
        String2TensorStorage storage;
        for (const char* name : {"qkv_proj.weight", "out_proj.weight"})
            storage[name].type = half ? GGML_TYPE_F16 : GGML_TYPE_F32;
        attention.init(context.get(), storage);
        std::map<std::string, ggml_tensor*> parameters;
        attention.get_param_tensors(parameters);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(buffer != nullptr, "parameter allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto& [name, tensor] : parameters) {
            auto values = make_values(ggml_nelements(tensor), static_cast<int>(name.size()), 1.f / 64.f);
            if (name == "q_norm.weight" || name == "k_norm.weight")
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = 1.f + static_cast<float>(i % 7) / 128.f;
            if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> packed(values.size());
                for (size_t i = 0; i < values.size(); ++i)
                    packed[i] = ggml_fp32_to_fp16(values[i]);
                ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size() * sizeof(packed[0]));
            } else
                upload(tensor, values);
        }
    }
};

static void test_attention(ggml_backend_t backend, ggml_backend_t cpu, MiniMaxH3::Attention& attention, int64_t tokens, int64_t batch, int rotation, bool flash, bool capture_inputs, bool arbitrary) {
    Graph graph(backend);
    auto input = graph.input(288, tokens, batch);
    auto pe    = graph.input(2, 2, rotation / 2, tokens);
    std::vector<ggml_tensor*> outputs, flash_inputs;
    for (auto construction_backend : {cpu, backend}) {
        GGMLRunnerContext runner;
        runner.ggml_ctx           = graph.context.get();
        runner.backend            = construction_backend;
        runner.flash_attn_enabled = flash;
        outputs.push_back(attention.forward(&runner, input, pe));
        graph.retain(outputs.back());
        require(count(outputs.back(), GGML_OP_FLASH_ATTN_EXT) == (flash ? 1 : 0), "actual attention flash policy changed");
        if (capture_inputs) {
            for (auto node : reachable(outputs.back())) {
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    for (int index : {0, 1, 2})
                        flash_inputs.push_back(node->src[index]);
                }
            }
        }
    }
    const bool cuda      = sd_backend_is(backend, "CUDA");
    const int old_copies = count(outputs[0], GGML_OP_CONT), new_copies = count(outputs[1], GGML_OP_CONT);
    std::printf("actual attention tokens=%lld batch=%lld rotation=%d FA=%d capture=%d arbitrary=%d CONT=%d->%d\n",
                static_cast<long long>(tokens), static_cast<long long>(batch), rotation, flash, capture_inputs, arbitrary, old_copies, new_copies);
    std::fflush(stdout);
    require(old_copies - new_copies == (cuda && rotation == 96 ? 4 : 0), "unexpected broadcast Attention materializations");
    require(count(outputs[0], GGML_OP_REPEAT) == 4 && count(outputs[1], GGML_OP_REPEAT) == (cuda ? 0 : 4), "actual Attention must select broadcast RoPE only on CUDA");
    if (capture_inputs) {
        require(flash_inputs.size() == 6, "actual FA input capture missing");
        for (int index = 0; index < 3; ++index) {
            auto old_input = flash_inputs[index], new_input = flash_inputs[index + 3];
            require(old_input->type == (index == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16), "unexpected actual FA input type");
            require(old_input->type == new_input->type && ggml_are_same_shape(old_input, new_input), "actual FA input layout changed");
        }
        for (auto tensor : flash_inputs)
            graph.retain(tensor);
    }
    graph.allocate(backend);
    std::vector<std::vector<unsigned char>> first;
    for (int seed : {11, 23, 11}) {
        upload(input, make_values(ggml_nelements(input), seed));
        upload(pe, position_values(tokens, rotation, seed, arbitrary));
        graph.compute(backend);
        std::vector<std::vector<unsigned char>> actual;
        for (auto output : outputs)
            actual.push_back(download_bytes(output));
        compare_bytes(actual[1], actual[0], "actual Attention legacy versus automatic policy");
        if (capture_inputs) {
            for (auto tensor : flash_inputs)
                actual.push_back(download_bytes(tensor));
            for (size_t index = 0; index < 3; ++index)
                compare_bytes(actual[2 + index], actual[5 + index], "actual packed FA Q/K/V input");
        }
        if (first.empty())
            first = actual;
        else if (seed == 11) {
            for (size_t i = 0; i < actual.size(); ++i)
                compare_bytes(actual[i], first[i], "actual Attention repeated A/B/A");
        } else
            require(actual[0] != first[0], "changing Attention inputs had no effect");
    }
}

int main(int argc, char** argv) {
    try {
        const std::string name = argc > 1 ? argv[1] : "CPU";
        require(argc <= 2 && (name == "CPU" || name == "CUDA0"), "usage: test-h3-attention [CPU|CUDA0]");
        if (name == "CUDA0") {
            const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
            require(visible != nullptr && std::string(visible) == "0", "CUDA0 requires CUDA_VISIBLE_DEVICES=0");
        }
        // Both construction backends must retain the same unfused V layout.
#ifdef _WIN32
        require(_putenv_s("GGML_ROPE_FLUX_DISABLE", "1") == 0, "failed to set attention layout");
#else
        require(setenv("GGML_ROPE_FLUX_DISABLE", "1", 1) == 0, "failed to set attention layout");
#endif
        ggml_backend_load_all();
        using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
        Backend backend(ggml_backend_init_by_name(name.c_str(), nullptr), ggml_backend_free);
        Backend cpu(ggml_backend_init_by_name("CPU", nullptr), ggml_backend_free);
        require(backend != nullptr && cpu != nullptr, "requested backend unavailable");
        sd_backend_cpu_set_n_threads(backend.get(), 2);
        for (int rotation : {96, 128})
            for (int tokens : {1, 5})
                for (bool arbitrary : {false, true})
                    for (auto layout : {InputLayout::Packed, InputLayout::Padded, InputLayout::ChannelStride})
                        test_helper(backend.get(), tokens, 2, rotation, arbitrary, layout);
        for (bool half : {false, true}) {
            AttentionParameters parameters(backend.get(), half);
            std::printf("Actual Attention projection weights: %s\n", half ? "F16" : "F32");
            for (int rotation : {96, 128}) {
                for (bool flash : {false, true}) {
                    for (int tokens : {1, 5}) {
                        for (bool arbitrary : {false, true}) {
                            test_attention(backend.get(), cpu.get(), parameters.attention, tokens, 2, rotation, flash, false, arbitrary);
                            if (flash)
                                test_attention(backend.get(), cpu.get(), parameters.attention, tokens, 2, rotation, flash, true, arbitrary);
                        }
                    }
                }
            }
        }
        std::printf("H3 scalar RoPE, actual Attention outputs and packed FA inputs passed on %s\n", name.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
