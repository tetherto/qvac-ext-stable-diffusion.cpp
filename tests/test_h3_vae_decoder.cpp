#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/vae/minimax_h3_vae.hpp"

using ContextHandle   = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using BufferHandle    = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using AllocatorHandle = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

static void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

static ContextHandle make_context() {
    ggml_init_params params = {};
    params.mem_size         = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
    params.no_alloc         = true;
    ContextHandle context(ggml_init(params), ggml_free);
    require(context != nullptr, "context allocation failed");
    return context;
}

static void upload(ggml_tensor* tensor, const std::vector<float>& values) {
    require(tensor->type == GGML_TYPE_F32 && ggml_nelements(tensor) == static_cast<int64_t>(values.size()), "input shape mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static std::vector<float> download(ggml_tensor* tensor) {
    require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "unexpected output layout");
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

static std::vector<float> download_float(ggml_tensor* tensor) {
    if (tensor->type == GGML_TYPE_F32)
        return download(tensor);
    require(tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(tensor), "unexpected F16 attention input layout");
    std::vector<ggml_fp16_t> half(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, half.data(), 0, half.size() * sizeof(half[0]));
    std::vector<float> values(half.size());
    for (size_t i = 0; i < half.size(); ++i)
        values[i] = ggml_fp16_to_fp32(half[i]);
    return values;
}

static void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label, float tolerance = 2e-5f) {
    require(actual.size() == expected.size(), label + ": size mismatch");
    float max_error = 0.f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        max_error         = std::max(max_error, error);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error > tolerance * (1.f + std::fabs(expected[i]))) {
            throw std::runtime_error(label + ": mismatch at " + std::to_string(i) + ", got " + std::to_string(actual[i]) + ", expected " + std::to_string(expected[i]));
        }
    }
    std::printf("%s: max_absolute=%.9g\n", label.c_str(), max_error);
}

struct Graph {
    ContextHandle context = make_context();
    ggml_cgraph* graph    = ggml_new_graph_custom(context.get(), 4096, false);
    AllocatorHandle allocator;

    explicit Graph(ggml_backend_t backend)
        : allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {}

    ggml_tensor* input(int64_t n0, int64_t n1, int64_t n2 = 1, int64_t n3 = 1) {
        auto tensor = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, n0, n1, n2, n3);
        ggml_set_input(tensor);
        return tensor;
    }

    void output(ggml_tensor* tensor) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph, tensor);
    }

    void allocate(ggml_backend_t backend) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto tensor = ggml_graph_node(graph, i);
            require(ggml_backend_supports_op(backend, tensor), std::string("unsupported operation: ") + ggml_op_name(tensor->op));
        }
        require(ggml_gallocr_alloc_graph(allocator.get(), graph), "graph allocation failed");
    }

    void compute(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
    }
};

struct Structure {
    int copies   = 0;
    int swiglu   = 0;
    int silu     = 0;
    int multiply = 0;
    int flash    = 0;
    std::vector<ggml_tensor*> nodes;
};

static Structure structure(ggml_tensor* output) {
    auto context = make_context();
    auto graph   = ggml_new_graph_custom(context.get(), 4096, false);
    ggml_build_forward_expand(graph, output);
    Structure result;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        result.nodes.push_back(node);
        result.copies += node->op == GGML_OP_CONT;
        result.flash += node->op == GGML_OP_FLASH_ATTN_EXT;
        result.swiglu += node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU;
        result.silu += node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU;
        result.multiply += node->op == GGML_OP_MUL;
    }
    return result;
}

static bool is_qkv_view(ggml_tensor* tensor, int64_t tokens, int64_t batch) {
    return tensor->op == GGML_OP_VIEW && tensor->ne[0] == 64 && tensor->ne[1] == 32 &&
           tensor->ne[2] == tokens && tensor->ne[3] == batch && tensor->nb[0] == sizeof(float) &&
           tensor->nb[1] == 192 * sizeof(float) && tensor->nb[2] == 6144 * sizeof(float) &&
           tensor->nb[3] == 6144 * tokens * sizeof(float);
}

static std::vector<size_t> qkv_materializations(const Structure& graph, int64_t tokens, int64_t batch) {
    std::vector<size_t> offsets;
    for (auto node : graph.nodes) {
        if (node->op == GGML_OP_CONT && is_qkv_view(node->src[0], tokens, batch))
            offsets.push_back(node->src[0]->view_offs);
    }
    std::sort(offsets.begin(), offsets.end());
    return offsets;
}

static void check_qkv_layout(const Structure& graph, int64_t tokens, int64_t batch) {
    std::vector<size_t> qk_offsets;
    bool found_v = false;
    for (auto node : graph.nodes) {
        if (node->op == GGML_OP_RMS_NORM) {
            auto input = node->src[0];
            require(is_qkv_view(input, tokens, batch),
                    "Q/K RMSNorm must consume actual head-interleaved views");
            qk_offsets.push_back(input->view_offs);
        }
        if (is_qkv_view(node, tokens, batch) && node->view_offs == 128 * sizeof(float)) {
            found_v = true;
        }
    }
    std::sort(qk_offsets.begin(), qk_offsets.end());
    require(qk_offsets == std::vector<size_t>{0, 64 * sizeof(float)}, "Q/K head-local offsets differ from [Q64,K64,V64] packing");
    require(found_v, "reachable attention graph lost the head-interleaved V view");
}

static ggml_tensor* gate_activation(const Structure& graph) {
    ggml_tensor* activation = nullptr;
    for (auto node : graph.nodes) {
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->ne[0] == 8192 && node->src[0]->ne[1] == 2048) {
            require(activation == nullptr, "multiple output projections in feedforward graph");
            activation = node->src[1];
        }
    }
    require(activation != nullptr && activation->ne[0] == 8192, "missing actual pre-w2 gate activation");
    return activation;
}

struct Parameters {
    ContextHandle context = make_context();
    BufferHandle buffer{nullptr, ggml_backend_buffer_free};
    std::map<std::string, ggml_tensor*> tensors;

    void initialize(GGMLBlock& block, ggml_backend_t backend, bool half, const std::vector<std::string>& weights) {
        String2TensorStorage storage;
        for (const auto& name : weights)
            storage[name].type = half ? GGML_TYPE_F16 : GGML_TYPE_F32;
        block.init(context.get(), storage);
        block.get_param_tensors(tensors);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(buffer != nullptr, "parameter allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    void fill(const std::function<float(const std::string&, size_t)>& value) {
        constexpr size_t chunk_size = 65536;
        std::vector<float> values(chunk_size);
        std::vector<ggml_fp16_t> half(chunk_size);
        for (const auto& [name, tensor] : tensors) {
            const size_t count = ggml_nelements(tensor);
            for (size_t begin = 0; begin < count; begin += chunk_size) {
                const size_t length = std::min(chunk_size, count - begin);
                for (size_t i = 0; i < length; ++i)
                    values[i] = value(name, begin + i);
                if (tensor->type == GGML_TYPE_F16) {
                    for (size_t i = 0; i < length; ++i)
                        half[i] = ggml_fp32_to_fp16(values[i]);
                    ggml_backend_tensor_set(tensor, half.data(), begin * sizeof(half[0]), length * sizeof(half[0]));
                } else {
                    require(tensor->type == GGML_TYPE_F32, "unexpected parameter type");
                    ggml_backend_tensor_set(tensor, values.data(), begin * sizeof(float), length * sizeof(float));
                }
            }
        }
    }

    void dense() {
        fill([](const std::string& name, size_t index) {
            const size_t seed = name.size() * 13 + static_cast<unsigned char>(name[0]);
            return (static_cast<float>((index * 17 + seed) % 251) - 125.f) / (name.find("bias") != std::string::npos ? 4096.f : 8192.f);
        });
    }
};

static std::vector<float> input_values(int64_t tokens, int64_t batch, int seed) {
    std::vector<float> result(2048 * tokens * batch);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t h = 0; h < 2048; ++h) {
                result[h + 2048 * (t + tokens * b)] = (static_cast<float>((h * 5 + t * 7 + b * 29 + seed) % 127) - 63.f) / 64.f;
            }
        }
    }
    return result;
}

static ggml_tensor* position_input(Graph& graph, int64_t tokens, std::vector<float>& values) {
    std::vector<std::vector<float>> ids;
    for (int64_t i = 0; i < tokens; ++i)
        ids.push_back({0.2f * i, 0.1f * i * i, -0.3f * i});
    values = Rope::embed_nd(ids, 1, 100.f, {16, 16, 16});
    return graph.input(2, 2, 24, tokens);
}

static void run_case(ggml_backend_t backend, ggml_backend_t cpu, MiniMaxH3VAE::DecoderAttention& attention, MiniMaxH3VAE::DecoderFeedForward& feedforward, int64_t tokens, int64_t batch, bool flash, bool oracle) {
    Graph graph(backend);
    auto input = graph.input(2048, tokens, batch);
    std::vector<float> positions;
    auto pe = position_input(graph, tokens, positions);
    std::vector<ggml_tensor*> outputs;
    std::vector<Structure> structures;
    const bool cuda = sd_backend_is(backend, "CUDA");
    for (bool automatic : {false, true}) {
        GGMLRunnerContext context;
        context.ggml_ctx           = graph.context.get();
        context.backend            = automatic ? backend : cpu;
        context.flash_attn_enabled = flash;
        outputs.push_back(attention.forward(&context, input, pe));
        outputs.push_back(feedforward.forward(&context, input));
        structures.push_back(structure(outputs[outputs.size() - 2]));
        structures.push_back(structure(outputs.back()));
        graph.output(outputs[outputs.size() - 2]);
        graph.output(outputs.back());
    }
    std::printf("tokens=%lld batch=%lld flash=%d structure: attention CONT %d -> %d, feedforward CONT %d -> %d, SwiGLU %d\n",
                static_cast<long long>(tokens), static_cast<long long>(batch), flash,
                structures[0].copies, structures[2].copies, structures[1].copies, structures[3].copies, structures[3].swiglu);
    std::fflush(stdout);
    require(structures[0].flash == (flash ? 1 : 0) && structures[2].flash == structures[0].flash, "attention implementation differs between policies");
    const std::vector<size_t> legacy_qkv_copies = {0, 64 * sizeof(float), 128 * sizeof(float)};
    require(qkv_materializations(structures[0], tokens, batch) == legacy_qkv_copies, "legacy attention must materialize all three head-interleaved QKV chunks");
    require(qkv_materializations(structures[2], tokens, batch) == (cuda ? std::vector<size_t>{} : legacy_qkv_copies), "automatic attention must eliminate all three direct QKV materializations only on CUDA");
    // Singleton legacy V is already contiguous after permutation; the CUDA view still needs packing.
    const int removed_attention_copies = cuda ? (tokens == 1 ? 2 : 3) : 0;
    require(structures[0].copies - structures[2].copies == removed_attention_copies, "unexpected total attention materialization count for sequence shape");
    require(structures[1].copies - structures[3].copies == (cuda ? 2 : 0), "feedforward must eliminate precisely two gate materializations only on CUDA");
    require(structures[1].swiglu == 0 && structures[1].silu == 1, "legacy feedforward graph changed");
    require(structures[3].swiglu == (cuda ? 1 : 0) && structures[3].silu == (cuda ? 0 : 1), "automatic feedforward must select the real packed SwiGLU node only on CUDA");
    require(structures[1].multiply == 1 && structures[3].multiply == (cuda ? 0 : 1), "automatic feedforward must eliminate the explicit gate multiplication only on CUDA");
    if (cuda)
        check_qkv_layout(structures[2], tokens, batch);
    std::vector<ggml_tensor*> gates;
    std::vector<ggml_tensor*> flash_inputs;
    if (oracle) {
        // Isolate gate semantics from w2's backend-dependent activation conversion.
        for (int policy : {1, 3}) {
            gates.push_back(gate_activation(structures[policy]));
            graph.output(gates.back());
        }
        if (cuda && flash) {
            for (int policy : {0, 2}) {
                for (auto node : structures[policy].nodes) {
                    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                        for (int input_index : {0, 1, 2}) {
                            flash_inputs.push_back(node->src[input_index]);
                            graph.output(flash_inputs.back());
                        }
                    }
                }
            }
            require(flash_inputs.size() == 6, "missing actual flash attention inputs");
        }
    }
    graph.allocate(backend);
    std::vector<std::vector<float>> first;
    for (int seed : {11, 23, 11}) {
        auto values = input_values(tokens, batch, seed);
        upload(input, values);
        upload(pe, positions);
        graph.compute(backend);
        std::vector<std::vector<float>> actual;
        for (auto output : outputs)
            actual.push_back(download(output));
        const std::string label = "tokens=" + std::to_string(tokens) + " batch=" + std::to_string(batch) +
                                  " flash=" + std::to_string(flash) + " seed=" + std::to_string(seed);
        compare(actual[2], actual[0], label + " actual attention copy/views");
        compare(actual[3], actual[1], label + " actual feedforward split/fused");
        if (oracle) {
            std::vector<float> expected_attention(values.size()), expected_gate(8192 * tokens * batch, 0.f);
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t h = 0; h < 2048; ++h) {
                    float mean = 0.f;
                    for (int64_t t = 0; t < tokens; ++t)
                        mean += values[h + 2048 * (t + tokens * b)] / tokens;
                    for (int64_t t = 0; t < tokens; ++t)
                        expected_attention[h + 2048 * (t + tokens * b)] = mean;
                }
            }
            for (size_t i = 0; i < values.size(); ++i) {
                const float x                               = values[i];
                expected_gate[i % 2048 + 8192 * (i / 2048)] = (x / (1.f + std::exp(-x))) * (0.5f * x + 0.25f);
            }
            for (int policy : {0, 2}) {
                if (cuda && flash) {
                    // Check head mapping independently of CUDA FA's accumulation precision.
                    const size_t base = policy / 2 * 3;
                    const auto v      = flash_inputs[base + 2];
                    require(v->type == GGML_TYPE_F16 && v->ne[0] == 64 && v->ne[1] == tokens && v->ne[2] == 32 * batch && v->ne[3] == 1,
                            "unexpected packed flash V type or shape");
                    std::vector<float> expected_v(values.size());
                    for (int64_t b = 0; b < batch; ++b)
                        for (int64_t head = 0; head < 32; ++head)
                            for (int64_t t = 0; t < tokens; ++t)
                                for (int64_t lane = 0; lane < 64; ++lane)
                                    expected_v[lane + 64 * (t + tokens * (head + 32 * b))] = values[head * 64 + lane + 2048 * (t + tokens * b)];
                    compare(download_float(v), expected_v, label + " independent packed F16 V head mapping", 0.f);
                    require(flash_inputs[base]->type == GGML_TYPE_F32 && flash_inputs[base + 1]->type == GGML_TYPE_F16, "unexpected flash Q/K input types");
                    for (size_t index : {base, base + 1})
                        compare(download_float(flash_inputs[index]), std::vector<float>(ggml_nelements(flash_inputs[index]), 0.f), label + " independent flash Q/K zeros", 0.f);
                    float max_mean_error = 0.f;
                    for (size_t i = 0; i < expected_attention.size(); ++i)
                        max_mean_error = std::max(max_mean_error, std::fabs(actual[policy][i] - expected_attention[i]));
                    std::printf("%s CUDA FA host-mean diagnostic policy=%d max_absolute=%.9g\n", label.c_str(), policy, max_mean_error);
                } else {
                    compare(actual[policy], expected_attention, label + " independent head-interleaved V mean oracle");
                }
                compare(download(gates[policy / 2]), expected_gate, label + " independent pre-w2 gate-first SwiGLU oracle");
            }
        }
        if (first.empty())
            first = actual;
        else if (seed == 11) {
            for (size_t i = 0; i < actual.size(); ++i)
                compare(actual[i], first[i], label + " repeated A/B/A", 0.f);
        } else {
            require(actual[0] != first[0] && actual[1] != first[1], "changed input did not change production block outputs");
        }
    }
    std::printf("Actual VAE policy structure: attention CONT %d -> %d, feedforward CONT %d -> %d, SwiGLU %d\n",
                structures[0].copies, structures[2].copies, structures[1].copies, structures[3].copies, structures[3].swiglu);
}

static void initialize_oracles(Parameters& attention, Parameters& feedforward) {
    attention.fill([](const std::string& name, size_t index) {
        if (name == "to_out.weight")
            return index % 2048 == index / 2048 ? 1.f : 0.f;
        if (name != "to_qkv.weight")
            return 0.f;
        const size_t row = index / 2048, input_channel = index % 2048;
        const size_t head = row / 192, lane = row % 192;
        return lane >= 128 && input_channel == head * 64 + lane - 128 ? 1.f : 0.f;
    });
    feedforward.fill([](const std::string& name, size_t index) {
        if (name == "w2.weight")
            return index % 8192 == index / 8192 ? 1.f : 0.f;
        if (name == "w1.bias")
            return index >= 8192 && index < 8192 + 2048 ? 0.25f : 0.f;
        if (name != "w1.weight")
            return 0.f;
        const size_t row = index / 2048, input_channel = index % 2048;
        if (row < 2048 && row == input_channel)
            return 1.f;
        return row >= 8192 && row < 8192 + 2048 && row - 8192 == input_channel ? 0.5f : 0.f;
    });
}

int main(int argc, char** argv) {
    try {
        const std::string name = argc > 1 ? argv[1] : "CPU";
        require(argc <= 2 && (name == "CPU" || name == "CUDA0"), "usage: test-h3-vae-decoder [CPU|CUDA0]");
        if (name == "CUDA0") {
            const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
            require(visible != nullptr && std::string(visible) == "0", "CUDA0 requires CUDA_VISIBLE_DEVICES=0");
        }
        // Both construction policies must use the same attention layout.
#ifdef _WIN32
        require(_putenv_s("GGML_ROPE_FLUX_DISABLE", "1") == 0, "failed to set attention layout");
#else
        require(setenv("GGML_ROPE_FLUX_DISABLE", "1", 1) == 0, "failed to set attention layout");
#endif
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_init_by_name(name.c_str(), nullptr), ggml_backend_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_init_by_name("CPU", nullptr), ggml_backend_free);
        require(backend != nullptr && cpu != nullptr, "requested backend is unavailable");
        sd_backend_cpu_set_n_threads(backend.get(), 2);
        for (bool half : {false, true}) {
            MiniMaxH3VAE::DecoderAttention attention;
            MiniMaxH3VAE::DecoderFeedForward feedforward;
            Parameters attention_parameters, feedforward_parameters;
            attention_parameters.initialize(attention, backend.get(), half, {"to_qkv.weight", "to_out.weight"});
            feedforward_parameters.initialize(feedforward, backend.get(), half, {"w1.weight", "w2.weight"});
            attention_parameters.dense();
            feedforward_parameters.dense();
            std::printf("VAE decoder weight type: %s\n", half ? "F16" : "F32");
            for (bool flash : {false, true}) {
                run_case(backend.get(), cpu.get(), attention, feedforward, 1, 1, flash, false);
                run_case(backend.get(), cpu.get(), attention, feedforward, 5, 2, flash, false);
            }
            initialize_oracles(attention_parameters, feedforward_parameters);
            for (bool flash : {false, true})
                run_case(backend.get(), cpu.get(), attention, feedforward, 4, 2, flash, true);
        }
        std::printf("Actual H3 VAE decoder blocks and independent mapping oracles passed on %s\n", name.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
