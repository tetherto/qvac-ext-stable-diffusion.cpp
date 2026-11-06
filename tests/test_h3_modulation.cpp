#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/diffusion/minimax_h3.hpp"

using ContextHandle   = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using BufferHandle    = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using AllocatorHandle = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

static void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static ContextHandle make_context() {
    ggml_init_params params = {};
    params.mem_size         = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
    params.no_alloc         = true;
    ContextHandle context(ggml_init(params), ggml_free);
    require(context != nullptr, "failed to create context");
    return context;
}

static int count_operations(ggml_tensor* output, ggml_op operation) {
    auto context = make_context();
    auto graph   = ggml_new_graph_custom(context.get(), 4096, false);
    ggml_build_forward_expand(graph, output);
    int count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        count += ggml_graph_node(graph, i)->op == operation;
    }
    return count;
}

static std::vector<float> make_values(size_t count, int seed, float scale = 1.f) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = scale * (static_cast<float>((i * 17 + seed * 29) % 127) - 63.f) / 64.f;
    }
    return values;
}

static void upload(ggml_tensor* tensor, const std::vector<float>& values) {
    require(static_cast<size_t>(ggml_nelements(tensor)) == values.size(), "upload shape mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static std::vector<float> download(ggml_tensor* tensor) {
    require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "unexpected output layout");
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

static void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label, float tolerance = 2e-5f) {
    require(actual.size() == expected.size(), label + ": size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || difference > tolerance * (1.f + std::fabs(expected[i]))) {
            throw std::runtime_error(label + ": mismatch at " + std::to_string(i) +
                                     ", got " + std::to_string(actual[i]) +
                                     ", expected " + std::to_string(expected[i]));
        }
    }
}

static void report_difference(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label) {
    float max_absolute = 0.f;
    float max_scaled   = 0.f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expected[i]);
        max_absolute           = std::max(max_absolute, difference);
        max_scaled             = std::max(max_scaled, difference / (1.f + std::fabs(expected[i])));
    }
    std::printf("%s: max_absolute=%.9g max_scaled=%.9g\n", label.c_str(), max_absolute, max_scaled);
    std::fflush(stdout);
}

struct TestGraph {
    ContextHandle context = make_context();
    ggml_cgraph* graph    = ggml_new_graph_custom(context.get(), 4096, false);
    AllocatorHandle allocator;

    explicit TestGraph(ggml_backend_t backend)
        : allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {}

    ggml_tensor* input(int64_t n0, int64_t n1, int64_t n2 = 1) {
        auto tensor = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, n0, n1, n2);
        ggml_set_input(tensor);
        return tensor;
    }

    void output(ggml_tensor* tensor) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph, tensor);
    }

    void allocate(ggml_backend_t backend) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            require(ggml_backend_supports_op(backend, node),
                    std::string("unsupported operation: ") + ggml_op_name(node->op));
        }
        require(ggml_gallocr_alloc_graph(allocator.get(), graph), "graph allocation failed");
    }

    void compute(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
    }
};

static std::vector<float> modulation_oracle(const std::vector<float>& x,
                                            const std::vector<float>& update,
                                            const std::vector<float>& projection,
                                            const std::vector<MiniMaxH3::TokenModulationSpan>& segments,
                                            int first,
                                            int second,
                                            bool residual) {
    constexpr int64_t hidden = 32, tokens = 13, batch = 2;
    int64_t output_tokens = 0;
    for (const auto& segment : segments) {
        output_tokens += segment.end - segment.start;
    }
    std::vector<float> result(hidden * output_tokens * batch);
    for (int64_t b = 0; b < batch; ++b) {
        int64_t out_token = 0;
        for (const auto& segment : segments) {
            for (int64_t t = segment.start; t < segment.end; ++t, ++out_token) {
                for (int64_t h = 0; h < hidden; ++h) {
                    const size_t input_index  = h + hidden * (t + tokens * b);
                    const size_t output_index = h + hidden * (out_token + output_tokens * b);
                    const size_t row          = segment.modulation_row * 6 * hidden + h;
                    const float a             = projection[row + first * hidden];
                    const float value         = x[input_index];
                    if (residual) {
                        result[output_index] = value + update[input_index] * a;
                    } else {
                        const float scale    = projection[row + second * hidden];
                        result[output_index] = (value + value * scale) + a;
                    }
                }
            }
        }
    }
    return result;
}

static void test_helpers(ggml_backend_t backend,
                         const std::vector<MiniMaxH3::TokenModulationSpan>& segments) {
    constexpr int64_t hidden = 32;
    TestGraph graph(backend);
    auto ctx        = graph.context.get();
    auto x          = graph.input(hidden, 13, 2);
    auto update     = graph.input(hidden, 13, 2);
    auto projection = graph.input(hidden * 6 * 3, 2);
    std::vector<ggml_tensor*> outputs;
    for (bool views : {false, true}) {
        for (int offset : {0, 3}) {
            outputs.push_back(MiniMaxH3::modulate_segments(ctx, x, projection, segments,
                                                           hidden, 6, 3, offset, offset + 1, views));
        }
        for (int gate : {2, 5}) {
            outputs.push_back(MiniMaxH3::gated_residual_segments(ctx, x, update, projection,
                                                                 segments, hidden, gate, views));
        }
    }
    for (auto output : outputs) {
        graph.output(output);
    }
    graph.allocate(backend);
    std::vector<std::vector<float>> first_outputs;
    for (int seed : {1, 2, 1}) {
        const auto values      = make_values(ggml_nelements(x), seed);
        const auto updates     = make_values(ggml_nelements(update), seed + 4);
        const auto projections = make_values(ggml_nelements(projection), seed + 9, 0.25f);
        upload(x, values);
        upload(update, updates);
        upload(projection, projections);
        graph.compute(backend);
        for (size_t i = 0; i < outputs.size(); ++i) {
            const int operation = i % 4;
            const bool residual = operation >= 2;
            const int first     = residual ? (operation == 2 ? 2 : 5) : operation * 3;
            const auto expected = modulation_oracle(values, updates, projections, segments,
                                                    first, first + 1, residual);
            const auto actual   = download(outputs[i]);
            compare(actual, expected, "helper scalar oracle");
            if (first_outputs.size() < outputs.size()) {
                first_outputs.push_back(actual);
            } else if (seed == 1) {
                compare(actual, first_outputs[i], "helper repeated input", 0.f);
            }
        }
    }
}

struct Parameters {
    ContextHandle context = make_context();
    BufferHandle buffer{nullptr, ggml_backend_buffer_free};
    std::map<std::string, ggml_tensor*> tensors;
    std::map<std::string, std::vector<float>> values;

    void initialize(GGMLBlock& block, ggml_backend_t backend, bool half_adaln) {
        String2TensorStorage storage;
        if (half_adaln) {
            storage["adaln_proj.linear.weight"].type = GGML_TYPE_F16;
        }
        block.init(context.get(), storage);
        block.get_param_tensors(tensors);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(buffer != nullptr, "parameter allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        int seed = 10;
        for (const auto& [name, tensor] : tensors) {
            auto data = make_values(ggml_nelements(tensor), seed++, 0.05f);
            if (name.find("norm") != std::string::npos && name.find("weight") != std::string::npos) {
                for (auto& value : data) {
                    value += 1.f;
                }
            }
            if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(data.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    half[i] = ggml_fp32_to_fp16(data[i]);
                    data[i] = ggml_fp16_to_fp32(half[i]);
                }
                ggml_backend_tensor_set(tensor, half.data(), 0, half.size() * sizeof(half[0]));
            } else {
                require(tensor->type == GGML_TYPE_F32, "unexpected parameter type");
                upload(tensor, data);
            }
            values[name] = std::move(data);
        }
    }
};

static MiniMaxH3::Config small_config() {
    MiniMaxH3::Config config;
    config.hidden_size           = 128;
    config.num_attention_heads   = 2;
    config.attention_head_dim    = 64;
    config.ffn_hidden_size       = 256;
    config.time_embed_dim        = 16;
    config.video_latent_channels = 8;
    config.audio_latent_channels = 5;
    config.patch_t = config.patch_h = config.patch_w = 1;
    return config;
}

static std::vector<float> final_layer_oracle(const MiniMaxH3::Config& config,
                                             const Parameters& parameters,
                                             const std::vector<float>& x,
                                             const std::vector<float>& time,
                                             const MiniMaxH3::TokenModulationSpan& span,
                                             const std::string& output_prefix,
                                             int64_t output_size) {
    const int64_t hidden = config.hidden_size, length = span.end - span.start;
    const auto& weights = parameters.values.at("adaln_proj.linear.weight");
    const auto& bias    = parameters.values.at("adaln_proj.linear.bias");
    std::vector<float> mods(hidden * 2);
    for (int64_t i = 0; i < hidden * 2; ++i) {
        float value = bias[i];
        for (int64_t j = 0; j < config.time_embed_dim; ++j) {
            const float t = time[j + span.modulation_row * config.time_embed_dim];
            value += t / (1.f + std::exp(-t)) * weights[j + i * config.time_embed_dim];
        }
        mods[i] = value;
    }
    const auto& norm       = parameters.values.at("norm.weight");
    const auto& out_weight = parameters.values.at(output_prefix + ".weight");
    const auto& out_bias   = parameters.values.at(output_prefix + ".bias");
    std::vector<float> result(output_size * length * 2), modulated(hidden);
    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t t = 0; t < length; ++t) {
            double squares = 0.;
            for (int64_t h = 0; h < hidden; ++h) {
                const float value = x[h + hidden * (span.start + t + 13 * b)];
                squares += value * value;
            }
            const float inv_rms = 1.f / std::sqrt(static_cast<float>(squares / hidden) + config.final_norm_eps);
            for (int64_t h = 0; h < hidden; ++h) {
                const float value = x[h + hidden * (span.start + t + 13 * b)] * inv_rms * norm[h];
                modulated[h]      = (value + value * mods[hidden + h]) + mods[h];
            }
            for (int64_t o = 0; o < output_size; ++o) {
                float value = out_bias[o];
                for (int64_t h = 0; h < hidden; ++h) {
                    value += out_weight[h + o * hidden] * modulated[h];
                }
                result[o + output_size * (t + length * b)] = value;
            }
        }
    }
    return result;
}

static void test_blocks(ggml_backend_t backend, ggml_backend_t reference_backend, bool half_adaln, bool flash, bool full_scalar_oracle) {
    const auto config = small_config();
    MiniMaxH3::FinalLayer final_layer(config);
    MiniMaxH3::TransformerBlock transformer(config);
    Parameters final_parameters, transformer_parameters;
    final_parameters.initialize(final_layer, backend, half_adaln);
    transformer_parameters.initialize(transformer, backend, half_adaln);
    TestGraph graph(backend);
    auto x                                                  = graph.input(config.hidden_size, 13, 2);
    auto time                                               = graph.input(config.time_embed_dim, 2);
    const std::vector<MiniMaxH3::TokenModulationSpan> spans = {
        {0, 1, 5}, {1, 4, 0}, {4, 6, 3}, {6, 7, 1}, {7, 11, 4}, {11, 13, 2}};
    const MiniMaxH3::TokenModulationSpan video{2, 9, 1}, audio{10, 13, 0};
    std::vector<ggml_tensor*> outputs;
    std::vector<int> reference_cont_counts;
    const char* output_labels[] = {"FinalLayer video", "FinalLayer audio", "TransformerBlock"};
    int flash_nodes             = 0;
    for (bool automatic_policy : {false, true}) {
        GGMLRunnerContext context;
        context.ggml_ctx = graph.context.get();
        // The CPU construction backend retains the original copy policy.
        // Both graphs must select the same attention path and execute below
        // on the requested backend, including all CUDA support checks.
        context.backend            = automatic_policy ? backend : reference_backend;
        context.flash_attn_enabled = flash;
        const auto final           = final_layer.forward(&context, x, time, video, audio);
        outputs.push_back(final.first);
        outputs.push_back(final.second);
        outputs.push_back(transformer.forward(&context, x, time, spans, nullptr));
        for (size_t i = outputs.size() - 3; i < outputs.size(); ++i) {
            const int cont_count = count_operations(outputs[i], GGML_OP_CONT);
            if (!automatic_policy) {
                reference_cont_counts.push_back(cont_count);
            } else {
                const size_t index        = i - 3;
                const int reference_count = reference_cont_counts[index];
                const bool cuda           = sd_backend_is(backend, "CUDA");
                require(cuda ? cont_count < reference_count : cont_count == reference_count,
                        std::string(output_labels[index]) + ": incorrect backend CONT policy");
                std::printf("%s CONT nodes: copy=%d automatic=%d\n", output_labels[index], reference_count, cont_count);
            }
            graph.output(outputs[i]);
        }
        int total_flash_nodes = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
            total_flash_nodes += ggml_graph_node(graph.graph, i)->op == GGML_OP_FLASH_ATTN_EXT;
        }
        require(total_flash_nodes - flash_nodes == (flash ? 1 : 0), "attention path mismatch");
        flash_nodes = total_flash_nodes;
    }
    graph.allocate(backend);
    std::vector<std::vector<float>> first_outputs;
    for (int seed : {3, 7, 3}) {
        const auto values = make_values(ggml_nelements(x), seed);
        const auto times  = make_values(ggml_nelements(time), seed + 3);
        upload(x, values);
        upload(time, times);
        graph.compute(backend);
        std::vector<std::vector<float>> actual;
        for (auto output : outputs) {
            actual.push_back(download(output));
        }
        for (size_t i = 0; i < 3; ++i) {
            compare(actual[i + 3], actual[i], "production block paths");
        }
        const float tolerance     = half_adaln ? 2e-4f : 2e-5f;
        const auto expected_video = final_layer_oracle(config, final_parameters, values, times, video, "video_out", config.video_latent_channels);
        const auto expected_audio = final_layer_oracle(config, final_parameters, values, times, audio, "audio_out", config.audio_latent_channels);
        const std::string fixture = "half_adaln=" + std::to_string(half_adaln) +
                                    " flash=" + std::to_string(flash) + " seed=" + std::to_string(seed);
        report_difference(actual[3], actual[0], fixture + " video views/copy");
        report_difference(actual[4], actual[1], fixture + " audio views/copy");
        report_difference(actual[0], expected_video, fixture + " video copy/oracle");
        report_difference(actual[3], expected_video, fixture + " video views/oracle");
        report_difference(actual[1], expected_audio, fixture + " audio copy/oracle");
        report_difference(actual[4], expected_audio, fixture + " audio views/oracle");
        if (full_scalar_oracle) {
            compare(actual[0], expected_video, fixture + " copy final video scalar oracle", tolerance);
            compare(actual[1], expected_audio, fixture + " copy final audio scalar oracle", tolerance);
            compare(actual[3], expected_video, fixture + " views final video scalar oracle", tolerance);
            compare(actual[4], expected_audio, fixture + " views final audio scalar oracle", tolerance);
        }
        if (first_outputs.empty()) {
            first_outputs = actual;
        } else if (seed == 3) {
            for (size_t i = 0; i < actual.size(); ++i) {
                compare(actual[i], first_outputs[i], "production block repeated input", 0.f);
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        const std::string name                = argc > 1 ? argv[1] : "CPU";
        const bool request_full_scalar_oracle = argc == 3 && std::string(argv[2]) == "--full-scalar-oracle";
        require(argc <= 3 && (argc != 3 || request_full_scalar_oracle) && (name == "CPU" || name == "CUDA0"),
                "usage: test-h3-modulation [CPU|CUDA0] [--full-scalar-oracle]");
        const bool full_scalar_oracle = name == "CPU" || request_full_scalar_oracle;
        if (name == "CUDA0") {
            const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
            require(visible != nullptr && std::string(visible) == "0", "CUDA0 requires CUDA_VISIBLE_DEVICES=0");
            if (full_scalar_oracle) {
                const char* tf32_override = std::getenv("NVIDIA_TF32_OVERRIDE");
                require(tf32_override != nullptr && std::string(tf32_override) == "0",
                        "CUDA full scalar oracle requires NVIDIA_TF32_OVERRIDE=0 before process startup");
            }
            // CUDA has no ROPE_FLUX implementation. Keep the CPU reference
            // construction backend on CUDA's attention layout path as well.
#ifdef _WIN32
            require(_putenv_s("GGML_ROPE_FLUX_DISABLE", "1") == 0, "failed to select attention layout");
#else
            require(setenv("GGML_ROPE_FLUX_DISABLE", "1", 1) == 0, "failed to select attention layout");
#endif
        }
        // Default CUDA cuBLAS enables TF32. Compare its real production paths
        // directly, and assert the full FP32 scalar oracle in a separate run.
        // Elementwise helper scalar assertions remain enabled in every run.
        std::printf("FinalLayer FP32 scalar assertions: %s\n", full_scalar_oracle ? "enabled" : "diagnostics only (default CUDA precision)");
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(name.c_str(), nullptr), ggml_backend_free);
        require(backend != nullptr, "backend initialization failed");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> reference_backend(
            ggml_backend_init_by_name("CPU", nullptr), ggml_backend_free);
        require(reference_backend != nullptr, "reference backend initialization failed");
        sd_backend_cpu_set_n_threads(backend.get(), 2);
        test_helpers(backend.get(), {{0, 1, 5}, {1, 4, 0}, {4, 6, 3}, {6, 7, 1}, {7, 11, 4}, {11, 13, 2}});
        test_helpers(backend.get(), {{0, 13, 4}});
        test_helpers(backend.get(), {{1, 4, 2}, {6, 12, 5}});
        for (bool half_adaln : {false, true}) {
            for (bool flash : {false, true}) {
                test_blocks(backend.get(), reference_backend.get(), half_adaln, flash, full_scalar_oracle);
            }
        }
        std::printf("H3 modulation helpers and production blocks passed on %s\n", name.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
