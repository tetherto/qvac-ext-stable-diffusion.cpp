#include <ggml-backend.h>
#include <ggml.h>
#include <stable-diffusion.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model/diffusion/minimax_h3.hpp"

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct Storage {
    ggml_context* ctx            = ggml_init({8 * 1024 * 1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~Storage() {
        if (buffer)
            ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};
static void run(ggml_backend_t backend, int width, int heads, int tokens, int batch) {
    Storage storage;
    auto* ctx       = storage.ctx;
    auto* projected = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2 * width, tokens, batch);
    auto* left      = ggml_cont(ctx, ggml_view_4d(ctx, projected, width, tokens, batch, 1,
                                                  projected->nb[1], projected->nb[2], projected->nb[3], 0));
    auto* right     = ggml_cont(ctx, ggml_view_4d(ctx, projected, width, tokens, batch, 1,
                                                  projected->nb[1], projected->nb[2], projected->nb[3], width * sizeof(float)));
    auto* reference = ggml_mul(ctx, ggml_silu(ctx, left), right);
    auto* fused     = ggml_swiglu(ctx, projected);
    auto* qkv       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * 128 * heads, tokens, batch);
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pairs{{reference, fused}};
    for (int index = 0; index < 3; ++index) {
        auto* old_slice  = ggml_cont(ctx, ggml_view_4d(ctx, qkv, 128 * heads, tokens, batch, 1,
                                                       qkv->nb[1], qkv->nb[2], qkv->nb[3], index * 128 * heads * sizeof(float)));
        auto* old_layout = ggml_reshape_4d(ctx, old_slice, 128, heads, tokens, batch);
        auto* view       = ggml_view_4d(ctx, qkv, 128, heads, tokens, batch,
                                        128 * qkv->nb[0], qkv->nb[1], qkv->nb[2], index * 128 * heads * sizeof(float));
        pairs.emplace_back(old_layout, ggml_cont(ctx, view));
        pairs.emplace_back(ggml_rms_norm(ctx, old_layout, 1e-6f), ggml_rms_norm(ctx, view, 1e-6f));
    }
    auto* graph = ggml_new_graph_custom(ctx, 128, false);
    for (const auto& pair : pairs) {
        ggml_build_forward_expand(graph, pair.first);
        ggml_build_forward_expand(graph, pair.second);
    }
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index)
        require(ggml_backend_supports_op(backend, ggml_graph_node(graph, index)), "GPU operation unsupported");
    storage.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(storage.buffer, "GPU allocation failed");
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (auto* tensor : {projected, qkv}) {
            std::vector<float> values(ggml_nelements(tensor));
            for (size_t index = 0; index < values.size(); ++index)
                values[index] = float(int((index * 17 + pattern * 11) % 129) - 64) / (pattern ? 7.0f : 3.0f);
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "GPU graph failed");
        for (size_t index = 0; index < pairs.size(); ++index) {
            const auto& pair = pairs[index];
            std::vector<float> a(ggml_nelements(pair.first)), b(a.size());
            ggml_backend_tensor_get(pair.first, a.data(), 0, ggml_nbytes(pair.first));
            ggml_backend_tensor_get(pair.second, b.data(), 0, ggml_nbytes(pair.second));
            if (index % 2 == 1)
                require(a == b, "QKV view is not exact");
            for (size_t element = 0; element < a.size(); ++element)
                require(std::isfinite(a[element]) && std::isfinite(b[element]) &&
                            std::abs(a[element] - b[element]) <= 2e-5f + 2e-6f * std::abs(a[element]),
                        "SwiGLU or strided normalization differs from reference");
        }
    }
}
static void run_blocks(ggml_backend_t backend, int heads, int tokens, int batch, bool flash) {
    Storage storage;
    GGMLRunnerContext runner;
    runner.ggml_ctx           = storage.ctx;
    runner.backend            = backend;
    runner.flash_attn_enabled = flash;
    MiniMaxH3::MLP mlp(128, 256);
    MiniMaxH3::Attention attention(128, heads, 128, 1e-5f);
    require(mlp.local_ops_enabled && attention.local_ops_enabled, "Preserve enabled H3 defaults");
    mlp.local_ops_enabled = attention.local_ops_enabled = false;
    mlp.init(storage.ctx);
    attention.init(storage.ctx);
    std::map<std::string, ggml_tensor*> params;
    mlp.get_param_tensors(params, "mlp");
    attention.get_param_tensors(params, "attention");
    auto* input = ggml_new_tensor_3d(storage.ctx, GGML_TYPE_F32, 128, tokens, batch);
    auto* graph = ggml_new_graph_custom(storage.ctx, 512, false);
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pairs;
    auto* reference_mlp       = mlp.forward(&runner, input);
    auto* reference_attention = attention.forward(&runner, input);
    mlp.local_ops_enabled = attention.local_ops_enabled = true;
    pairs.emplace_back(reference_mlp, mlp.forward(&runner, input));
    pairs.emplace_back(reference_attention, attention.forward(&runner, input));
    mlp.local_ops_enabled = attention.local_ops_enabled = false;
    pairs.emplace_back(reference_mlp, mlp.forward(&runner, input));
    pairs.emplace_back(reference_attention, attention.forward(&runner, input));
    for (const auto& pair : pairs) {
        ggml_build_forward_expand(graph, pair.first);
        ggml_build_forward_expand(graph, pair.second);
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), "H3 block operation unsupported");
    storage.buffer = ggml_backend_alloc_ctx_tensors(storage.ctx, backend);
    require(storage.buffer, "H3 block allocation failed");
    params["input"] = input;
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (const auto& param : params) {
            auto* tensor = param.second;
            require(tensor->type == GGML_TYPE_F32, "Expected F32 test parameters");
            std::vector<float> values(ggml_nelements(tensor));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = std::sin(float(i * 17 + pattern * 11 + param.first.size())) * 0.25f;
                if (param.first.find("_norm") != std::string::npos)
                    values[i] += 1.f;
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "H3 block graph failed");
        for (const auto& pair : pairs) {
            std::vector<float> a(ggml_nelements(pair.first)), b(a.size());
            ggml_backend_tensor_get(pair.first, a.data(), 0, ggml_nbytes(pair.first));
            ggml_backend_tensor_get(pair.second, b.data(), 0, ggml_nbytes(pair.second));
            for (size_t i = 0; i < a.size(); ++i)
                require(std::isfinite(a[i]) && std::isfinite(b[i]) &&
                            std::abs(a[i] - b[i]) <= 2e-5f + 2e-6f * std::abs(a[i]),
                        "Production H3 block output differs with local operations");
        }
    }
}

static void test_runner_toggle(ggml_backend_t backend) {
    String2TensorStorage weights;
    const std::string prefix = "model.diffusion_model";
    const auto shape         = [&](const char* name, int64_t in, int64_t out) {
        auto& weight  = weights[prefix + "." + name];
        weight.n_dims = 2;
        weight.ne[0]  = in;
        weight.ne[1]  = out;
        weight.type   = GGML_TYPE_F32;
    };
    shape("video_patch_proj.weight", 32, 128);
    shape("audio_patch_proj.weight", 8, 128);
    shape("condition_proj.weight", 128, 128);
    shape("blocks.0.attn.q_norm.weight", 128, 1);
    shape("blocks.0.attn.qkv_proj.weight", 128, 768);
    shape("blocks.0.mlp.fc1.weight", 128, 512);
    shape("token_refiner.blocks.0.attn.q_norm.weight", 128, 1);
    shape("time_embedder.proj_in.weight", 64, 64);
    shape("time_embedder.proj_out.weight", 64, 64);
    MiniMaxH3::MiniMaxH3Runner runner(backend, weights, prefix);
    std::vector<GGMLBlock*> blocks;
    runner.model.get_all_blocks(blocks);
    for (bool enabled : {false, true, false}) {
        runner.set_local_ops_enabled(enabled);
        int mlps = 0, attentions = 0;
        for (auto* block : blocks) {
            if (auto* mlp = dynamic_cast<MiniMaxH3::MLP*>(block)) {
                require(mlp->local_ops_enabled == enabled, "Runner must toggle every H3 MLP");
                ++mlps;
            }
            if (auto* attention = dynamic_cast<MiniMaxH3::Attention*>(block)) {
                require(attention->local_ops_enabled == enabled, "Runner must toggle every H3 attention");
                ++attentions;
            }
        }
        require(mlps == 2 && attentions == 2, "Check transformer and token-refiner blocks");
    }
}

int main(int argc, char** argv) {
    require(!sd_set_h3_local_ops(nullptr, true), "Null context must be rejected");
    require(!sd_set_vae_conv_direct(nullptr, false), "Null VAE context must be rejected");
    const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
    ggml_backend_load_all();
    auto* backend = ggml_backend_init_by_type(gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend && gpu) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!backend) {
        std::cerr << "Requested backend unavailable\n";
        return 1;
    }
    try {
        run(backend, 128, 1, 1, 1);
        run(backend, 14336, 5, 17, 2);
        run(backend, 14336, 48, 127, 1);
        test_runner_toggle(backend);
        for (bool flash : {false, true}) {
            run_blocks(backend, 1, 1, 1, flash);
            run_blocks(backend, 3, 17, 2, flash);
        }
        std::cout << "PASS H3 primitive and production-block equivalence, flash attention on/off, changing inputs, batches and runner toggles\n";
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
}
