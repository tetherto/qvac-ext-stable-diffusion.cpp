#include <ggml-backend.h>
#include <ggml.h>
#include <stable-diffusion.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "model/diffusion/minimax_h3.hpp"
#include "model/vae/audio_vae.hpp"
#include "model/vae/vae.hpp"
#include "conditioning/conditioner.hpp"
#include "pipeline/diffusion_engine.h"

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

static String2TensorStorage small_h3_weights() {
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
    return weights;
}

static void test_runner_toggle(ggml_backend_t backend) {
    MiniMaxH3::MiniMaxH3Runner runner(backend, small_h3_weights(), "model.diffusion_model");
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


struct EmptyConditioner : public Conditioner {
    SDCondition get_learned_condition(int, const ConditionerParams&) override { return {}; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>&) override {}
    void set_fit_module(SDBackendModule) override {}
    void set_flash_attention_enabled(bool) override {}
};

struct EmptyAudioVAE : public AudioVAERunner {
    explicit EmptyAudioVAE(ggml_backend_t backend) : AudioVAERunner(backend) {}
    void get_param_tensors(std::map<std::string, ggml_tensor*>&) override {}
    size_t get_params_mem_size() override { return 0; }
    std::string get_desc() override { return "empty_audio_vae"; }
    sd::Tensor<float> decode(int, const sd::Tensor<float>& input) override { return input; }
    int output_sample_rate() const override { return 48000; }
};

static void check_h3_controls(StableDiffusionGGML& sd, bool enabled) {
    auto h3 = std::dynamic_pointer_cast<MiniMaxH3::MiniMaxH3Runner>(sd.diffusion_model);
    require(h3 != nullptr, "Expected H3 runner");
    std::vector<GGMLBlock*> blocks;
    h3->model.get_all_blocks(blocks);
    for (auto* block : blocks) {
        if (auto* mlp = dynamic_cast<MiniMaxH3::MLP*>(block))
            require(mlp->local_ops_enabled == enabled, "H3 MLP context setting lost");
        if (auto* attention = dynamic_cast<MiniMaxH3::Attention*>(block))
            require(attention->local_ops_enabled == enabled, "H3 attention context setting lost");
    }
}

static void check_vae_controls(StableDiffusionGGML& sd, bool enabled) {
    for (GGMLRunner* runner : {static_cast<GGMLRunner*>(sd.first_stage_model.get()),
                               static_cast<GGMLRunner*>(sd.preview_vae.get()),
                               static_cast<GGMLRunner*>(sd.audio_vae_model.get())}) {
        require(runner->get_context().conv2d_direct_enabled == enabled, "VAE 2D context setting lost");
        require(runner->get_context().conv3d_direct_enabled, "VAE 3D configuration changed");
    }
}

static void test_context_controls(ggml_backend_t backend) {
    StableDiffusionGGML sd;
    require(!sd.set_h3_local_ops(false), "Missing H3 runner must be rejected");
    require(!sd.set_vae_conv_direct(true), "Missing VAEs must be rejected");
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    sd.config_ = std::make_unique<StableDiffusionGGML::ModelConfig>(params);
    sd.version = VERSION_MINIMAX_H3;
    sd.model_manager = std::make_shared<ModelManager>();
    sd.cond_stage_model = std::make_shared<EmptyConditioner>();
    sd.conv_direct_assignment.set_default("on");
    sd.flash_attn_assignment.set_default("off");
    sd.fit_dry_run = true;
    const auto replace_runners = [&] {
        sd.diffusion_model = std::make_shared<MiniMaxH3::MiniMaxH3Runner>(backend, small_h3_weights());
        sd.first_stage_model = std::make_shared<FakeVAE>(VERSION_FLUX2, backend);
        sd.preview_vae = std::make_shared<FakeVAE>(VERSION_FLUX2, backend);
        sd.audio_vae_model = std::make_shared<EmptyAudioVAE>(backend);
    };
    replace_runners();
    require(sd.validate_and_load_runners(), "Initial runner validation failed");
    check_h3_controls(sd, true);
    check_vae_controls(sd, true);

    StableDiffusionGGML other;
    other.diffusion_model = std::make_shared<MiniMaxH3::MiniMaxH3Runner>(backend, small_h3_weights());
    other.first_stage_model = std::make_shared<FakeVAE>(VERSION_FLUX2, backend);
    require(other.set_vae_conv_direct(true), "Other context VAE setup failed");

    for (bool enabled : {false, true, false}) {
        require(sd.set_h3_local_ops(enabled), "H3 context setter failed");
        require(sd.set_vae_conv_direct(enabled), "VAE context setter failed");
        check_h3_controls(sd, enabled);
        check_vae_controls(sd, enabled);
        require(sd.conv_direct_for(SDBackendModule::DIFFUSION, false), "VAE override affected diffusion");
        {
            StableDiffusionGGML::ContextOperation operation(sd);
            require(operation.acquired, "Could not acquire context guard");
            require(!sd.set_h3_local_ops(!enabled), "Reentrant H3 setter accepted busy context");
            require(!sd.set_vae_conv_direct(!enabled), "Reentrant VAE setter accepted busy context");
            bool h3_accepted = true, vae_accepted = true;
            std::thread concurrent([&] {
                h3_accepted = sd.set_h3_local_ops(!enabled);
                vae_accepted = sd.set_vae_conv_direct(!enabled);
            });
            concurrent.join();
            require(!h3_accepted && !vae_accepted, "Concurrent setters accepted busy context");
        }
        check_h3_controls(sd, enabled);
        check_vae_controls(sd, enabled);
        require(sd.build_runners({StableDiffusionGGML::RunnerGroup::ControlNet}), "ControlNet runner update failed");
        check_h3_controls(sd, enabled);
        check_vae_controls(sd, enabled);
        replace_runners();
        require(sd.validate_and_load_runners(), "Replacement runner validation failed");
        check_h3_controls(sd, enabled);
        check_vae_controls(sd, enabled);
        check_h3_controls(other, true);
        require(other.first_stage_model->get_context().conv2d_direct_enabled, "VAE setting leaked to other context");
    }
    sd.diffusion_model.reset();
    require(!sd.set_h3_local_ops(true), "Non-H3 context accepted H3 controls");
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
        test_context_controls(backend);
        for (bool flash : {false, true}) {
            run_blocks(backend, 1, 1, 1, flash);
            run_blocks(backend, 3, 17, 2, flash);
        }
        std::cout << "PASS H3 primitive and production-block equivalence, flash attention on/off, changing inputs, batches, runner rebuilds and guarded context controls\n";
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
}
