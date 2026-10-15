#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "model/diffusion/qwen_image_2_1.h"
#include "model/vae/wan_vae.hpp"
#include "runtime/denoiser.hpp"

static void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_attention_head_size() {
    const std::string prefix = "model.diffusion_model";
    String2TensorStorage weights;
    require(Qwen::QwenImage21Config::detect_from_weights(weights, prefix).head_dim == 128,
            "default attention head size");
    auto& norm_q  = weights[prefix + ".transformer_blocks.0.attn.norm_q.weight"];
    norm_q.n_dims = 1;
    norm_q.ne[0]  = 128;
    require(Qwen::QwenImage21Config::detect_from_weights(weights, prefix).head_dim == 128,
            "checkpoint attention head size");
    for (int64_t head_dim : {-128, 0, 64, 256}) {
        norm_q.ne[0]  = head_dim;
        bool rejected = false;
        try {
            Qwen::QwenImage21Config::detect_from_weights(weights, prefix);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()) == "Qwen Image 2.1: unsupported attention head size";
        }
        require(rejected, "invalid attention head size must fail before model construction");
    }
}

// Inspect dependencies, not just the output shape: the erroneous temporal
// convolution also produces one frame, but corrupts the decoded image.
static bool depends_on(ggml_tensor* tensor, ggml_tensor* parameter, std::set<ggml_tensor*>& visited) {
    if (tensor == parameter)
        return true;
    if (tensor == nullptr || !visited.insert(tensor).second)
        return false;
    for (auto source : tensor->src) {
        if (depends_on(source, parameter, visited))
            return true;
    }
    return false;
}

static void test_first_frame_upsampling() {
    ggml_context* ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "graph context");
    GGMLRunnerContext runner;
    runner.ggml_ctx = ctx;
    // Exercise the flag installed on Qwen's actual decoder blocks, and retain
    // the existing Wan video behavior with a separate default resampler.
    WAN::WanVAE vae(true, VERSION_QWEN_IMAGE_2_1);
    std::vector<GGMLBlock*> blocks;
    vae.get_all_blocks(blocks);
    int checked = 0;
    for (auto block : blocks) {
        auto resample = dynamic_cast<WAN::Resample*>(block);
        if (!resample)
            continue;
        resample->init(ctx);
        std::map<std::string, ggml_tensor*> params;
        resample->get_param_tensors(params);
        auto time_weight = params.find("time_conv.weight");
        if (time_weight == params.end())
            continue;
        auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 1, params.at("resample.1.bias")->ne[0]);
        std::vector<ggml_tensor*> cache(1, nullptr);
        int index   = 0;
        auto output = resample->forward(&runner, x, 1, cache, index, 0);
        std::set<ggml_tensor*> visited;
        require(!depends_on(output, time_weight->second, visited), "Qwen image must skip temporal convolution");
        require(output->ne[2] == 1 && index == 1, "Qwen first-frame shape and cache index");
        ++checked;
    }
    require(checked == 3, "all three Qwen temporal upsamplers checked");
    WAN::Resample wan(4, "upsample3d", true);
    wan.init(ctx);
    std::map<std::string, ggml_tensor*> params;
    wan.get_param_tensors(params);
    auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 1, 4);
    std::vector<ggml_tensor*> cache(1, nullptr);
    int index   = 0;
    auto output = wan.forward(&runner, x, 1, cache, index, 0);
    std::set<ggml_tensor*> visited;
    require(depends_on(output, params.at("time_conv.weight"), visited), "Wan video temporal convolution preserved");
    require(output->ne[2] == 1 && index == 1, "Wan first-frame shape preserved");
    ggml_free(ctx);
}

int main() {
    test_attention_head_size();
    test_first_frame_upsampling();
    auto text = Qwen::QwenImage21Layout::build(3, {}, {{4, 4}});
    require(text.prefix_length == 3 && text.positions.size() == 19, "text and target token layout");
    auto slots = sd::Tensor<int32_t>::from_vector({0, 1, 0, 2, 2, 0});
    auto edit  = Qwen::QwenImage21Layout::build(6, slots, {{2, 2}, {2, 4}, {4, 4}});
    require(edit.prefix_length == 15 && edit.positions.size() == 31, "ordered reference layout");
    require(edit.segments[1].image_index == 0 && edit.segments[3].image_index == 1, "reference order");
    bool rejected = false;
    try {
        Qwen::QwenImage21Layout::build(6, slots, {{2, 4}, {2, 4}, {4, 4}});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "mismatched vision and VAE tokens must fail");
    FluxScheduler qwen(1024, VERSION_QWEN_IMAGE_2_1);
    auto sigmas = qwen.get_sigmas(30, 0, 1, nullptr);
    require(sigmas.size() == 31 && sigmas.front() == 1.f && sigmas.back() == 0.f, "sigma endpoints");
    require(std::abs(sigmas[29] - 0.02f) < 1e-6f, "official terminal shift");
    for (size_t i = 1; i < sigmas.size(); ++i)
        require(sigmas[i] < sigmas[i - 1], "monotone sigma schedule");
    FluxScheduler flux(1024);
    require(flux.shift_terminal == 0.f && flux.max_image_seq_len == 4096, "FLUX defaults preserved");
    require(!sd_ctx_uses_reference_images(nullptr) && sd_ctx_get_image_channels(nullptr) == 0, "null capability query");
    std::puts("Qwen Image 2.1 configuration, layout, VAE and schedule tests passed");
}
