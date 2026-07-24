#pragma once

#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "ggml_extend.hpp"
#include "model.h"

class LamWav2VecFrontend final : public GGMLRunner {
public:
    static constexpr int kLayers = 7;
    static constexpr int kChannels = 512;

    LamWav2VecFrontend(ggml_backend_t backend,
                       ggml_backend_t params_backend,
                       const String2TensorStorage& tensor_storage_map)
        : GGMLRunner(backend, params_backend) {
        static constexpr std::array<int64_t, kLayers> kernels = {10, 3, 3, 3, 3, 2, 2};
        static constexpr std::array<int64_t, kLayers> input_channels = {1, 512, 512, 512, 512, 512, 512};

        for (int i = 0; i < kLayers; ++i) {
            const std::string name =
                "backbone.audio_encoder.feature_extractor.conv_layers." +
                std::to_string(i) + ".conv.weight";
            const auto found = tensor_storage_map.find(name);
            const ggml_type type =
                found == tensor_storage_map.end() ? GGML_TYPE_F32 : found->second.type;
            conv_weights_[i] = ggml_new_tensor_3d(
                params_ctx, type, kernels[i], input_channels[i], kChannels);
            ggml_set_name(conv_weights_[i], name.c_str());
        }

        const auto norm_weight_name =
            "backbone.audio_encoder.feature_extractor.conv_layers.0.layer_norm.weight";
        const auto norm_bias_name =
            "backbone.audio_encoder.feature_extractor.conv_layers.0.layer_norm.bias";
        norm_weight_ = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, kChannels);
        norm_bias_ = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, kChannels);
        ggml_set_name(norm_weight_, norm_weight_name);
        ggml_set_name(norm_bias_, norm_bias_name);

        feature_norm_weight_ = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, kChannels);
        feature_norm_bias_ = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, kChannels);
        feature_projection_weight_ =
            ggml_new_tensor_2d(params_ctx, GGML_TYPE_F32, kChannels, 768);
        feature_projection_bias_ = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, 768);
        ggml_set_name(feature_norm_weight_,
                      "backbone.audio_encoder.feature_projection.layer_norm.weight");
        ggml_set_name(feature_norm_bias_,
                      "backbone.audio_encoder.feature_projection.layer_norm.bias");
        ggml_set_name(feature_projection_weight_,
                      "backbone.audio_encoder.feature_projection.projection.weight");
        ggml_set_name(feature_projection_bias_,
                      "backbone.audio_encoder.feature_projection.projection.bias");

        pos_weight_g_ = ggml_new_tensor_3d(params_ctx, GGML_TYPE_F32, 128, 1, 1);
        pos_weight_v_ = ggml_new_tensor_3d(params_ctx, GGML_TYPE_F32, 128, 48, 768);
        pos_bias_ = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, 768);
        ggml_set_name(pos_weight_g_,
                      "backbone.audio_encoder.encoder.pos_conv_embed.conv.weight_g");
        ggml_set_name(pos_weight_v_,
                      "backbone.audio_encoder.encoder.pos_conv_embed.conv.weight_v");
        ggml_set_name(pos_bias_,
                      "backbone.audio_encoder.encoder.pos_conv_embed.conv.bias");
    }

    std::string get_desc() override {
        return "lam-a2e-wav2vec-frontend";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        for (int i = 0; i < kLayers; ++i) {
            tensors[ggml_get_name(conv_weights_[i])] = conv_weights_[i];
        }
        tensors[ggml_get_name(norm_weight_)] = norm_weight_;
        tensors[ggml_get_name(norm_bias_)] = norm_bias_;
        tensors[ggml_get_name(feature_norm_weight_)] = feature_norm_weight_;
        tensors[ggml_get_name(feature_norm_bias_)] = feature_norm_bias_;
        tensors[ggml_get_name(feature_projection_weight_)] = feature_projection_weight_;
        tensors[ggml_get_name(feature_projection_bias_)] = feature_projection_bias_;
        tensors[ggml_get_name(pos_weight_g_)] = pos_weight_g_;
        tensors[ggml_get_name(pos_weight_v_)] = pos_weight_v_;
        tensors[ggml_get_name(pos_bias_)] = pos_bias_;
    }

    bool load(ModelLoader& loader, int n_threads) {
        if (!alloc_params_buffer()) {
            return false;
        }
        std::map<std::string, ggml_tensor*> tensors;
        get_param_tensors(tensors);
        return loader.load_tensors(tensors, {}, n_threads, false);
    }

    ggml_tensor* forward_frontend(ggml_tensor* x) {
        static constexpr std::array<int, kLayers> strides = {5, 2, 2, 2, 2, 2, 2};
        for (int i = 0; i < kLayers; ++i) {
            x = ggml_conv_1d(compute_ctx, conv_weights_[i], x, strides[i], 0, 1);
            if (i == 0) {
                const int64_t frames = x->ne[0];
                const int64_t channels = x->ne[1];
                const int64_t batch = x->ne[2];
                // GGML GroupNorm treats ne[2] as channels. Convert from the
                // frontend's [time, channels, batch] layout to
                // [1, time, channels, batch] for PyTorch-equivalent
                // GroupNorm(channels, channels), then restore the layout.
                auto* grouped =
                    ggml_reshape_4d(compute_ctx, x, 1, frames, channels, batch);
                grouped = ggml_group_norm(compute_ctx, grouped, kChannels, 1e-5f);
                x = ggml_reshape_3d(compute_ctx, grouped, frames, channels, batch);
                auto* norm_weight = ggml_reshape_3d(
                    compute_ctx, norm_weight_, 1, kChannels, 1);
                auto* norm_bias = ggml_reshape_3d(
                    compute_ctx, norm_bias_, 1, kChannels, 1);
                x = ggml_add(
                    compute_ctx,
                    ggml_mul(compute_ctx, x, ggml_repeat(compute_ctx, norm_weight, x)),
                    ggml_repeat(compute_ctx, norm_bias, x));
            }
            // HuggingFace Wav2Vec2 `hidden_act="gelu"` uses the exact
            // error-function form, not the tanh approximation.
            x = ggml_gelu_erf(compute_ctx, x);
        }
        return x;
    }

    ggml_tensor* interpolate_frontend(ggml_tensor* x) {
        // PyTorch F.interpolate(..., mode="linear", align_corners=True).
        // A precomputed interpolation matrix avoids backend-specific image
        // interpolation semantics and preserves the exact audio time mapping.
        const int64_t input_frames = x->ne[0];
        constexpr int64_t output_frames = 30;
        interpolation_weights_.assign(
            static_cast<size_t>(input_frames * output_frames), 0.0f);

        for (int64_t output_index = 0; output_index < output_frames; ++output_index) {
            const float source =
                static_cast<float>(output_index) * static_cast<float>(input_frames - 1) /
                static_cast<float>(output_frames - 1);
            const int64_t lower = static_cast<int64_t>(std::floor(source));
            const int64_t upper = std::min(lower + 1, input_frames - 1);
            const float upper_weight = source - static_cast<float>(lower);
            const size_t column_offset = static_cast<size_t>(input_frames * output_index);
            interpolation_weights_[column_offset + static_cast<size_t>(lower)] +=
                1.0f - upper_weight;
            interpolation_weights_[column_offset + static_cast<size_t>(upper)] +=
                upper_weight;
        }

        auto* weights = ggml_new_tensor_2d(
            compute_ctx, GGML_TYPE_F32, input_frames, output_frames);
        set_backend_tensor_data(weights, interpolation_weights_.data());
        return ggml_mul_mat(compute_ctx, weights, x);
    }

    ggml_tensor* normalize_features(ggml_tensor* x) {
        // PyTorch receives [batch, time, channel] and normalizes channel.
        // GGML normalizes ne0, so transpose to [channel, time, batch] first.
        x = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, x));
        x = ggml_norm(compute_ctx, x, 1e-5f);
        auto* norm_weight = ggml_reshape_4d(compute_ctx, feature_norm_weight_, kChannels, 1, 1, 1);
        auto* norm_bias = ggml_reshape_4d(compute_ctx, feature_norm_bias_, kChannels, 1, 1, 1);
        x = ggml_add(
            compute_ctx,
            ggml_mul(compute_ctx, x, ggml_repeat(compute_ctx, norm_weight, x)),
            ggml_repeat(compute_ctx, norm_bias, x));
        return ggml_cont(compute_ctx, x);
    }

    ggml_tensor* forward_projection(ggml_tensor* x) {
        x = interpolate_frontend(x);
        x = normalize_features(x);
        return project_normalized(x);
    }

    ggml_tensor* project_normalized(ggml_tensor* x) {
        return ggml_ext_linear(
            compute_ctx,
            x,
            feature_projection_weight_,
            feature_projection_bias_,
            false,
            1.0f);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& pcm, int stage) {
        ggml_cgraph* graph = ggml_new_graph(compute_ctx);
        ggml_tensor* x = forward_frontend(make_input(pcm));
        if (stage == 1) {
            x = interpolate_frontend(x);
        } else if (stage == 2) {
            x = normalize_features(interpolate_frontend(x));
        } else if (stage >= 3) {
            x = forward_projection(x);
        }

        ggml_build_forward_expand(graph, x);
        return graph;
    }

    sd::Tensor<float> compute_frontend(const sd::Tensor<float>& pcm, int n_threads) {
        auto build = [&]() { return build_graph(pcm, 0); };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(build, n_threads, true), 3);
    }

    sd::Tensor<float> compute_projection(const sd::Tensor<float>& pcm, int n_threads) {
        const auto normalized = compute_normalized_cpu_reference(pcm, n_threads);
        auto build = [&]() {
            ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_tensor* x = project_normalized(make_input(normalized));
            ggml_build_forward_expand(graph, x);
            return graph;
        };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(build, n_threads, true), 3);
    }

    sd::Tensor<float> compute_interpolated(const sd::Tensor<float>& pcm, int n_threads) {
        auto build = [&]() { return build_graph(pcm, 1); };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(build, n_threads, true), 4);
    }

    sd::Tensor<float> compute_normalized(const sd::Tensor<float>& pcm, int n_threads) {
        auto build = [&]() { return build_graph(pcm, 2); };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(build, n_threads, true), 4);
    }

    sd::Tensor<float> compute_normalized_cpu_reference(
        const sd::Tensor<float>& pcm,
        int n_threads) {
        const auto frontend = compute_frontend(pcm, n_threads);
        const int64_t input_frames = frontend.shape()[0];
        constexpr int64_t output_frames = 30;
        sd::Tensor<float> normalized({kChannels, output_frames, 1});

        const auto* gamma = static_cast<const float*>(ggml_get_data(feature_norm_weight_));
        const auto* beta = static_cast<const float*>(ggml_get_data(feature_norm_bias_));
        for (int64_t output_index = 0; output_index < output_frames; ++output_index) {
            const float source =
                static_cast<float>(output_index) * static_cast<float>(input_frames - 1) /
                static_cast<float>(output_frames - 1);
            const int64_t lower = static_cast<int64_t>(std::floor(source));
            const int64_t upper = std::min(lower + 1, input_frames - 1);
            const float upper_weight = source - static_cast<float>(lower);

            float mean = 0.0f;
            for (int64_t channel = 0; channel < kChannels; ++channel) {
                const float value =
                    frontend.values()[lower + input_frames * channel] * (1.0f - upper_weight) +
                    frontend.values()[upper + input_frames * channel] * upper_weight;
                normalized.values()[channel + kChannels * output_index] = value;
                mean += value;
            }
            mean /= kChannels;

            float variance = 0.0f;
            for (int64_t channel = 0; channel < kChannels; ++channel) {
                const float delta =
                    normalized.values()[channel + kChannels * output_index] - mean;
                variance += delta * delta;
            }
            const float inverse_std = 1.0f / std::sqrt(variance / kChannels + 1e-5f);
            for (int64_t channel = 0; channel < kChannels; ++channel) {
                const float value =
                    normalized.values()[channel + kChannels * output_index];
                normalized.values()[channel + kChannels * output_index] =
                    (value - mean) * inverse_std * gamma[channel] + beta[channel];
            }
        }
        return normalized;
    }

    sd::Tensor<float> compute_position_cpu_reference(
        const sd::Tensor<float>& pcm,
        int n_threads) {
        const auto projected = compute_projection(pcm, n_threads);
        constexpr int64_t channels = 768;
        constexpr int64_t groups = 16;
        constexpr int64_t channels_per_group = channels / groups;
        constexpr int64_t kernel = 128;
        constexpr int64_t padding = kernel / 2;
        constexpr int64_t frames = 30;

        const auto* weight_g = static_cast<const float*>(ggml_get_data(pos_weight_g_));
        const auto* weight_v = static_cast<const float*>(ggml_get_data(pos_weight_v_));
        const auto* bias = static_cast<const float*>(ggml_get_data(pos_bias_));
        sd::Tensor<float> output({channels, frames, 1});

        std::array<float, kernel> norms{};
        for (int64_t kernel_index = 0; kernel_index < kernel; ++kernel_index) {
            double sum = 0.0;
            for (int64_t output_channel = 0; output_channel < channels; ++output_channel) {
                for (int64_t input_channel = 0; input_channel < channels_per_group; ++input_channel) {
                    const size_t offset = static_cast<size_t>(
                        kernel_index + kernel * (input_channel + channels_per_group * output_channel));
                    sum += static_cast<double>(weight_v[offset]) * weight_v[offset];
                }
            }
            norms[kernel_index] = static_cast<float>(std::sqrt(sum));
        }

        for (int64_t output_channel = 0; output_channel < channels; ++output_channel) {
            const int64_t group = output_channel / channels_per_group;
            for (int64_t frame = 0; frame < frames; ++frame) {
                float value = bias[output_channel];
                for (int64_t input_channel = 0; input_channel < channels_per_group; ++input_channel) {
                    const int64_t source_channel = group * channels_per_group + input_channel;
                    for (int64_t kernel_index = 0; kernel_index < kernel; ++kernel_index) {
                        const int64_t source_frame = frame + kernel_index - padding;
                        if (source_frame < 0 || source_frame >= frames) {
                            continue;
                        }
                        const size_t weight_offset = static_cast<size_t>(
                            kernel_index + kernel * (input_channel + channels_per_group * output_channel));
                        const float weight =
                            weight_v[weight_offset] * weight_g[kernel_index] / norms[kernel_index];
                        value += weight *
                                 projected.values()[source_channel + channels * source_frame];
                    }
                }
                // Wav2Vec2SamePadLayer drops the final padded frame. This
                // loop emits the retained 30 frames directly.
                output.values()[output_channel + channels * frame] =
                    0.5f * value * (1.0f + std::erf(value * 0.70710678118f));
            }
        }
        return output;
    }

private:
    std::array<ggml_tensor*, kLayers> conv_weights_{};
    ggml_tensor* norm_weight_ = nullptr;
    ggml_tensor* norm_bias_ = nullptr;
    ggml_tensor* feature_norm_weight_ = nullptr;
    ggml_tensor* feature_norm_bias_ = nullptr;
    ggml_tensor* feature_projection_weight_ = nullptr;
    ggml_tensor* feature_projection_bias_ = nullptr;
    ggml_tensor* pos_weight_g_ = nullptr;
    ggml_tensor* pos_weight_v_ = nullptr;
    ggml_tensor* pos_bias_ = nullptr;
    std::vector<float> interpolation_weights_;
};
