#ifndef __SD_MODEL_VAE_LTX_VAE_HPP__
#define __SD_MODEL_VAE_LTX_VAE_HPP__

#include <algorithm>
#include <array>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "model/diffusion/ltxv.hpp"
#include "model/vae/vae.hpp"
#include "model/vae/wan_vae.hpp"
#include "model_loader.h"
#include "ltx_vae_temporal.hpp"

namespace LTXVAE {

    struct EncoderStreamingGraphState {
        std::vector<ggml_tensor*> previous_conv_history;
        std::vector<ggml_tensor*> next_conv_history;
        std::array<ggml_tensor*, LTX_ENCODER_TEMPORAL_LEVELS> previous_pending_x = {};
        std::array<ggml_tensor*, LTX_ENCODER_TEMPORAL_LEVELS> previous_pending_h = {};
        std::array<ggml_tensor*, LTX_ENCODER_TEMPORAL_LEVELS> next_pending_x     = {};
        std::array<ggml_tensor*, LTX_ENCODER_TEMPORAL_LEVELS> next_pending_h     = {};
        size_t conv_index = 0;
        int temporal_level = 0;
        bool allow_missing_history = false;
        bool preflight             = false;
        bool exact                 = true;
    };

    static inline ggml_tensor* apply_scale_shift(ggml_context* ctx,
                                                 ggml_tensor* x,
                                                 ggml_tensor* scale,
                                                 ggml_tensor* shift) {
        x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x = ggml_add(ctx, x, shift);
        return x;
    }

    static inline ggml_tensor* reshape_channel_broadcast(ggml_context* ctx,
                                                         ggml_tensor* x) {
        return ggml_reshape_4d(ctx, x, 1, 1, 1, ggml_nelements(x));
    }

    static inline std::pair<ggml_tensor*, ggml_tensor*> get_shift_scale(ggml_context* ctx,
                                                                        ggml_tensor* table,
                                                                        ggml_tensor* timestep,
                                                                        int64_t channels,
                                                                        int parts) {
        GGML_ASSERT(timestep != nullptr);
        GGML_ASSERT(ggml_nelements(timestep) == channels * parts);

        auto timestep_view = ggml_reshape_2d(ctx, timestep, channels, parts);
        auto values        = ggml_add(ctx, table, timestep_view);
        auto chunks        = ggml_ext_chunk(ctx, values, parts, 1, false);
        auto shift         = reshape_channel_broadcast(ctx, ggml_cont(ctx, chunks[0]));
        auto scale         = reshape_channel_broadcast(ctx, ggml_cont(ctx, chunks[1]));
        return {shift, scale};
    }

    static inline ggml_tensor* depth_to_space_3d(ggml_context* ctx,
                                                 ggml_tensor* x,
                                                 int64_t c,
                                                 int factor_t,
                                                 int factor_s,
                                                 bool drop_first_temporal_frame) {
        // x: [B*c*p1*p2*p3, T, H, W], B == 1, p2 == p3 == factor_s, p1 == factor_t
        // return: [B*c, T*p1, H*p2, W*p2]
        // Match: rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)")
        const int64_t T = x->ne[2];
        const int64_t H = x->ne[1];
        const int64_t W = x->ne[0];

        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 1, 3, 2));        // [T, C, H, W]
        x = ggml_reshape_4d(ctx, x, W, H, factor_s, factor_s * factor_t * c * T);  // [T*c*p1*p2, p3, H, W]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));        // [T*c*p1*p2, H, W, p3]
        x = ggml_reshape_4d(ctx, x, factor_s * W, H, factor_s, factor_t * c * T);  // [T*c*p1, p2, H, W*p3]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));        // [T*c*p1, H, p2, W*p3]
        x = ggml_reshape_4d(ctx, x, factor_s * W * factor_s * H, factor_t, c, T);  // [T, c, p1, H*p2*W*p3]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 1, 3, 2));        // [c, T, p1, H*p2*W*p3]
        x = ggml_reshape_4d(ctx, x, factor_s * W, factor_s * H, factor_t * T, c);  // [T, c, T*p1, H*p2*W*p3]

        if (drop_first_temporal_frame && factor_t > 1 && x->ne[2] > 0) {
            x = ggml_ext_slice(ctx, x, 2, 1, x->ne[2]);
        }

        return x;
    }

    static inline ggml_tensor* patchify(ggml_context* ctx,
                                        ggml_tensor* x,
                                        int patch_size) {
        return WAN::WanVAE::patchify(ctx, x, patch_size, 1);
    }

    class CausalConv3d : public GGMLBlock {
    protected:
        int time_kernel_size;

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     int kernel_size                  = 3,
                     std::tuple<int, int, int> stride = {1, 1, 1},
                     int dilation                     = 1,
                     bool bias                        = true,
                     bool force_prec_f32              = false) {
            time_kernel_size = kernel_size;
            blocks["conv"]   = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                     out_channels,
                                                                     {kernel_size, kernel_size, kernel_size},
                                                                     stride,
                                                                     {0, kernel_size / 2, kernel_size / 2},
                                                                     {dilation, 1, 1},
                                                                     bias,
                                                                     force_prec_f32));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool causal = true) {
            // x: [B*C, T, H, W], B == 1
            auto conv = std::dynamic_pointer_cast<Conv3d>(blocks["conv"]);

            if (causal) {
                auto first_frame     = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                auto first_frame_pad = first_frame;
                for (int i = 1; i < time_kernel_size - 1; i++) {
                    first_frame_pad = ggml_concat(ctx->ggml_ctx, first_frame_pad, first_frame, 2);
                }
                x = ggml_concat(ctx->ggml_ctx, first_frame_pad, x, 2);
            } else {
                auto first_frame     = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                auto first_frame_pad = first_frame;
                for (int i = 1; i < (time_kernel_size - 1) / 2; i++) {
                    first_frame_pad = ggml_concat(ctx->ggml_ctx, first_frame_pad, first_frame, 2);
                }

                auto last_frame     = ggml_ext_slice(ctx->ggml_ctx, x, 2, x->ne[2] - 1, x->ne[2]);
                auto last_frame_pad = last_frame;
                for (int i = 1; i < (time_kernel_size - 1) / 2; i++) {
                    last_frame_pad = ggml_concat(ctx->ggml_ctx, last_frame_pad, last_frame, 2);
                }
                x = ggml_concat(ctx->ggml_ctx, first_frame_pad, x, 2);
                x = ggml_concat(ctx->ggml_ctx, x, last_frame_pad, 2);
            }
            return conv->forward(ctx, x);
        }

        ggml_tensor* forward_encoder_chunk(GGMLRunnerContext* ctx,
                                           ggml_tensor* x,
                                           EncoderStreamingGraphState& state) {
            auto conv     = std::dynamic_pointer_cast<Conv3d>(blocks["conv"]);
            const int pad = time_kernel_size - 1;
            GGML_ASSERT(x != nullptr);
            GGML_ASSERT(x->ne[2] > 0);

            ggml_tensor* history = state.conv_index < state.previous_conv_history.size()
                                       ? state.previous_conv_history[state.conv_index]
                                       : nullptr;
            if (history == nullptr) {
                if (!state.allow_missing_history && !state.preflight) {
                    state.exact = false;
                }
                auto first_frame = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                history         = first_frame;
                for (int i = 1; i < pad; ++i) {
                    history = ggml_concat(ctx->ggml_ctx, history, first_frame, 2);
                }
            }

            GGML_ASSERT(history->ne[2] == pad);
            auto padded = pad > 0 ? ggml_concat(ctx->ggml_ctx, history, x, 2) : x;
            auto next_history = ggml_ext_slice(ctx->ggml_ctx,
                                               padded,
                                               2,
                                               padded->ne[2] - pad,
                                               padded->ne[2]);
            next_history = ggml_cont(ctx->ggml_ctx, next_history);
            state.next_conv_history.push_back(next_history);
            state.conv_index++;
            return conv->forward(ctx, padded);
        }

        // Chunked forward: uses feat_map to carry temporal context across frames.
        // feat_map[feat_idx] holds the last `pad` frames from the previous chunk at
        // this layer.  nullptr means first chunk → fall back to repeat-first-frame.
        // The cache entry is a contiguous copy (not a view) so that the large
        // intermediate tensor `x` can be freed by GGML after this iteration ends.
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             std::vector<ggml_tensor*>& feat_map,
                             int& feat_idx,
                             int chunk_idx,
                             bool causal      = true,
                             int temporal_pad = 0) {
            auto conv     = std::dynamic_pointer_cast<Conv3d>(blocks["conv"]);
            const int pad = causal ? (time_kernel_size - 1) : (time_kernel_size - 1) / 2;

            ggml_tensor* prev = (feat_idx < (int)feat_map.size()) ? feat_map[feat_idx] : nullptr;

            GGML_ASSERT(x->ne[2] >= temporal_pad);

            int end_idx   = (int)x->ne[2] - temporal_pad;
            int start_idx = std::max(end_idx - pad, 0);

            // Save a contiguous copy of the last `pad` frames so the large `x`
            // tensor is not kept alive across iterations by a dangling view.
            if (feat_idx < (int)feat_map.size() && end_idx - start_idx > 0) {
                GGML_ASSERT(start_idx >= 0);
                GGML_ASSERT(end_idx > 0);

                auto slice         = ggml_ext_slice(ctx->ggml_ctx, x, 2, start_idx, end_idx);
                feat_map[feat_idx] = ggml_cont(ctx->ggml_ctx, slice);
            }
            feat_idx++;

            if (pad > 0) {
                ggml_tensor* left_pad;
                if (prev != nullptr) {
                    left_pad = prev;
                } else {
                    auto first_frame = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                    left_pad         = first_frame;
                    for (int i = 1; i < pad; i++) {
                        left_pad = ggml_concat(ctx->ggml_ctx, left_pad, first_frame, 2);
                    }
                }
                x = ggml_concat(ctx->ggml_ctx, left_pad, x, 2);
            }

            if (!causal && pad > 0) {
                auto last_frame = ggml_ext_slice(ctx->ggml_ctx, x, 2, x->ne[2] - 1, x->ne[2]);
                auto right_pad  = last_frame;
                for (int i = 1; i < pad; i++) {
                    right_pad = ggml_concat(ctx->ggml_ctx, right_pad, last_frame, 2);
                }
                x = ggml_concat(ctx->ggml_ctx, x, right_pad, 2);
            }

            return conv->forward(ctx, x);
        }
    };

    struct PixelNorm3D : public UnaryBlock {
        float eps;

        PixelNorm3D(float eps = 1e-8f)
            : eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto h = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));
            h      = ggml_rms_norm(ctx->ggml_ctx, h, eps);
            h      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));
            return h;
        }
    };

    struct PixArtAlphaCombinedTimestepSizeEmbeddings : public GGMLBlock {
        int64_t embedding_dim;

        PixArtAlphaCombinedTimestepSizeEmbeddings(int64_t embedding_dim)
            : embedding_dim(embedding_dim) {
            blocks["timestep_embedder"] = std::make_shared<LTXV::TimestepEmbedder>(embedding_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
            auto timestep_embedder = std::dynamic_pointer_cast<LTXV::TimestepEmbedder>(blocks["timestep_embedder"]);
            return timestep_embedder->forward(ctx, timestep);
        }
    };

    struct ResnetBlock3D : public GGMLBlock {
        int64_t channels;
        bool timestep_conditioning;

    protected:
        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            if (timestep_conditioning) {
                params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, 4);
            }
        }

    public:
        ResnetBlock3D(int64_t channels,
                      float eps                  = 1e-6f,
                      bool timestep_conditioning = false,
                      bool force_conv_prec_f32   = false)
            : channels(channels), timestep_conditioning(timestep_conditioning) {
            blocks["norm1"] = std::make_shared<PixelNorm3D>(eps);
            blocks["conv1"] = std::make_shared<CausalConv3d>(
                channels, channels, 3, std::tuple<int, int, int>{1, 1, 1},
                1, true, force_conv_prec_f32);
            blocks["norm2"] = std::make_shared<PixelNorm3D>(eps);
            blocks["conv2"] = std::make_shared<CausalConv3d>(
                channels, channels, 3, std::tuple<int, int, int>{1, 1, 1},
                1, true, force_conv_prec_f32);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep = nullptr,
                             bool causal           = false) {
            auto norm1 = std::dynamic_pointer_cast<PixelNorm3D>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<PixelNorm3D>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            ggml_tensor* shift1 = nullptr;
            ggml_tensor* scale1 = nullptr;
            ggml_tensor* shift2 = nullptr;
            ggml_tensor* scale2 = nullptr;
            if (timestep_conditioning) {
                GGML_ASSERT(timestep != nullptr);
                auto values = ggml_add(ctx->ggml_ctx,
                                       params["scale_shift_table"],
                                       ggml_reshape_2d(ctx->ggml_ctx, timestep, channels, 4));
                auto chunks = ggml_ext_chunk(ctx->ggml_ctx, values, 4, 1, false);
                shift1      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[0]));
                scale1      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[1]));
                shift2      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[2]));
                scale2      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[3]));
            }

            auto h = norm1->forward(ctx, x);
            if (timestep_conditioning) {
                h = apply_scale_shift(ctx->ggml_ctx, h, scale1, shift1);
            }
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv1->forward(ctx, h, causal);

            h = norm2->forward(ctx, h);
            if (timestep_conditioning) {
                h = apply_scale_shift(ctx->ggml_ctx, h, scale2, shift2);
            }
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv2->forward(ctx, h, causal);

            return ggml_add(ctx->ggml_ctx, h, x);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             bool causal,
                             std::vector<ggml_tensor*>& feat_map,
                             int& feat_idx,
                             int chunk_idx,
                             int temporal_pad = 0) {
            auto norm1 = std::dynamic_pointer_cast<PixelNorm3D>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<PixelNorm3D>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            ggml_tensor* shift1 = nullptr;
            ggml_tensor* scale1 = nullptr;
            ggml_tensor* shift2 = nullptr;
            ggml_tensor* scale2 = nullptr;
            if (timestep_conditioning) {
                GGML_ASSERT(timestep != nullptr);
                auto values = ggml_add(ctx->ggml_ctx,
                                       params["scale_shift_table"],
                                       ggml_reshape_2d(ctx->ggml_ctx, timestep, channels, 4));
                auto chunks = ggml_ext_chunk(ctx->ggml_ctx, values, 4, 1, false);
                shift1      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[0]));
                scale1      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[1]));
                shift2      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[2]));
                scale2      = reshape_channel_broadcast(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, chunks[3]));
            }

            auto h = norm1->forward(ctx, x);
            if (timestep_conditioning) {
                h = apply_scale_shift(ctx->ggml_ctx, h, scale1, shift1);
            }
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv1->forward(ctx, h, feat_map, feat_idx, chunk_idx, causal, temporal_pad);

            h = norm2->forward(ctx, h);
            if (timestep_conditioning) {
                h = apply_scale_shift(ctx->ggml_ctx, h, scale2, shift2);
            }
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv2->forward(ctx, h, feat_map, feat_idx, chunk_idx, causal, temporal_pad);

            return ggml_add(ctx->ggml_ctx, h, x);
        }

        ggml_tensor* forward_encoder_chunk(GGMLRunnerContext* ctx,
                                           ggml_tensor* x,
                                           EncoderStreamingGraphState& state) {
            auto norm1 = std::dynamic_pointer_cast<PixelNorm3D>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<PixelNorm3D>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto h = norm1->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = conv1->forward_encoder_chunk(ctx, h, state);
            h      = norm2->forward(ctx, h);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = conv2->forward_encoder_chunk(ctx, h, state);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    struct UNetMidBlock3D : public GGMLBlock {
        int64_t channels;
        int num_layers;
        bool timestep_conditioning;

        UNetMidBlock3D(int64_t channels,
                       int num_layers,
                       bool timestep_conditioning,
                       bool force_conv_prec_f32 = false)
            : channels(channels),
              num_layers(num_layers),
              timestep_conditioning(timestep_conditioning) {
            if (timestep_conditioning) {
                blocks["time_embedder"] = std::make_shared<PixArtAlphaCombinedTimestepSizeEmbeddings>(channels * 4);
            }
            for (int i = 0; i < num_layers; i++) {
                blocks["res_blocks." + std::to_string(i)] =
                    std::make_shared<ResnetBlock3D>(
                        channels,
                        1e-6f,
                        timestep_conditioning,
                        force_conv_prec_f32);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep = nullptr,
                             bool causal           = false) {
            ggml_tensor* timestep_embed = nullptr;
            if (timestep_conditioning) {
                GGML_ASSERT(timestep != nullptr);
                auto time_embedder = std::dynamic_pointer_cast<PixArtAlphaCombinedTimestepSizeEmbeddings>(blocks["time_embedder"]);
                timestep_embed     = time_embedder->forward(ctx, timestep);
            }

            for (int i = 0; i < num_layers; i++) {
                auto resnet = std::dynamic_pointer_cast<ResnetBlock3D>(blocks["res_blocks." + std::to_string(i)]);
                x           = resnet->forward(ctx, x, timestep_embed, causal);
            }
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             bool causal,
                             std::vector<ggml_tensor*>& feat_map,
                             int& feat_idx,
                             int chunk_idx,
                             int temporal_pad = 0) {
            ggml_tensor* timestep_embed = nullptr;
            if (timestep_conditioning) {
                GGML_ASSERT(timestep != nullptr);
                auto time_embedder = std::dynamic_pointer_cast<PixArtAlphaCombinedTimestepSizeEmbeddings>(blocks["time_embedder"]);
                timestep_embed     = time_embedder->forward(ctx, timestep);
            }
            for (int i = 0; i < num_layers; i++) {
                auto resnet = std::dynamic_pointer_cast<ResnetBlock3D>(blocks["res_blocks." + std::to_string(i)]);
                x           = resnet->forward(ctx, x, timestep_embed, causal, feat_map, feat_idx, chunk_idx, temporal_pad);
            }
            return x;
        }

        ggml_tensor* forward_encoder_chunk(GGMLRunnerContext* ctx,
                                           ggml_tensor* x,
                                           EncoderStreamingGraphState& state) {
            GGML_ASSERT(!timestep_conditioning);
            for (int i = 0; i < num_layers; i++) {
                auto resnet = std::dynamic_pointer_cast<ResnetBlock3D>(blocks["res_blocks." + std::to_string(i)]);
                x           = resnet->forward_encoder_chunk(ctx, x, state);
            }
            return x;
        }
    };

    struct DepthToSpaceUpsample : public GGMLBlock {
        int64_t in_channels;
        int factor_t;
        int factor_s;
        int out_channels_reduction_factor;
        bool residual;

        DepthToSpaceUpsample(int64_t in_channels,
                             int factor_t                      = 2,
                             int factor_s                      = 2,
                             int out_channels_reduction_factor = 2,
                             bool residual                     = true)
            : in_channels(in_channels),
              factor_t(factor_t),
              factor_s(factor_s),
              out_channels_reduction_factor(out_channels_reduction_factor),
              residual(residual) {
            const int64_t factor  = static_cast<int64_t>(factor_t) * static_cast<int64_t>(factor_s) * static_cast<int64_t>(factor_s);
            const int64_t out_dim = (factor * in_channels) / out_channels_reduction_factor;
            blocks["conv"]        = std::make_shared<CausalConv3d>(in_channels, out_dim, 3);
        }

        int64_t get_output_channels() const {
            return in_channels / out_channels_reduction_factor;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool causal = false) {
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);

            ggml_tensor* x_in = nullptr;
            if (residual) {
                x_in       = depth_to_space_3d(ctx->ggml_ctx, x, in_channels / (factor_t * factor_s * factor_s), factor_t, factor_s, factor_t > 1);
                int repeat = (factor_t * factor_s * factor_s) / out_channels_reduction_factor;
                auto res   = x_in;
                for (int i = 1; i < repeat; i++) {
                    res = ggml_concat(ctx->ggml_ctx, res, x_in, 3);
                }
                x_in = res;
            }

            x = conv->forward(ctx, x, causal);
            x = depth_to_space_3d(ctx->ggml_ctx, x, get_output_channels(), factor_t, factor_s, factor_t > 1);
            if (residual) {
                x = ggml_add(ctx->ggml_ctx, x, x_in);
            }
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool causal,
                             std::vector<ggml_tensor*>& feat_map,
                             int& feat_idx,
                             int chunk_idx,
                             int temporal_pad = 0) {
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);

            bool drop_first = (chunk_idx == 0) && (factor_t > 1);

            ggml_tensor* x_in = nullptr;
            if (residual) {
                x_in       = depth_to_space_3d(ctx->ggml_ctx, x, in_channels / (factor_t * factor_s * factor_s), factor_t, factor_s, drop_first);
                int repeat = (factor_t * factor_s * factor_s) / out_channels_reduction_factor;
                auto res   = x_in;
                for (int i = 1; i < repeat; i++) {
                    res = ggml_concat(ctx->ggml_ctx, res, x_in, 3);
                }
                x_in = res;
            }

            x = conv->forward(ctx, x, feat_map, feat_idx, chunk_idx, causal, temporal_pad);
            x = depth_to_space_3d(ctx->ggml_ctx, x, get_output_channels(), factor_t, factor_s, drop_first);
            if (residual) {
                x = ggml_add(ctx->ggml_ctx, x, x_in);
            }
            return x;
        }
    };

    struct SpaceToDepthDownsample : public GGMLBlock {
        int64_t in_channels;
        int64_t out_channels;
        int factor_t;
        int factor_s;

        SpaceToDepthDownsample(int64_t in_channels,
                               int64_t out_channels,
                               int factor_t,
                               int factor_s,
                               bool force_conv_prec_f32 = false)
            : in_channels(in_channels),
              out_channels(out_channels),
              factor_t(factor_t),
              factor_s(factor_s) {
            const int64_t factor = static_cast<int64_t>(factor_t) * static_cast<int64_t>(factor_s) * static_cast<int64_t>(factor_s);
            GGML_ASSERT(out_channels % factor == 0);

            blocks["conv"]            = std::make_shared<CausalConv3d>(in_channels,
                                                            out_channels / factor,
                                                            3,
                                                            std::tuple<int, int, int>{1, 1, 1},
                                                            1,
                                                            true,
                                                            force_conv_prec_f32);
            blocks["skip_downsample"] = std::make_shared<WAN::AvgDown3D>(in_channels, out_channels, factor_t, factor_s);
            blocks["conv_downsample"] = std::make_shared<WAN::AvgDown3D>(out_channels / factor, out_channels, factor_t, factor_s);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool causal = true) {
            auto conv            = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);
            auto skip_downsample = std::dynamic_pointer_cast<WAN::AvgDown3D>(blocks["skip_downsample"]);
            auto conv_downsample = std::dynamic_pointer_cast<WAN::AvgDown3D>(blocks["conv_downsample"]);

            if (factor_t > 1 && x->ne[2] > 0) {
                auto first_frame     = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                auto first_frame_pad = first_frame;
                for (int i = 1; i < factor_t - 1; ++i) {
                    first_frame_pad = ggml_concat(ctx->ggml_ctx, first_frame_pad, first_frame, 2);
                }
                x = ggml_concat(ctx->ggml_ctx, first_frame_pad, x, 2);
            }

            auto residual = skip_downsample->forward(ctx, x);
            auto h        = conv->forward(ctx, x, causal);
            h             = conv_downsample->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, residual);
        }

        ggml_tensor* forward_encoder_chunk(GGMLRunnerContext* ctx,
                                           ggml_tensor* x,
                                           const EncoderTemporalPhase& phase,
                                           EncoderStreamingGraphState& state) {
            auto conv            = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);
            auto skip_downsample = std::dynamic_pointer_cast<WAN::AvgDown3D>(blocks["skip_downsample"]);
            auto conv_downsample = std::dynamic_pointer_cast<WAN::AvgDown3D>(blocks["conv_downsample"]);

            if (factor_t == 1) {
                auto h        = conv->forward_encoder_chunk(ctx, x, state);
                auto residual = skip_downsample->forward(ctx, x);
                h             = conv_downsample->forward(ctx, h);
                return ggml_add(ctx->ggml_ctx, h, residual);
            }

            GGML_ASSERT(factor_t == 2);
            const int level = state.temporal_level++;
            GGML_ASSERT(level < LTX_ENCODER_TEMPORAL_LEVELS);
            if (x == nullptr || x->ne[2] != phase.level_input_frames[level]) {
                state.exact = false;
                return nullptr;
            }

            if (phase.first_frame[level]) {
                auto first_frame = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x                = ggml_concat(ctx->ggml_ctx, first_frame, x, 2);
            }

            auto h = conv->forward_encoder_chunk(ctx, x, state);
            if (phase.pending_before[level]) {
                ggml_tensor* pending_x = state.previous_pending_x[level];
                ggml_tensor* pending_h = state.previous_pending_h[level];
                if (pending_x == nullptr || pending_h == nullptr) {
                    if (!state.preflight) {
                        state.exact = false;
                    }
                    pending_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                    pending_h = ggml_ext_slice(ctx->ggml_ctx, h, 2, 0, 1);
                }
                x = ggml_concat(ctx->ggml_ctx, pending_x, x, 2);
                h = ggml_concat(ctx->ggml_ctx, pending_h, h, 2);
            }

            const int64_t grouped_frames = x->ne[2];
            const int64_t complete_frames = grouped_frames - (phase.pending_after[level] ? 1 : 0);
            if (grouped_frames != phase.level_output_frames[level] * 2 +
                                      (phase.pending_after[level] ? 1 : 0)) {
                state.exact = false;
            }

            // Keep one fixed-size slot for each branch even when the phase has no
            // pending frame. The phase bit decides whether the slot is consumed;
            // fixed slots let the cache replace state in-place without retaining
            // stale tensors or growing with the number of chunks.
            state.next_pending_x[level] = ggml_cont(
                ctx->ggml_ctx,
                ggml_ext_slice(ctx->ggml_ctx, x, 2, grouped_frames - 1, grouped_frames));
            state.next_pending_h[level] = ggml_cont(
                ctx->ggml_ctx,
                ggml_ext_slice(ctx->ggml_ctx, h, 2, grouped_frames - 1, grouped_frames));

            if (complete_frames == 0) {
                return nullptr;
            }
            if (complete_frames != grouped_frames) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, complete_frames);
                h = ggml_ext_slice(ctx->ggml_ctx, h, 2, 0, complete_frames);
            }

            auto residual = skip_downsample->forward(ctx, x);
            h             = conv_downsample->forward(ctx, h);
            state.allow_missing_history =
                level + 1 < LTX_ENCODER_TEMPORAL_LEVELS
                    ? phase.first_frame[level + 1]
                    : phase.first_frame[level];
            return ggml_add(ctx->ggml_ctx, h, residual);
        }
    };

    struct PerChannelStatistics : public GGMLBlock {
    protected:
        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["std-of-means"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
            params["mean-of-means"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        }

    public:
        ggml_tensor* un_normalize(GGMLRunnerContext* ctx,
                                  ggml_tensor* x) {
            auto std_tensor  = reshape_channel_broadcast(ctx->ggml_ctx, params["std-of-means"]);
            auto mean_tensor = reshape_channel_broadcast(ctx->ggml_ctx, params["mean-of-means"]);
            return ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, std_tensor), mean_tensor);
        }

        ggml_tensor* normalize(GGMLRunnerContext* ctx,
                               ggml_tensor* x) {
            auto std_tensor  = reshape_channel_broadcast(ctx->ggml_ctx, params["std-of-means"]);
            auto mean_tensor = reshape_channel_broadcast(ctx->ggml_ctx, params["mean-of-means"]);
            return ggml_div(ctx->ggml_ctx, ggml_sub(ctx->ggml_ctx, x, mean_tensor), std_tensor);
        }
    };

    struct DecoderConfig {
        struct Block {
            std::string type;
            int num_layers = 0;
            int multiplier = 1;
        };

        std::vector<Block> blocks;
    };

    struct EncoderConfig {
        struct Block {
            std::string type;
            int num_layers = 0;
            int multiplier = 1;
        };

        std::vector<Block> blocks;
    };

    static inline EncoderConfig get_default_encoder_config(int version);

    static inline bool has_tensor(const String2TensorStorage& tensor_storage_map,
                                  const std::string& name) {
        return tensor_storage_map.find(name) != tensor_storage_map.end();
    }

    static inline int64_t get_tensor_ne0(const String2TensorStorage& tensor_storage_map,
                                         const std::string& name,
                                         int64_t fallback = 0) {
        auto iter = tensor_storage_map.find(name);
        if (iter == tensor_storage_map.end()) {
            return fallback;
        }
        return iter->second.ne[0];
    }

    static inline DecoderConfig infer_decoder_config_from_weights(const String2TensorStorage& tensor_storage_map,
                                                                  const std::string& prefix,
                                                                  int64_t conv_in_channels) {
        DecoderConfig cfg;
        const std::string decoder_prefix = prefix + ".decoder.up_blocks.";

        int64_t current_channels = conv_in_channels;
        for (int block_idx = 0;; ++block_idx) {
            const std::string block_prefix = decoder_prefix + std::to_string(block_idx);
            const std::string res0_bias    = block_prefix + ".res_blocks.0.conv1.conv.bias";
            const std::string conv_bias    = block_prefix + ".conv.conv.bias";

            if (has_tensor(tensor_storage_map, res0_bias)) {
                int num_layers = 0;
                while (has_tensor(tensor_storage_map,
                                  block_prefix + ".res_blocks." + std::to_string(num_layers) + ".conv1.conv.bias")) {
                    num_layers++;
                }
                cfg.blocks.push_back({"res_x", num_layers, 1});
                current_channels = get_tensor_ne0(tensor_storage_map, res0_bias, current_channels);
                continue;
            }

            if (!has_tensor(tensor_storage_map, conv_bias)) {
                break;
            }

            int64_t next_channels = 0;
            for (int next_idx = block_idx + 1;; ++next_idx) {
                const std::string next_res0_bias = decoder_prefix + std::to_string(next_idx) + ".res_blocks.0.conv1.conv.bias";
                const std::string next_conv_bias = decoder_prefix + std::to_string(next_idx) + ".conv.conv.bias";
                if (has_tensor(tensor_storage_map, next_res0_bias)) {
                    next_channels = get_tensor_ne0(tensor_storage_map, next_res0_bias);
                    break;
                }
                if (!has_tensor(tensor_storage_map, next_conv_bias)) {
                    break;
                }
            }
            if (next_channels <= 0 || current_channels % next_channels != 0) {
                next_channels = std::max<int64_t>(1, current_channels / 2);
            }

            const int64_t conv_out_dim = get_tensor_ne0(tensor_storage_map, conv_bias);
            const int64_t reduction    = std::max<int64_t>(1, current_channels / next_channels);
            const int64_t factor       = next_channels > 0 ? conv_out_dim / next_channels : 0;

            if (factor == 8) {
                cfg.blocks.push_back({"compress_all", 0, static_cast<int>(reduction)});
            } else if (factor == 4) {
                cfg.blocks.push_back({"compress_space", 0, static_cast<int>(reduction)});
            } else if (factor == 2) {
                cfg.blocks.push_back({"compress_time", 0, static_cast<int>(reduction)});
            } else {
                LOG_WARN("unexpected LTX VAE upsample factor at '%s': conv_out=%lld current=%lld next=%lld, falling back to compress_all x%d",
                         block_prefix.c_str(),
                         (long long)conv_out_dim,
                         (long long)current_channels,
                         (long long)next_channels,
                         (int)reduction);
                cfg.blocks.push_back({"compress_all", 0, static_cast<int>(reduction)});
            }
            current_channels = next_channels;
        }

        return cfg;
    }

    static inline EncoderConfig infer_encoder_config_from_weights(const String2TensorStorage& tensor_storage_map,
                                                                  const std::string& prefix,
                                                                  int version) {
        EncoderConfig cfg;
        const std::string encoder_prefix = prefix + ".encoder.down_blocks.";

        int64_t current_channels = get_tensor_ne0(tensor_storage_map,
                                                  prefix + ".encoder.conv_in.conv.bias",
                                                  0);
        for (int block_idx = 0;; ++block_idx) {
            const std::string block_prefix = encoder_prefix + std::to_string(block_idx);
            const std::string res0_bias    = block_prefix + ".res_blocks.0.conv1.conv.bias";
            const std::string conv_bias    = block_prefix + ".conv.conv.bias";

            if (has_tensor(tensor_storage_map, res0_bias)) {
                int num_layers = 0;
                while (has_tensor(tensor_storage_map,
                                  block_prefix + ".res_blocks." + std::to_string(num_layers) + ".conv1.conv.bias")) {
                    num_layers++;
                }
                cfg.blocks.push_back({"res_x", num_layers, 1});
                current_channels = get_tensor_ne0(tensor_storage_map, res0_bias, current_channels);
                continue;
            }

            if (!has_tensor(tensor_storage_map, conv_bias)) {
                break;
            }

            const int64_t conv_channels = get_tensor_ne0(tensor_storage_map, conv_bias);
            int64_t next_channels       = 0;
            for (int next_idx = block_idx + 1;; ++next_idx) {
                const std::string next_res0_bias = encoder_prefix + std::to_string(next_idx) + ".res_blocks.0.conv1.conv.bias";
                const std::string next_conv_bias = encoder_prefix + std::to_string(next_idx) + ".conv.conv.bias";
                if (has_tensor(tensor_storage_map, next_res0_bias)) {
                    next_channels = get_tensor_ne0(tensor_storage_map, next_res0_bias);
                    break;
                }
                if (!has_tensor(tensor_storage_map, next_conv_bias)) {
                    break;
                }
            }

            const int64_t multiplier = current_channels > 0 && next_channels > 0 && next_channels % current_channels == 0
                                           ? std::max<int64_t>(1, next_channels / current_channels)
                                           : 1;
            const int64_t factor     = conv_channels > 0 && next_channels > 0 && next_channels % conv_channels == 0
                                           ? next_channels / conv_channels
                                           : 0;

            if (factor == 8) {
                cfg.blocks.push_back({"compress_all_res", 0, static_cast<int>(multiplier)});
            } else if (factor == 4) {
                cfg.blocks.push_back({"compress_space_res", 0, static_cast<int>(multiplier)});
            } else if (factor == 2) {
                cfg.blocks.push_back({"compress_time_res", 0, static_cast<int>(multiplier)});
            } else {
                LOG_WARN("unexpected LTX VAE encoder downsample factor at '%s': conv_out=%lld current=%lld next=%lld, falling back to compress_all_res x%d",
                         block_prefix.c_str(),
                         (long long)conv_channels,
                         (long long)current_channels,
                         (long long)next_channels,
                         (int)multiplier);
                cfg.blocks.push_back({"compress_all_res", 0, static_cast<int>(multiplier)});
            }
            if (next_channels > 0) {
                current_channels = next_channels;
            } else if (current_channels > 0) {
                current_channels *= multiplier;
            }
        }

        if (cfg.blocks.empty()) {
            return get_default_encoder_config(version);
        }
        return cfg;
    }

    static inline int detect_ltx_vae_version(const String2TensorStorage& tensor_storage_map,
                                             const std::string& prefix) {
        const std::string v2_probe = prefix + ".encoder.down_blocks.1.conv.conv.bias";
        if (tensor_storage_map.find(v2_probe) != tensor_storage_map.end()) {
            return 2;
        }
        return 1;
    }

    static inline bool detect_ltx_vae_timestep_conditioning(const String2TensorStorage& tensor_storage_map,
                                                            const std::string& prefix) {
        return tensor_storage_map.find(prefix + ".decoder.timestep_scale_multiplier") != tensor_storage_map.end();
    }

    static inline EncoderConfig get_default_encoder_config(int version) {
        EncoderConfig cfg;
        if (version < 2) {
            GGML_ABORT("LTX VAE encoder is only implemented for version >= 2");
        }

        cfg.blocks = {
            {"res_x", 4, 1},
            {"compress_space_res", 0, 2},
            {"res_x", 6, 1},
            {"compress_time_res", 0, 2},
            {"res_x", 6, 1},
            {"compress_all_res", 0, 2},
            {"res_x", 2, 1},
            {"compress_all_res", 0, 2},
            {"res_x", 2, 1},
        };
        return cfg;
    }

    struct Encoder : public GGMLBlock {
        int version;
        int patch_size;
        int64_t in_channels;
        int64_t latent_channels;
        size_t causal_conv_count = 2;

        Encoder(int version,
                const String2TensorStorage& tensor_storage_map,
                const std::string& prefix,
                int patch_size          = 4,
                int64_t in_channels     = 3,
                int64_t latent_channels = 128)
            : version(version),
              patch_size(patch_size),
              in_channels(in_channels),
              latent_channels(latent_channels) {
            auto cfg         = infer_encoder_config_from_weights(tensor_storage_map, prefix, version);
            int64_t channels = get_tensor_ne0(tensor_storage_map,
                                              prefix + ".encoder.conv_in.conv.bias",
                                              0);
            GGML_ASSERT(channels > 0);
            int64_t in_dim = in_channels * patch_size * patch_size;

            // Temporal chunk shapes must not change Vulkan accumulation order.
            // Keep every encoder projection on the deterministic F32 route so
            // streamed and monolithic encodes are numerically equivalent.
            blocks["conv_in"] = std::make_shared<CausalConv3d>(
                in_dim, channels, 3, std::tuple<int, int, int>{1, 1, 1},
                1, true, true);

            for (int block_idx = 0; block_idx < static_cast<int>(cfg.blocks.size()); ++block_idx) {
                const auto& block = cfg.blocks[block_idx];
                if (block.type == "res_x") {
                    causal_conv_count += 2 * block.num_layers;
                    blocks["down_blocks." + std::to_string(block_idx)] = std::make_shared<UNetMidBlock3D>(channels,
                                                                                                          block.num_layers,
                                                                                                          false,
                                                                                                          true);
                } else if (block.type == "compress_space_res") {
                    causal_conv_count++;
                    int64_t next_channels                              = channels * block.multiplier;
                    blocks["down_blocks." + std::to_string(block_idx)] = std::make_shared<SpaceToDepthDownsample>(channels,
                                                                                                                  next_channels,
                                                                                                                  1,
                                                                                                                  2,
                                                                                                                  true);
                    channels                                           = next_channels;
                } else if (block.type == "compress_time_res") {
                    causal_conv_count++;
                    int64_t next_channels                              = channels * block.multiplier;
                    blocks["down_blocks." + std::to_string(block_idx)] = std::make_shared<SpaceToDepthDownsample>(channels,
                                                                                                                  next_channels,
                                                                                                                  2,
                                                                                                                  1,
                                                                                                                  true);
                    channels                                           = next_channels;
                } else if (block.type == "compress_all_res") {
                    causal_conv_count++;
                    int64_t next_channels = channels * block.multiplier;
                    blocks["down_blocks." + std::to_string(block_idx)] = std::make_shared<SpaceToDepthDownsample>(channels,
                                                                                                                  next_channels,
                                                                                                                  2,
                                                                                                                  2,
                                                                                                                  true);
                    channels                                           = next_channels;
                } else {
                    GGML_ABORT("Unsupported LTX VAE encoder block");
                }
            }

            blocks["conv_norm_out"] = std::make_shared<PixelNorm3D>();
            blocks["conv_out"]      = std::make_shared<CausalConv3d>(
                channels, latent_channels + 1, 3,
                std::tuple<int, int, int>{1, 1, 1}, 1, true, true);
        }

        size_t get_causal_conv_count() const {
            return causal_conv_count;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x) {
            auto conv_in       = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto conv_norm_out = std::dynamic_pointer_cast<PixelNorm3D>(blocks["conv_norm_out"]);
            auto conv_out      = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);

            x = conv_in->forward(ctx, x, true);

            int block_idx = 0;
            while (blocks.find("down_blocks." + std::to_string(block_idx)) != blocks.end()) {
                auto mid_block = std::dynamic_pointer_cast<UNetMidBlock3D>(blocks["down_blocks." + std::to_string(block_idx)]);
                if (mid_block) {
                    x = mid_block->forward(ctx, x, nullptr, true);
                } else {
                    auto downsample = std::dynamic_pointer_cast<SpaceToDepthDownsample>(blocks["down_blocks." + std::to_string(block_idx)]);
                    x               = downsample->forward(ctx, x, true);
                }
                block_idx++;
            }

            x = conv_norm_out->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = conv_out->forward(ctx, x, true);

            auto last_channel = ggml_ext_slice(ctx->ggml_ctx, x, 3, x->ne[3] - 1, x->ne[3]);
            auto repeat_shape = ggml_new_tensor_4d(ctx->ggml_ctx, last_channel->type, last_channel->ne[0], last_channel->ne[1], last_channel->ne[2], latent_channels - 1);
            auto repeated     = ggml_repeat(ctx->ggml_ctx, last_channel, repeat_shape);
            return ggml_concat(ctx->ggml_ctx, x, repeated, 3);
        }

        ggml_tensor* forward_chunk(GGMLRunnerContext* ctx,
                                   ggml_tensor* x,
                                   const EncoderTemporalPhase& phase,
                                   EncoderStreamingGraphState& state) {
            auto conv_in       = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto conv_norm_out = std::dynamic_pointer_cast<PixelNorm3D>(blocks["conv_norm_out"]);
            auto conv_out      = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);

            x = conv_in->forward_encoder_chunk(ctx, x, state);

            int block_idx = 0;
            while (blocks.find("down_blocks." + std::to_string(block_idx)) != blocks.end()) {
                auto mid_block = std::dynamic_pointer_cast<UNetMidBlock3D>(blocks["down_blocks." + std::to_string(block_idx)]);
                if (mid_block) {
                    x = mid_block->forward_encoder_chunk(ctx, x, state);
                } else {
                    auto downsample = std::dynamic_pointer_cast<SpaceToDepthDownsample>(blocks["down_blocks." + std::to_string(block_idx)]);
                    x               = downsample->forward_encoder_chunk(ctx, x, phase, state);
                    if (x == nullptr) {
                        return nullptr;
                    }
                }
                block_idx++;
            }

            x = conv_norm_out->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = conv_out->forward_encoder_chunk(ctx, x, state);

            auto last_channel = ggml_ext_slice(ctx->ggml_ctx, x, 3, x->ne[3] - 1, x->ne[3]);
            auto repeat_shape = ggml_new_tensor_4d(ctx->ggml_ctx,
                                                   last_channel->type,
                                                   last_channel->ne[0],
                                                   last_channel->ne[1],
                                                   last_channel->ne[2],
                                                   latent_channels - 1);
            auto repeated = ggml_repeat(ctx->ggml_ctx, last_channel, repeat_shape);
            return ggml_concat(ctx->ggml_ctx, x, repeated, 3);
        }
    };

    struct Decoder : public GGMLBlock {
        int version;
        int patch_size;
        bool causal_decoder;
        bool timestep_conditioning;
        int64_t in_channels;
        int64_t hidden_channels;

    protected:
        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            if (timestep_conditioning) {
                params["timestep_scale_multiplier"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                params["last_scale_shift_table"]    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_channels, 2);
            }
        }

    public:
        Decoder(int version,
                const String2TensorStorage& tensor_storage_map,
                const std::string& prefix,
                int patch_size             = 4,
                bool causal_decoder        = false,
                bool timestep_conditioning = true,
                int64_t in_channels        = 128,
                int64_t hidden_channels    = 128)
            : version(version),
              patch_size(patch_size),
              causal_decoder(causal_decoder),
              timestep_conditioning(timestep_conditioning),
              in_channels(in_channels),
              hidden_channels(hidden_channels) {
            const int64_t conv_in_out_channels = get_tensor_ne0(tensor_storage_map,
                                                                prefix + ".decoder.conv_in.conv.bias",
                                                                hidden_channels);
            auto cfg                           = infer_decoder_config_from_weights(tensor_storage_map,
                                                                                   prefix,
                                                                                   conv_in_out_channels);
            int64_t channels                   = conv_in_out_channels;

            blocks["conv_in"] = std::make_shared<CausalConv3d>(in_channels, channels, 3);

            for (int block_idx = 0; block_idx < static_cast<int>(cfg.blocks.size()); ++block_idx) {
                const auto& block = cfg.blocks[block_idx];
                if (block.type == "res_x") {
                    blocks["up_blocks." + std::to_string(block_idx)] = std::make_shared<UNetMidBlock3D>(channels,
                                                                                                        block.num_layers,
                                                                                                        timestep_conditioning);
                } else if (block.type == "compress_all") {
                    blocks["up_blocks." + std::to_string(block_idx)] = std::make_shared<DepthToSpaceUpsample>(channels,
                                                                                                              2,
                                                                                                              2,
                                                                                                              block.multiplier,
                                                                                                              false);
                    channels /= block.multiplier;
                } else if (block.type == "compress_time") {
                    blocks["up_blocks." + std::to_string(block_idx)] = std::make_shared<DepthToSpaceUpsample>(channels,
                                                                                                              2,
                                                                                                              1,
                                                                                                              block.multiplier,
                                                                                                              false);
                    channels /= block.multiplier;
                } else if (block.type == "compress_space") {
                    blocks["up_blocks." + std::to_string(block_idx)] = std::make_shared<DepthToSpaceUpsample>(channels,
                                                                                                              1,
                                                                                                              2,
                                                                                                              block.multiplier,
                                                                                                              false);
                    channels /= block.multiplier;
                } else {
                    GGML_ABORT("Unsupported LTX VAE decoder block");
                }
            }

            hidden_channels         = channels;
            blocks["conv_norm_out"] = std::make_shared<PixelNorm3D>();
            blocks["conv_out"]      = std::make_shared<CausalConv3d>(hidden_channels, 3 * patch_size * patch_size, 3);
            if (timestep_conditioning) {
                blocks["last_time_embedder"] = std::make_shared<PixArtAlphaCombinedTimestepSizeEmbeddings>(hidden_channels * 2);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep) {
            auto conv_in       = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto conv_norm_out = std::dynamic_pointer_cast<PixelNorm3D>(blocks["conv_norm_out"]);
            auto conv_out      = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);

            ggml_tensor* scaled_timestep = timestep;
            if (timestep_conditioning) {
                auto multiplier = ggml_ext_cast_f32(ctx->ggml_ctx, ctx->backend, params["timestep_scale_multiplier"]);
                scaled_timestep = ggml_mul(ctx->ggml_ctx, timestep, multiplier);
            }

            x = conv_in->forward(ctx, x, causal_decoder);

            int block_idx = 0;
            while (blocks.find("up_blocks." + std::to_string(block_idx)) != blocks.end()) {
                auto mid_block = std::dynamic_pointer_cast<UNetMidBlock3D>(blocks["up_blocks." + std::to_string(block_idx)]);
                if (mid_block) {
                    x = mid_block->forward(ctx, x, scaled_timestep, causal_decoder);
                } else {
                    auto upsample = std::dynamic_pointer_cast<DepthToSpaceUpsample>(blocks["up_blocks." + std::to_string(block_idx)]);
                    x             = upsample->forward(ctx, x, causal_decoder);
                }
                block_idx++;
            }

            x = conv_norm_out->forward(ctx, x);
            if (timestep_conditioning) {
                auto last_time_embedder = std::dynamic_pointer_cast<PixArtAlphaCombinedTimestepSizeEmbeddings>(blocks["last_time_embedder"]);
                auto timestep_embed     = last_time_embedder->forward(ctx, scaled_timestep);
                auto [shift, scale]     = get_shift_scale(ctx->ggml_ctx,
                                                          params["last_scale_shift_table"],
                                                          timestep_embed,
                                                          hidden_channels,
                                                          2);
                x                       = apply_scale_shift(ctx->ggml_ctx, x, scale, shift);
            }
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = conv_out->forward(ctx, x, causal_decoder);
            return x;
        }

        // Process a single latent frame through the complete decoder (conv_in → up_blocks
        // → final layers), using feat_map to carry per-layer causal context from the
        // previous frame.  Designed for tiled temporal decode: each iteration receives
        // 1 latent frame so that intermediate tensors can be freed between iterations.
        ggml_tensor* forward_tiled_frame(GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* timestep,
                                         std::vector<ggml_tensor*>& feat_map,
                                         int& feat_idx,
                                         int chunk_idx,
                                         int& temporal_pad) {
            auto conv_in       = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto conv_norm_out = std::dynamic_pointer_cast<PixelNorm3D>(blocks["conv_norm_out"]);
            auto conv_out      = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);

            ggml_tensor* scaled_timestep = timestep;
            if (timestep_conditioning && timestep != nullptr) {
                auto multiplier = ggml_ext_cast_f32(ctx->ggml_ctx, ctx->backend, params["timestep_scale_multiplier"]);
                scaled_timestep = ggml_mul(ctx->ggml_ctx, timestep, multiplier);
            }

            // conv_in with feat_map for left temporal context
            x = conv_in->forward(ctx, x, feat_map, feat_idx, chunk_idx, causal_decoder, temporal_pad);

            // up_blocks
            int block_idx = 0;
            while (blocks.find("up_blocks." + std::to_string(block_idx)) != blocks.end()) {
                auto mid_block = std::dynamic_pointer_cast<UNetMidBlock3D>(blocks["up_blocks." + std::to_string(block_idx)]);
                if (mid_block) {
                    x = mid_block->forward(ctx, x, scaled_timestep, causal_decoder,
                                           feat_map, feat_idx, chunk_idx, temporal_pad);
                } else {
                    auto upsample = std::dynamic_pointer_cast<DepthToSpaceUpsample>(
                        blocks["up_blocks." + std::to_string(block_idx)]);
                    x = upsample->forward(ctx, x, causal_decoder,
                                          feat_map, feat_idx, chunk_idx, temporal_pad);
                    temporal_pad *= upsample->factor_t;
                }
                block_idx++;
            }

            x = conv_norm_out->forward(ctx, x);
            if (timestep_conditioning) {
                auto last_time_embedder = std::dynamic_pointer_cast<PixArtAlphaCombinedTimestepSizeEmbeddings>(blocks["last_time_embedder"]);
                auto timestep_embed     = last_time_embedder->forward(ctx, scaled_timestep);
                auto [shift, scale]     = get_shift_scale(ctx->ggml_ctx,
                                                          params["last_scale_shift_table"],
                                                          timestep_embed,
                                                          hidden_channels,
                                                          2);
                x                       = apply_scale_shift(ctx->ggml_ctx, x, scale, shift);
            }
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = conv_out->forward(ctx, x, feat_map, feat_idx, chunk_idx, causal_decoder, temporal_pad);
            return x;
        }
    };

    struct VideoVAE : public GGMLBlock {
        int version;
        float decode_timestep;
        bool timestep_conditioning;
        int patch_size;
        bool decode_only;

        VideoVAE(int version,
                 bool decode_only,
                 bool timestep_conditioning,
                 int patch_size,
                 const String2TensorStorage& tensor_storage_map,
                 const std::string& prefix,
                 float decode_timestep = 0.05f)
            : version(version),
              decode_timestep(decode_timestep),
              timestep_conditioning(timestep_conditioning),
              patch_size(patch_size),
              decode_only(decode_only) {
            if (!decode_only) {
                blocks["encoder"] = std::make_shared<Encoder>(version,
                                                              tensor_storage_map,
                                                              prefix,
                                                              patch_size);
            }
            blocks["decoder"]                = std::make_shared<Decoder>(version,
                                                          tensor_storage_map,
                                                          prefix,
                                                          patch_size,
                                                          false,
                                                          timestep_conditioning);
            blocks["per_channel_statistics"] = std::make_shared<PerChannelStatistics>();
        }

        size_t get_encoder_causal_conv_count() const {
            auto encoder = std::dynamic_pointer_cast<Encoder>(blocks.at("encoder"));
            GGML_ASSERT(encoder != nullptr);
            return encoder->get_causal_conv_count();
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx,
                            ggml_tensor* z,
                            ggml_tensor* timestep) {
            auto decoder   = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);
            auto processor = std::dynamic_pointer_cast<PerChannelStatistics>(blocks["per_channel_statistics"]);
            auto latents   = processor->un_normalize(ctx, z);
            auto out       = decoder->forward(ctx, latents, timestep);
            out            = WAN::WanVAE::unpatchify(ctx->ggml_ctx, out, patch_size, 1);
            return out;
        }

        // Tiled temporal decode: each latent frame is processed through the COMPLETE
        // decoder individually.  Per-layer causal context is passed via feat_map
        // (contiguous copies, not views) so that each iteration's large intermediate
        // tensors can be freed by GGML before the next iteration starts.
        ggml_tensor* decode_tiled(GGMLRunnerContext* ctx,
                                  ggml_tensor* z,
                                  ggml_tensor* timestep,
                                  int temporal_window_size  = 1,
                                  int temporal_tile_overlap = 0) {
            auto decoder   = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);
            auto processor = std::dynamic_pointer_cast<PerChannelStatistics>(blocks["per_channel_statistics"]);
            auto latents   = processor->un_normalize(ctx, z);

            const int64_t T = z->ne[2];
            if (T <= 1) {
                auto out = decoder->forward(ctx, latents, timestep);
                return WAN::WanVAE::unpatchify(ctx->ggml_ctx, out, patch_size, 1);
            }

            // feat_map holds ggml_tensor* nodes (contiguous copies at each conv layer).
            // 128 slots is generous enough for any supported decoder configuration.
            std::vector<ggml_tensor*> feat_map(128, nullptr);

            // Ensure window size is at least 1
            int window  = std::max(1, temporal_window_size);
            int overlap = std::max(0, temporal_tile_overlap);

            if (overlap >= window) {
                LOG_WARN("temporal_tile_overlap (%d) is greater than or equal to temporal_tile_frames (%d), adjusting values to avoid empty decode windows",
                         overlap, window);
                overlap = window - 1;
            }
            LOG_DEBUG("Using temporal tiling: temporal_tile_frames = %d, temporal_tile_overlap = %d, total frames = %d, resulting in %d tiles",
                      window,
                      overlap,
                      (int)T,
                      (T + window - overlap - 1) / (window - overlap));
            ggml_tensor* out = nullptr;
            for (int i = 0; i < (int)T - overlap; i += (window - overlap)) {
                int feat_idx = 0;

                // Calculate the end index for the current temporal chunk
                int end_i = std::min((int)T, i + window);
                if (end_i >= (int)T) {
                    overlap = 0;  // avoid overlap issues in the last chunk
                }

                int chunk_overlap = overlap;  // modified by forward_tiled_frame temporal inflation

                auto z_chunk = ggml_ext_slice(ctx->ggml_ctx, latents, 2, i, end_i);

                auto out_chunk = decoder->forward_tiled_frame(ctx, z_chunk, timestep,
                                                              feat_map, feat_idx, i, chunk_overlap);

                // discard overlap frames if it's not the final chunk
                if (overlap > 0 && end_i < (int)T) {
                    out_chunk = ggml_ext_slice(ctx->ggml_ctx, out_chunk, 2, 0, out_chunk->ne[2] - chunk_overlap);
                }

                out = (out == nullptr) ? out_chunk : ggml_concat(ctx->ggml_ctx, out, out_chunk, 2);
            }

            return WAN::WanVAE::unpatchify(ctx->ggml_ctx, out, patch_size, 1);
        }

        ggml_tensor* decode_tiled_chunk(GGMLRunnerContext* ctx,
                                        ggml_tensor* z,
                                        ggml_tensor* timestep,
                                        std::vector<ggml_tensor*>& feat_map,
                                        int chunk_idx,
                                        int temporal_tile_overlap,
                                        int& feat_idx) {
            auto decoder   = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);
            auto processor = std::dynamic_pointer_cast<PerChannelStatistics>(blocks["per_channel_statistics"]);
            auto latents   = processor->un_normalize(ctx, z);

            feat_idx          = 0;
            int chunk_overlap = temporal_tile_overlap;  // modified by forward_tiled_frame temporal inflation
            auto out_chunk    = decoder->forward_tiled_frame(ctx, latents, timestep,
                                                             feat_map, feat_idx, chunk_idx, chunk_overlap);
            if (chunk_overlap > 0) {
                out_chunk = ggml_ext_slice(ctx->ggml_ctx, out_chunk, 2, 0, out_chunk->ne[2] - chunk_overlap);
            }
            return WAN::WanVAE::unpatchify(ctx->ggml_ctx, out_chunk, patch_size, 1);
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx,
                            ggml_tensor* x) {
            GGML_ASSERT(!decode_only);
            auto encoder   = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);
            auto processor = std::dynamic_pointer_cast<PerChannelStatistics>(blocks["per_channel_statistics"]);

            x         = patchify(ctx->ggml_ctx, x, patch_size);
            auto out  = encoder->forward(ctx, x);
            auto mean = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3, false)[0];
            mean      = ggml_cont(ctx->ggml_ctx, mean);
            return processor->normalize(ctx, mean);
        }

        ggml_tensor* encode_chunk(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  const EncoderTemporalPhase& phase,
                                  EncoderStreamingGraphState& state) {
            GGML_ASSERT(!decode_only);
            auto encoder   = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);
            auto processor = std::dynamic_pointer_cast<PerChannelStatistics>(blocks["per_channel_statistics"]);

            x        = patchify(ctx->ggml_ctx, x, patch_size);
            auto out = encoder->forward_chunk(ctx, x, phase, state);
            if (out == nullptr) {
                return nullptr;
            }
            auto mean = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3, false)[0];
            mean      = ggml_cont(ctx->ggml_ctx, mean);
            return processor->normalize(ctx, mean);
        }

        ggml_tensor* normalize_latents(GGMLRunnerContext* ctx,
                                       ggml_tensor* x) {
            auto processor = std::dynamic_pointer_cast<PerChannelStatistics>(blocks["per_channel_statistics"]);
            return processor->normalize(ctx, x);
        }

        ggml_tensor* un_normalize_latents(GGMLRunnerContext* ctx,
                                          ggml_tensor* x) {
            auto processor = std::dynamic_pointer_cast<PerChannelStatistics>(blocks["per_channel_statistics"]);
            return processor->un_normalize(ctx, x);
        }
    };

}  // namespace LTXVAE

struct LTXVideoVAE : public VAE {
    static constexpr int DEFAULT_TEMPORAL_TILE_FRAMES  = 4;
    static constexpr int DEFAULT_TEMPORAL_TILE_OVERLAP = 1;
    static constexpr int DEFAULT_ENCODER_CHUNK_FRAMES  = 49;
    static constexpr int MIN_ENCODER_CHUNK_FRAMES      = 9;

    bool decode_only;
    bool temporal_tiling_enabled = false;
    int temporal_tile_frames     = DEFAULT_TEMPORAL_TILE_FRAMES;
    int temporal_tile_overlap    = DEFAULT_TEMPORAL_TILE_OVERLAP;
    int encoder_chunk_frames     = DEFAULT_ENCODER_CHUNK_FRAMES;
    bool encoder_chunk_frames_configured = false;
    int ltx_vae_version;
    bool timestep_conditioning;
    int patch_size;
    sd::Tensor<float> decode_timestep_tensor;
    LTXVAE::VideoVAE vae;

    LTXVideoVAE(ggml_backend_t backend,
                const String2TensorStorage& tensor_storage_map,
                const std::string& prefix,
                bool decode_only                                    = true,
                SDVersion version                                   = VERSION_LTXAV,
                std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : decode_only(decode_only),
          ltx_vae_version(LTXVAE::detect_ltx_vae_version(tensor_storage_map, prefix)),
          timestep_conditioning(LTXVAE::detect_ltx_vae_timestep_conditioning(tensor_storage_map, prefix)),
          patch_size(4),
          decode_timestep_tensor(sd::Tensor<float>::from_vector({0.05f})),
          vae(LTXVAE::detect_ltx_vae_version(tensor_storage_map, prefix),
              decode_only,
              LTXVAE::detect_ltx_vae_timestep_conditioning(tensor_storage_map, prefix),
              patch_size,
              tensor_storage_map,
              prefix),
          VAE(version, backend, prefix, weight_manager) {
        vae.init(params_ctx, tensor_storage_map, prefix);
        decode_timestep_tensor.values()[0] = vae.decode_timestep;
    }

    std::string get_desc() override {
        return "ltx_video_vae";
    }

    void set_temporal_tiling_enabled(bool enabled) override {
        temporal_tiling_enabled = enabled;
    }

    void set_tiling_params(const sd_tiling_params_t& params) override {
        temporal_tiling_enabled = params.temporal_tiling;
        temporal_tile_frames    = DEFAULT_TEMPORAL_TILE_FRAMES;
        temporal_tile_overlap   = DEFAULT_TEMPORAL_TILE_OVERLAP;
        encoder_chunk_frames    = DEFAULT_ENCODER_CHUNK_FRAMES;
        encoder_chunk_frames_configured = false;

        for (const auto& [key, value] : parse_key_value_args(params.extra_tiling_args, "LTX VAE extra tiling arg")) {
            int parsed = 0;
            if (!parse_strict_int(value, parsed)) {
                LOG_WARN("ignoring invalid LTX VAE extra tiling arg '%s=%s'", key.c_str(), value.c_str());
            } else if (key == "temporal_tile_frames") {
                temporal_tile_frames = std::max(1, parsed);
            } else if (key == "temporal_tile_overlap") {
                temporal_tile_overlap = std::max(0, parsed);
            } else if (key == "encoder_chunk_frames") {
                encoder_chunk_frames = std::max(MIN_ENCODER_CHUNK_FRAMES, parsed);
                encoder_chunk_frames_configured = true;
                if (parsed < MIN_ENCODER_CHUNK_FRAMES) {
                    LOG_WARN("encoder_chunk_frames=%d is below the minimum exact production "
                             "payload; using %d",
                             parsed,
                             MIN_ENCODER_CHUNK_FRAMES);
                }
            } else {
                LOG_WARN("ignoring unknown LTX VAE extra tiling arg '%s'", key.c_str());
            }
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        vae.get_param_tensors(tensors, weight_prefix);
    }

    struct TemporalTilePlan {
        int frames    = 1;
        int overlap   = 0;
        int stride    = 1;
        int num_tiles = 1;
    };

    TemporalTilePlan resolve_temporal_tile_plan(int64_t total_frames) const {
        TemporalTilePlan plan;
        plan.frames  = std::max(1, temporal_tile_frames);
        plan.overlap = std::max(0, temporal_tile_overlap);

        if (plan.overlap >= plan.frames) {
            LOG_WARN("temporal_tile_overlap (%d) is greater than or equal to temporal_tile_frames (%d), adjusting values to avoid empty decode windows",
                     plan.overlap,
                     plan.frames);
            plan.overlap = plan.frames - 1;
        }
        if (total_frames > 1 && plan.overlap >= total_frames) {
            LOG_WARN("temporal_tile_overlap (%d) is greater than or equal to total latent frames (%lld), adjusting values to decode at least one tile",
                     plan.overlap,
                     (long long)total_frames);
            plan.overlap = static_cast<int>(total_frames - 1);
        }

        plan.stride          = std::max(1, plan.frames - plan.overlap);
        int64_t tiled_frames = std::max<int64_t>(1, total_frames - plan.overlap);
        plan.num_tiles       = total_frames > 0 ? static_cast<int>((tiled_frames + plan.stride - 1) / plan.stride) : 0;
        return plan;
    }

    std::string temporal_feat_cache_name(size_t feat_idx) const {
        return "ltx_vae_temporal_feat:" + std::to_string(feat_idx);
    }

    std::string encoder_conv_cache_name(size_t index) const {
        return "ltx_vae_encoder_conv_history:" + std::to_string(index);
    }

    std::string encoder_pending_cache_name(int level, bool conv_branch) const {
        return std::string("ltx_vae_encoder_pending_") +
               (conv_branch ? "h:" : "x:") + std::to_string(level);
    }

    struct EncoderStreamingPlan {
        LTXVAE::EncoderChunkPlan temporal;
        size_t max_compute_bytes = 0;
        size_t max_cache_bytes   = 0;
        size_t runtime_param_bytes = 0;
        size_t capacity_bytes    = std::numeric_limits<size_t>::max();
    };

    ggml_cgraph* build_encoder_chunk_graph(const sd::Tensor<float>& x_chunk,
                                           const LTXVAE::EncoderTemporalPhase& phase,
                                           bool preflight,
                                           bool* exact_out,
                                           bool* has_output_out,
                                           size_t* state_bytes_out) {
        ggml_cgraph* gf = new_graph_custom(20480);
        ggml_tensor* x  = make_input(x_chunk);

        LTXVAE::EncoderStreamingGraphState state;
        const size_t expected_conv_count = vae.get_encoder_causal_conv_count();
        state.allow_missing_history = phase.first_frame[0];
        state.preflight             = preflight;
        if (!preflight) {
            state.previous_conv_history.resize(
                expected_conv_count,
                nullptr);
            for (size_t i = 0; i < state.previous_conv_history.size(); ++i) {
                state.previous_conv_history[i] = get_cache_tensor_by_name(encoder_conv_cache_name(i));
            }
            for (int level = 0; level < LTXVAE::LTX_ENCODER_TEMPORAL_LEVELS; ++level) {
                state.previous_pending_x[level] = get_cache_tensor_by_name(
                    encoder_pending_cache_name(level, false));
                state.previous_pending_h[level] = get_cache_tensor_by_name(
                    encoder_pending_cache_name(level, true));
            }
        }

        auto runner_ctx  = get_context();
        ggml_tensor* out = vae.encode_chunk(&runner_ctx, x, phase, state);
        bool exact       = state.exact;
        const bool has_output = out != nullptr;
        if (has_output != (phase.output_frames > 0)) {
            exact = false;
        }

        int expected_temporal_levels = 0;
        for (int level = 0; level < LTXVAE::LTX_ENCODER_TEMPORAL_LEVELS; ++level) {
            if (phase.level_input_frames[level] > 0) {
                expected_temporal_levels++;
            }
        }
        if (state.temporal_level != expected_temporal_levels) {
            exact = false;
        }
        if (has_output &&
            state.conv_index != expected_conv_count) {
            exact = false;
        }

        ggml_backend_buffer_type_t cache_buft =
            ggml_backend_get_default_buffer_type(runtime_backend);
        const size_t cache_alignment =
            ggml_backend_buft_get_alignment(cache_buft);
        auto cache_alloc_size = [&](ggml_tensor* tensor) {
            const size_t alloc_size =
                ggml_backend_buft_get_alloc_size(cache_buft, tensor);
            return cache_alignment > 0
                       ? ((alloc_size + cache_alignment - 1) / cache_alignment) *
                             cache_alignment
                       : alloc_size;
        };

        size_t state_bytes = 0;
        for (size_t i = 0; i < state.next_conv_history.size(); ++i) {
            ggml_tensor* history = state.next_conv_history[i];
            if (history == nullptr) {
                exact = false;
                continue;
            }
            state_bytes += cache_alloc_size(history);
            if (!preflight) {
                cache(encoder_conv_cache_name(i), history);
            }
            ggml_build_forward_expand(gf, history);
        }
        for (int level = 0; level < LTXVAE::LTX_ENCODER_TEMPORAL_LEVELS; ++level) {
            ggml_tensor* pending_x = state.next_pending_x[level];
            ggml_tensor* pending_h = state.next_pending_h[level];
            if (pending_x == nullptr || pending_h == nullptr) {
                continue;
            }
            state_bytes += cache_alloc_size(pending_x) +
                           cache_alloc_size(pending_h);
            if (!preflight) {
                cache(encoder_pending_cache_name(level, false), pending_x);
                cache(encoder_pending_cache_name(level, true), pending_h);
            }
            ggml_build_forward_expand(gf, pending_x);
            ggml_build_forward_expand(gf, pending_h);
        }
        if (out != nullptr) {
            ggml_build_forward_expand(gf, out);
        }

        if (exact_out != nullptr) {
            *exact_out = exact;
        }
        if (has_output_out != nullptr) {
            *has_output_out = has_output;
        }
        if (state_bytes_out != nullptr) {
            *state_bytes_out = state_bytes;
        }
        if (!exact) {
            LOG_WARN("unable to build exact LTX VAE encoder chunk graph: "
                     "input=%lld output=%lld has_output=%d conv=%zu/%d "
                     "history=%zu temporal_levels=%d/%d state_exact=%d",
                     (long long)phase.input_frames,
                     (long long)phase.output_frames,
                     has_output ? 1 : 0,
                     state.conv_index,
                     (int)expected_conv_count,
                     state.next_conv_history.size(),
                     state.temporal_level,
                     expected_temporal_levels,
                     state.exact ? 1 : 0);
        }
        return gf;
    }

    bool measure_encoder_chunk(const sd::Tensor<float>& x_chunk,
                               const LTXVAE::EncoderTemporalPhase& phase,
                               size_t* compute_bytes,
                               size_t* state_bytes) {
        bool exact      = false;
        bool has_output = false;
        size_t graph_state_bytes = 0;
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_encoder_chunk_graph(x_chunk,
                                             phase,
                                             true,
                                             &exact,
                                             &has_output,
                                             &graph_state_bytes);
        };

        ggml_cgraph* gf = nullptr;
        if (!prepare_compute_graph(get_graph, &gf)) {
            return false;
        }
        const size_t required = measure_compute_buffer_size(gf);
        free_compute_ctx();

        if (!exact || has_output != (phase.output_frames > 0)) {
            return false;
        }
        if (compute_bytes != nullptr) {
            *compute_bytes = required;
        }
        if (state_bytes != nullptr) {
            *state_bytes = graph_state_bytes;
        }
        return true;
    }

    bool measure_full_graph(const sd::Tensor<float>& input,
                            size_t* compute_bytes) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(input, false);
        };
        ggml_cgraph* gf = nullptr;
        if (!prepare_compute_graph(get_graph, &gf)) {
            return false;
        }
        const size_t required = measure_compute_buffer_size(gf);
        free_compute_ctx();
        if (compute_bytes != nullptr) {
            *compute_bytes = required;
        }
        return true;
    }

    bool resolve_encoder_streaming_plan(const sd::Tensor<float>& input,
                                        bool force_chunking,
                                        EncoderStreamingPlan* selected) {
        GGML_ASSERT(selected != nullptr);
        const int64_t total_frames = input.shape()[2];
        const sd::VaeFallbackCapacity capacity = get_vae_fallback_capacity();
        const size_t budget = sd::vae_fallback_budget(capacity);
        const size_t runtime_param_bytes =
            params_backend == runtime_backend ? 0 : get_params_buffer_size();

        size_t full_required = 0;
        if (!measure_full_graph(input, &full_required)) {
            return false;
        }

        const bool capacity_driven = full_required > budget;
        if (!force_chunking && !capacity_driven) {
            return false;
        }
        if (!force_chunking && !sd_backend_is(runtime_backend, "Vulkan")) {
            return false;
        }

        int max_candidate = static_cast<int>(
            std::min<int64_t>(std::max(MIN_ENCODER_CHUNK_FRAMES,
                                       encoder_chunk_frames),
                              total_frames));
        max_candidate = ((max_candidate - 1) / 8) * 8 + 1;
        for (int candidate = max_candidate;
             candidate >= MIN_ENCODER_CHUNK_FRAMES;
             candidate -= 8) {
            auto temporal = LTXVAE::make_encoder_chunk_plan(total_frames, candidate);
            if (!temporal.exact) {
                continue;
            }

            size_t max_compute = 0;
            size_t max_cache   = 0;
            bool fits          = true;
            int64_t start      = 0;
            std::set<std::array<int64_t, 17>> measured_phases;
            for (const auto& phase : temporal.chunks) {
                std::array<int64_t, 17> phase_key = {
                    phase.input_frames,
                    phase.output_frames,
                    phase.level_input_frames[0],
                    phase.level_input_frames[1],
                    phase.level_input_frames[2],
                    phase.level_output_frames[0],
                    phase.level_output_frames[1],
                    phase.level_output_frames[2],
                    phase.first_frame[0],
                    phase.first_frame[1],
                    phase.first_frame[2],
                    phase.pending_before[0],
                    phase.pending_before[1],
                    phase.pending_before[2],
                    phase.pending_after[0],
                    phase.pending_after[1],
                    phase.pending_after[2],
                };
                if (!measured_phases.insert(phase_key).second) {
                    start += phase.input_frames;
                    continue;
                }
                auto x_chunk = sd::ops::slice(input,
                                              2,
                                              start,
                                              start + phase.input_frames);
                size_t required = 0;
                size_t state_bytes = 0;
                if (!measure_encoder_chunk(x_chunk, phase, &required, &state_bytes)) {
                    fits = false;
                    break;
                }
                max_compute = std::max(max_compute, required);
                max_cache   = std::max(max_cache, state_bytes);

                const bool logical_fits =
                    required <= capacity.max_buffer_bytes &&
                    state_bytes <= capacity.max_buffer_bytes;
                const size_t free_budget = capacity.free_memory_bytes > 0
                                               ? sd::vae_fallback_scaled_budget(
                                                     capacity.free_memory_bytes,
                                                     capacity.free_memory_ratio)
                                               : std::numeric_limits<size_t>::max();
                const bool memory_fits =
                    runtime_param_bytes <= free_budget &&
                    state_bytes <= free_budget - runtime_param_bytes &&
                    required <= free_budget - runtime_param_bytes - state_bytes;
                if (!logical_fits || !memory_fits) {
                    fits = false;
                    break;
                }
                start += phase.input_frames;
            }

            if (!fits) {
                continue;
            }
            selected->temporal         = std::move(temporal);
            selected->max_compute_bytes = max_compute;
            selected->max_cache_bytes   = max_cache;
            selected->runtime_param_bytes = runtime_param_bytes;
            selected->capacity_bytes    = budget;
            return true;
        }
        return false;
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
        ggml_cgraph* gf       = new_graph_custom(20480);
        ggml_tensor* z        = make_input(z_tensor);
        ggml_tensor* timestep = nullptr;
        if (timestep_conditioning) {
            timestep = make_input(decode_timestep_tensor);
        }

        auto runner_ctx = get_context();
        ggml_tensor* out;
        out = decode_graph ? vae.decode(&runner_ctx, z, timestep) : vae.encode(&runner_ctx, z);
        ggml_build_forward_expand(gf, out);

        return gf;
    }

    ggml_cgraph* build_temporal_tile_graph(const sd::Tensor<float>& z_chunk_tensor,
                                           int chunk_idx,
                                           int chunk_overlap) {
        ggml_cgraph* gf       = new_graph_custom(20480);
        ggml_tensor* z        = make_input(z_chunk_tensor);
        ggml_tensor* timestep = nullptr;
        if (timestep_conditioning) {
            timestep = make_input(decode_timestep_tensor);
        }

        std::vector<ggml_tensor*> feat_map(128, nullptr);
        for (size_t feat_idx = 0; feat_idx < feat_map.size(); ++feat_idx) {
            feat_map[feat_idx] = get_cache_tensor_by_name(temporal_feat_cache_name(feat_idx));
        }

        auto runner_ctx  = get_context();
        int feat_count   = 0;
        ggml_tensor* out = vae.decode_tiled_chunk(&runner_ctx,
                                                  z,
                                                  timestep,
                                                  feat_map,
                                                  chunk_idx,
                                                  chunk_overlap,
                                                  feat_count);

        for (int feat_idx = 0; feat_idx < feat_count && feat_idx < static_cast<int>(feat_map.size()); ++feat_idx) {
            ggml_tensor* feat_cache = feat_map[static_cast<size_t>(feat_idx)];
            if (feat_cache != nullptr) {
                cache(temporal_feat_cache_name(static_cast<size_t>(feat_idx)), feat_cache);
                ggml_build_forward_expand(gf, feat_cache);
            }
        }

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> decode_temporal_tiled_streaming(const int n_threads,
                                                      const sd::Tensor<float>& input,
                                                      size_t expected_dim) {
        const int64_t total_frames = input.shape()[2];
        TemporalTilePlan plan      = resolve_temporal_tile_plan(total_frames);

        LOG_DEBUG("Using streaming temporal tiling: temporal_tile_frames=%d, temporal_tile_overlap=%d, total latent frames=%lld, resulting in %d tiles",
                  plan.frames,
                  plan.overlap,
                  (long long)total_frames,
                  plan.num_tiles);

        free_cache_ctx_and_buffer();
        cache_tensor_map.clear();

        sd::Tensor<float> output;
        for (int64_t start = 0; start < total_frames - plan.overlap; start += plan.stride) {
            const int64_t end       = std::min<int64_t>(total_frames, start + plan.frames);
            const int chunk_overlap = end < total_frames ? plan.overlap : 0;
            auto z_chunk            = sd::ops::slice(input, 2, start, end);

            LOG_DEBUG("LTX VAE temporal tile %lld/%d: latent frames [%lld, %lld), overlap=%d",
                      (long long)(start / plan.stride + 1),
                      plan.num_tiles,
                      (long long)start,
                      (long long)end,
                      chunk_overlap);

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_temporal_tile_graph(z_chunk,
                                                 static_cast<int>(start),
                                                 chunk_overlap);
            };
            auto chunk = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, true, true, true),
                                                         expected_dim);
            if (chunk.empty()) {
                free_cache_ctx_and_buffer();
                cache_tensor_map.clear();
                return {};
            }
            output = output.empty() ? std::move(chunk) : sd::ops::concat(output, chunk, 2);
        }

        free_cache_ctx_and_buffer();
        cache_tensor_map.clear();
        return output;
    }

    sd::Tensor<float> encode_temporal_chunked_streaming(
        const int n_threads,
        const sd::Tensor<float>& input,
        size_t expected_dim,
        const EncoderStreamingPlan& plan) {
        LOG_INFO("Using exact LTX VAE encoder streaming: first_payload=%d, "
                 "continuation_payload=%d, total_frames=%lld, chunks=%zu, "
                 "max_compute=%.2f MiB, state=%.2f MiB, runtime_params=%.2f MiB, "
                 "budget=%.2f MiB (%zu bytes)",
                 plan.temporal.first_chunk_frames,
                 plan.temporal.continuation_frames,
                 (long long)plan.temporal.total_frames,
                 plan.temporal.chunks.size(),
                 plan.max_compute_bytes / (1024.0 * 1024.0),
                 plan.max_cache_bytes / (1024.0 * 1024.0),
                 plan.runtime_param_bytes / (1024.0 * 1024.0),
                 plan.capacity_bytes == std::numeric_limits<size_t>::max()
                     ? 0.0
                     : plan.capacity_bytes / (1024.0 * 1024.0),
                 plan.capacity_bytes);

        free_cache_ctx_and_buffer();
        cache_tensor_map.clear();

        sd::Tensor<float> output;
        int64_t start = 0;
        bool failed   = false;
        try {
            for (size_t chunk_index = 0;
                 chunk_index < plan.temporal.chunks.size();
                 ++chunk_index) {
                const auto& phase = plan.temporal.chunks[chunk_index];
                auto x_chunk      = sd::ops::slice(input,
                                              2,
                                              start,
                                              start + phase.input_frames);
                bool exact        = false;
                bool has_output   = false;
                auto get_graph = [&]() -> ggml_cgraph* {
                    return build_encoder_chunk_graph(x_chunk,
                                                     phase,
                                                     false,
                                                     &exact,
                                                     &has_output,
                                                     nullptr);
                };

                auto result = GGMLRunner::compute<float>(
                    get_graph,
                    n_threads,
                    true,
                    phase.output_frames == 0);
                if (!result.has_value() || !exact ||
                    has_output != (phase.output_frames > 0)) {
                    failed = true;
                    break;
                }

                if (phase.output_frames > 0) {
                    auto chunk = restore_trailing_singleton_dims(
                        std::move(*result),
                        expected_dim);
                    if (chunk.empty() || chunk.shape()[2] != phase.output_frames) {
                        failed = true;
                        break;
                    }
                    output = output.empty()
                                 ? std::move(chunk)
                                 : sd::ops::concat(output, chunk, 2);
                }

                LOG_INFO("LTX VAE encoder chunk %zu/%zu: input=[%lld,%lld), "
                         "levels=%lld/%lld/%lld -> %lld/%lld/%lld, output=%lld, "
                         "first=%d/%d/%d, pending=%d/%d/%d -> %d/%d/%d",
                         chunk_index + 1,
                         plan.temporal.chunks.size(),
                         (long long)start,
                         (long long)(start + phase.input_frames),
                         (long long)phase.level_input_frames[0],
                         (long long)phase.level_input_frames[1],
                         (long long)phase.level_input_frames[2],
                         (long long)phase.level_output_frames[0],
                         (long long)phase.level_output_frames[1],
                         (long long)phase.level_output_frames[2],
                         (long long)phase.output_frames,
                         phase.first_frame[0] ? 1 : 0,
                         phase.first_frame[1] ? 1 : 0,
                         phase.first_frame[2] ? 1 : 0,
                         phase.pending_before[0] ? 1 : 0,
                         phase.pending_before[1] ? 1 : 0,
                         phase.pending_before[2] ? 1 : 0,
                         phase.pending_after[0] ? 1 : 0,
                         phase.pending_after[1] ? 1 : 0,
                         phase.pending_after[2] ? 1 : 0);
                start += phase.input_frames;
            }
        } catch (const std::exception& error) {
            LOG_ERROR("exact LTX VAE encoder stream aborted: %s", error.what());
            free_cache_ctx_and_buffer();
            cache_tensor_map.clear();
            return {};
        } catch (...) {
            LOG_ERROR("exact LTX VAE encoder stream aborted by an unknown error");
            free_cache_ctx_and_buffer();
            cache_tensor_map.clear();
            return {};
        }

        free_cache_ctx_and_buffer();
        cache_tensor_map.clear();

        const int64_t expected_frames =
            LTXVAE::ltx_encoder_temporal_output_frames(input.shape()[2]);
        if (failed || start != input.shape()[2] || output.empty() ||
            output.shape()[2] != expected_frames) {
            return {};
        }
        return output;
    }

    ggml_cgraph* build_latent_statistics_graph(const sd::Tensor<float>& z_tensor, bool normalize) {
        ggml_cgraph* gf = new_graph_custom(1024);
        ggml_tensor* z  = make_input(z_tensor);

        auto runner_ctx  = get_context();
        ggml_tensor* out = normalize ? vae.normalize_latents(&runner_ctx, z)
                                     : vae.un_normalize_latents(&runner_ctx, z);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z,
                               bool decode_graph) override {
        if (!decode_graph && decode_only) {
            LOG_ERROR("LTX video VAE encode requires encoder weights");
            return {};
        }
        sd::Tensor<float> input = z;
        size_t expected_dim     = static_cast<size_t>(z.dim());
        if (!decode_graph) {
            if (input.dim() == 4) {
                input        = input.unsqueeze(2);
                expected_dim = 5;
            } else if (input.dim() != 5) {
                LOG_ERROR("LTX video VAE encoder expects 4D image or 5D video input, got dim=%lld",
                          (long long)input.dim());
                return {};
            }

            int64_t cropped_t = std::max<int64_t>(1, 1 + ((input.shape()[2] - 1) / 8) * 8);
            if (cropped_t != input.shape()[2]) {
                input = sd::ops::slice(input, 2, 0, cropped_t);
            }
        }
        if (decode_graph && temporal_tiling_enabled && input.dim() == 5 && input.shape()[2] > 1) {
            return decode_temporal_tiled_streaming(n_threads, input, expected_dim);
        }
        if (!decode_graph && input.shape()[2] > 1) {
            const bool force_chunking = encoder_chunk_frames_configured;
            EncoderStreamingPlan plan;
            if (resolve_encoder_streaming_plan(input, force_chunking, &plan)) {
                auto chunked = encode_temporal_chunked_streaming(
                    n_threads,
                    input,
                    expected_dim,
                    plan);
                if (!chunked.empty()) {
                    return chunked;
                }
                LOG_WARN("exact LTX VAE encoder stream failed after preflight; "
                         "retrying the complete graph so automatic CPU fallback can preserve semantics");
            } else if (force_chunking) {
                LOG_WARN("no exact LTX VAE encoder chunk plan fits the configured backend capacity; "
                         "retrying the complete graph so automatic CPU fallback can preserve semantics");
            }
        }
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(input, decode_graph);
        };
        auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), expected_dim);
        if (result.empty()) {
            return {};
        }
        return result;
    }

    sd::Tensor<float> apply_latent_statistics(const int n_threads,
                                              const sd::Tensor<float>& z,
                                              bool normalize) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_latent_statistics_graph(z, normalize);
        };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false),
                                               static_cast<size_t>(z.dim()));
    }

    sd::Tensor<float> normalize_latents(const int n_threads,
                                        const sd::Tensor<float>& z) {
        return apply_latent_statistics(n_threads, z, true);
    }

    sd::Tensor<float> un_normalize_latents(const int n_threads,
                                           const sd::Tensor<float>& z) {
        return apply_latent_statistics(n_threads, z, false);
    }

    int get_encoder_output_channels(int input_channels) override {
        SD_UNUSED(input_channels);
        return 256;
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        SD_UNUSED(rng);
        if (vae_output.dim() >= 4 && vae_output.shape()[3] > 128) {
            return sd::ops::slice(vae_output, 3, 0, 128);
        }
        return vae_output;
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    void test(const std::string& input_path) {
        auto z = sd::load_tensor_from_file_as_tensor<float>(input_path);
        print_sd_tensor(z, false, "ltx_vae_z");

        z = diffusion_to_vae_latents(z);

        int64_t t0 = ggml_time_ms();
        auto out   = _compute(8, z, true);
        int64_t t1 = ggml_time_ms();

        GGML_ASSERT(!out.empty());
        print_sd_tensor(out, false, "ltx_vae_out");
        LOG_DEBUG("ltx vae test done in %lldms", t1 - t0);
    }

    static void load_from_file_and_test(const std::string& model_path,
                                        const std::string& input_path) {
        // ggml_backend_t backend = ggml_backend_cuda_init(0);
        ggml_backend_t backend = sd_backend_cpu_init();
        LOG_INFO("loading ltx vae from '%s'", model_path.c_str());

        auto model_manager        = std::make_shared<ModelManager>();
        ModelLoader& model_loader = model_manager->loader();
        if (!model_loader.init_from_file_and_convert_name(model_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
            return;
        }

        auto& tensor_storage_map         = model_loader.get_tensor_storage_map();
        std::shared_ptr<LTXVideoVAE> vae = std::make_shared<LTXVideoVAE>(backend,
                                                                         tensor_storage_map,
                                                                         "first_stage_model",
                                                                         true,
                                                                         VERSION_LTXAV,
                                                                         model_manager);

        if (!model_manager->register_runner_params("LTX VAE test",
                                                   *vae,
                                                   ModelManager::ResidencyMode::ParamBackend,
                                                   backend,
                                                   backend) ||
            !model_manager->validate_registered_tensors()) {
            LOG_ERROR("register ltx vae tensors with model manager failed");
            return;
        }

        LOG_INFO("ltx vae model loaded");
        vae->test(input_path);
    }
};

#endif  // __SD_MODEL_VAE_LTX_VAE_HPP__
