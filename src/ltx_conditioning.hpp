#ifndef __SD_LTX_CONDITIONING_HPP__
#define __SD_LTX_CONDITIONING_HPP__

#include <cmath>
#include <cstdint>

#include "core/tensor.hpp"

namespace sd {

    enum class LtxReferenceValidation {
        VALID,
        NONFINITE_ATTENTION_STRENGTH,
        ATTENTION_STRENGTH_OUT_OF_RANGE,
        NONFINITE_DOWNSCALE_FACTOR,
        UNSUPPORTED_DOWNSCALE_FACTOR,
        INVALID_REFERENCE_COUNT,
        MISSING_REFERENCE_IMAGES,
    };

    template <typename Image>
    inline LtxReferenceValidation validate_ltx_reference_inputs(
        const Image* images,
        int image_count) {
        if (image_count < 0) {
            return LtxReferenceValidation::INVALID_REFERENCE_COUNT;
        }
        if (image_count > 0 && images == nullptr) {
            return LtxReferenceValidation::MISSING_REFERENCE_IMAGES;
        }
        return LtxReferenceValidation::VALID;
    }

    inline LtxReferenceValidation validate_ltx_reference_parameters(
        float attention_strength,
        float downscale_factor) {
        if (!std::isfinite(attention_strength)) {
            return LtxReferenceValidation::NONFINITE_ATTENTION_STRENGTH;
        }
        if (attention_strength < 0.f || attention_strength > 1.f) {
            return LtxReferenceValidation::ATTENTION_STRENGTH_OUT_OF_RANGE;
        }
        if (!std::isfinite(downscale_factor)) {
            return LtxReferenceValidation::NONFINITE_DOWNSCALE_FACTOR;
        }
        if (downscale_factor != 1.f) {
            return LtxReferenceValidation::UNSUPPORTED_DOWNSCALE_FACTOR;
        }
        return LtxReferenceValidation::VALID;
    }

    inline float ltxv_latent_corner_to_pixel_frame(
        int64_t corner_index,
        int temporal_scale,
        bool causal_temporal_positioning) {
        float pixel_t = static_cast<float>(corner_index * temporal_scale);
        if (causal_temporal_positioning) {
            pixel_t = std::max(
                0.f,
                pixel_t + 1.f - static_cast<float>(temporal_scale));
        }
        return pixel_t;
    }

    inline void set_ltxv_video_position(Tensor<float>* positions,
                                        int64_t token,
                                        float t_start,
                                        float t_end,
                                        float h_start,
                                        float h_end,
                                        float w_start,
                                        float w_end) {
        positions->index(0, 0, token, 0) = t_start;
        positions->index(1, 0, token, 0) = t_end;
        positions->index(0, 1, token, 0) = h_start;
        positions->index(1, 1, token, 0) = h_end;
        positions->index(0, 2, token, 0) = w_start;
        positions->index(1, 2, token, 0) = w_end;
    }

    inline Tensor<float> build_ltxv_video_positions(
        int64_t width,
        int64_t height,
        int64_t target_latent_frames,
        int64_t conditioning_latent_frames,
        int conditioning_frame_idx,
        int conditioning_pixel_frames,
        int fps,
        int spatial_scale,
        int temporal_scale,
        bool causal_temporal_positioning,
        bool conditioning_first = false) {
        if (width <= 0 || height <= 0 || target_latent_frames <= 0 ||
            conditioning_latent_frames <= 0 || fps <= 0) {
            return {};
        }

        const int64_t total_tokens =
            width * height *
            (target_latent_frames + conditioning_latent_frames);
        Tensor<float> positions({2, 3, total_tokens, 1});
        int64_t token = 0;

        auto append_positions = [&](bool conditioning) {
            const int64_t frame_count = conditioning
                                            ? conditioning_latent_frames
                                            : target_latent_frames;
            for (int64_t t = 0; t < frame_count; ++t) {
                const int64_t start_corner =
                    (conditioning ? conditioning_frame_idx : 0) + t;
                const int64_t end_corner = start_corner + 1;
                float t_start            = ltxv_latent_corner_to_pixel_frame(
                    start_corner,
                    temporal_scale,
                    causal_temporal_positioning);
                float t_end = ltxv_latent_corner_to_pixel_frame(
                    end_corner,
                    temporal_scale,
                    causal_temporal_positioning);
                if (conditioning && conditioning_pixel_frames == 1) {
                    t_end = t_start + 1.f;
                }
                t_start /= static_cast<float>(fps);
                t_end /= static_cast<float>(fps);
                for (int64_t h = 0; h < height; ++h) {
                    const float h_start =
                        static_cast<float>(h * spatial_scale);
                    const float h_end =
                        static_cast<float>((h + 1) * spatial_scale);
                    for (int64_t w = 0; w < width; ++w) {
                        const float w_start =
                            static_cast<float>(w * spatial_scale);
                        const float w_end =
                            static_cast<float>((w + 1) * spatial_scale);
                        set_ltxv_video_position(&positions,
                                                token++,
                                                t_start,
                                                t_end,
                                                h_start,
                                                h_end,
                                                w_start,
                                                w_end);
                    }
                }
            }
        };

        if (conditioning_first) {
            append_positions(true);
            append_positions(false);
        } else {
            append_positions(false);
            append_positions(true);
        }
        return positions;
    }

}  // namespace sd

#endif
