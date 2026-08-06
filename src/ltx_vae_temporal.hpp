#ifndef __SD_LTX_VAE_TEMPORAL_HPP__
#define __SD_LTX_VAE_TEMPORAL_HPP__

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace LTXVAE {

    constexpr int LTX_ENCODER_TEMPORAL_LEVELS = 3;
    constexpr int LTX_DECODER_TEMPORAL_LEVELS = 3;

    enum class DecoderBackendKind {
        VULKAN,
        METAL,
        CUDA,
        OTHER,
    };

    enum class DecoderExecutionMode {
        DEFAULT,
        EXACT_STATEFUL,
    };

    inline DecoderExecutionMode decoder_execution_mode(
        DecoderBackendKind backend) {
        return backend == DecoderBackendKind::VULKAN
                   ? DecoderExecutionMode::EXACT_STATEFUL
                   : DecoderExecutionMode::DEFAULT;
    }

    inline int64_t ltx_decoder_temporal_output_frames(int64_t latent_frames) {
        return latent_frames > 0 ? 1 + (latent_frames - 1) * 8 : 0;
    }

    struct DecoderConvPhase {
        int64_t input_frames  = 0;
        int64_t output_frames = 0;
        bool started_before   = false;
        bool started_after    = false;
        bool final            = false;
    };

    struct DecoderConvTemporalState {
        bool started = false;

        DecoderConvPhase advance(int64_t input_frames, bool final) {
            DecoderConvPhase phase;
            phase.input_frames  = std::max<int64_t>(0, input_frames);
            phase.started_before = started;
            phase.final          = final;

            if (!started) {
                if (phase.input_frames > 0) {
                    phase.output_frames =
                        final ? phase.input_frames
                              : std::max<int64_t>(0, phase.input_frames - 1);
                    started = !final;
                }
            } else {
                // A started kernel-3 convolution owns one pending output. A
                // non-final call replaces it with the newest pending output;
                // final flush supplies the model's single repeat-right frame.
                phase.output_frames =
                    final ? phase.input_frames + 1 : phase.input_frames;
                if (final) {
                    started = false;
                }
            }
            phase.started_after = started;
            return phase;
        }
    };

    struct DecoderResidualTemporalState {
        int64_t pending_frames = 0;

        bool advance(int64_t input_frames,
                     int64_t output_frames,
                     bool final) {
            if (input_frames < 0 || output_frames < 0) {
                return false;
            }
            const int64_t available = pending_frames + input_frames;
            if (output_frames > available) {
                return false;
            }
            pending_frames = available - output_frames;
            return !final || pending_frames == 0;
        }
    };

    struct DecoderUpsamplePhase {
        int64_t input_frames  = 0;
        int64_t output_frames = 0;
        bool drop_first       = false;
    };

    struct DecoderUpsampleTemporalState {
        bool dropped_first = false;

        DecoderUpsamplePhase advance(int64_t input_frames) {
            DecoderUpsamplePhase phase;
            phase.input_frames = std::max<int64_t>(0, input_frames);
            if (phase.input_frames == 0) {
                return phase;
            }
            phase.drop_first = !dropped_first;
            dropped_first    = true;
            phase.output_frames =
                phase.input_frames * 2 - (phase.drop_first ? 1 : 0);
            return phase;
        }
    };

    struct DecoderTemporalState {
        std::vector<DecoderConvTemporalState> convolutions;
        std::vector<DecoderResidualTemporalState> residuals;
        std::array<DecoderUpsampleTemporalState,
                   LTX_DECODER_TEMPORAL_LEVELS>
            upsamplers = {};
        int64_t consumed_latent_frames = 0;
        int64_t emitted_frames         = 0;

        DecoderConvPhase advance_convolution(size_t index,
                                             int64_t input_frames,
                                             bool final) {
            if (index >= convolutions.size()) {
                convolutions.resize(index + 1);
            }
            return convolutions[index].advance(input_frames, final);
        }

        bool advance_residual(size_t index,
                              int64_t input_frames,
                              int64_t output_frames,
                              bool final) {
            if (index >= residuals.size()) {
                residuals.resize(index + 1);
            }
            return residuals[index].advance(input_frames,
                                             output_frames,
                                             final);
        }

        DecoderUpsamplePhase advance_upsampler(size_t index,
                                               int64_t input_frames) {
            if (index >= upsamplers.size()) {
                return {};
            }
            return upsamplers[index].advance(input_frames);
        }

        bool exact_at_end(size_t expected_convolutions,
                          size_t expected_residuals,
                          int64_t expected_latent_frames) const {
            if (convolutions.size() != expected_convolutions ||
                residuals.size() != expected_residuals ||
                consumed_latent_frames != expected_latent_frames ||
                emitted_frames !=
                    ltx_decoder_temporal_output_frames(expected_latent_frames)) {
                return false;
            }
            for (const auto& convolution : convolutions) {
                if (convolution.started) {
                    return false;
                }
            }
            for (const auto& residual : residuals) {
                if (residual.pending_frames != 0) {
                    return false;
                }
            }
            for (const auto& upsampler : upsamplers) {
                if (!upsampler.dropped_first && expected_latent_frames > 0) {
                    return false;
                }
            }
            return true;
        }
    };

    enum class DecoderGraphPhase {
        INITIAL,
        STEADY,
        TAIL,
        FINAL_FLUSH,
    };

    struct DecoderCapacitySample {
        DecoderGraphPhase phase = DecoderGraphPhase::STEADY;
        size_t compute_bytes    = 0;
        size_t cache_bytes      = 0;
    };

    struct DecoderCapacityPlan {
        bool fits = false;
        size_t max_buffer_bytes  = std::numeric_limits<size_t>::max();
        size_t free_budget_bytes = std::numeric_limits<size_t>::max();
        size_t runtime_param_bytes = 0;
        size_t output_writer_bytes = 0;
        size_t max_compute_bytes   = 0;
        size_t max_cache_bytes     = 0;
        std::array<size_t, 4> phase_compute_bytes = {};
    };

    inline bool decoder_size_add_fits(size_t a,
                                      size_t b,
                                      size_t limit,
                                      size_t* result = nullptr) {
        if (a > limit || b > limit - a) {
            return false;
        }
        if (result != nullptr) {
            *result = a + b;
        }
        return true;
    }

    inline DecoderCapacityPlan make_decoder_capacity_plan(
        const std::vector<DecoderCapacitySample>& samples,
        size_t runtime_param_bytes,
        size_t output_writer_bytes,
        size_t max_buffer_bytes,
        size_t free_budget_bytes) {
        DecoderCapacityPlan plan;
        plan.max_buffer_bytes    = max_buffer_bytes;
        plan.free_budget_bytes   = free_budget_bytes;
        plan.runtime_param_bytes = runtime_param_bytes;
        plan.output_writer_bytes = output_writer_bytes;
        plan.fits                = !samples.empty();

        for (const auto& sample : samples) {
            plan.max_compute_bytes =
                std::max(plan.max_compute_bytes, sample.compute_bytes);
            plan.max_cache_bytes =
                std::max(plan.max_cache_bytes, sample.cache_bytes);
            const size_t phase_index = static_cast<size_t>(sample.phase);
            if (phase_index < plan.phase_compute_bytes.size()) {
                plan.phase_compute_bytes[phase_index] =
                    std::max(plan.phase_compute_bytes[phase_index],
                             sample.compute_bytes);
            }
            plan.fits = plan.fits &&
                        sample.compute_bytes <= max_buffer_bytes &&
                        sample.cache_bytes <= max_buffer_bytes;
        }
        plan.fits = plan.fits &&
                    output_writer_bytes <= max_buffer_bytes &&
                    runtime_param_bytes <= free_budget_bytes;

        // Cache replacement temporarily owns both old and new contiguous
        // buffers. Include that peak together with graph and resident params.
        size_t live_bytes = runtime_param_bytes;
        plan.fits = plan.fits &&
                    decoder_size_add_fits(live_bytes,
                                          plan.max_compute_bytes,
                                          free_budget_bytes,
                                          &live_bytes) &&
                    decoder_size_add_fits(live_bytes,
                                          plan.max_cache_bytes,
                                          free_budget_bytes,
                                          &live_bytes) &&
                    decoder_size_add_fits(live_bytes,
                                          plan.max_cache_bytes,
                                          free_budget_bytes,
                                          &live_bytes);
        return plan;
    }

    inline bool write_decoder_output_chunk_f32(
        float* output,
        const float* chunk,
        int64_t inner_size,
        int64_t outer_size,
        int64_t total_frames,
        int64_t output_begin,
        int64_t chunk_frames) {
        if (output == nullptr || chunk == nullptr ||
            inner_size <= 0 || outer_size <= 0 ||
            total_frames <= 0 || output_begin < 0 ||
            chunk_frames <= 0 ||
            output_begin > total_frames - chunk_frames) {
            return false;
        }
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const int64_t source =
                outer * chunk_frames * inner_size;
            const int64_t destination =
                (outer * total_frames + output_begin) * inner_size;
            std::copy_n(chunk + source,
                        chunk_frames * inner_size,
                        output + destination);
        }
        return true;
    }

    struct EncoderTemporalPhase {
        int64_t input_frames  = 0;
        int64_t output_frames = 0;
        std::array<int64_t, LTX_ENCODER_TEMPORAL_LEVELS> level_input_frames  = {};
        std::array<int64_t, LTX_ENCODER_TEMPORAL_LEVELS> level_output_frames = {};
        std::array<bool, LTX_ENCODER_TEMPORAL_LEVELS> first_frame            = {};
        std::array<bool, LTX_ENCODER_TEMPORAL_LEVELS> pending_before         = {};
        std::array<bool, LTX_ENCODER_TEMPORAL_LEVELS> pending_after          = {};
    };

    struct EncoderTemporalState {
        std::array<bool, LTX_ENCODER_TEMPORAL_LEVELS> started = {};
        std::array<bool, LTX_ENCODER_TEMPORAL_LEVELS> pending = {};
        int64_t consumed_frames = 0;
        int64_t emitted_frames  = 0;

        EncoderTemporalPhase advance(int64_t frames) {
            EncoderTemporalPhase phase;
            phase.input_frames = std::max<int64_t>(0, frames);

            int64_t level_frames = phase.input_frames;
            for (int level = 0; level < LTX_ENCODER_TEMPORAL_LEVELS; ++level) {
                phase.level_input_frames[level] = level_frames;
                phase.pending_before[level]     = pending[level];

                if (level_frames == 0) {
                    phase.pending_after[level]       = pending[level];
                    phase.level_output_frames[level] = 0;
                    continue;
                }

                phase.first_frame[level] = !started[level];
                int64_t grouped_frames   = level_frames + (pending[level] ? 1 : 0);
                if (!started[level]) {
                    // The unchunked encoder prepends one copy of the first frame
                    // before each factor-2 temporal SpaceToDepth level.
                    grouped_frames += 1;
                    started[level] = true;
                }

                level_frames                     = grouped_frames / 2;
                pending[level]                   = (grouped_frames % 2) != 0;
                phase.pending_after[level]       = pending[level];
                phase.level_output_frames[level] = level_frames;
            }

            phase.output_frames = level_frames;
            consumed_frames += phase.input_frames;
            emitted_frames += phase.output_frames;
            return phase;
        }

        bool exact_at_end() const {
            return !pending[0] && !pending[1] && !pending[2];
        }
    };

    struct EncoderChunkPlan {
        bool exact = false;
        int64_t total_frames = 0;
        int first_chunk_frames = 0;
        int continuation_frames = 0;
        std::vector<EncoderTemporalPhase> chunks;

        int64_t max_input_frames() const {
            int64_t result = 0;
            for (const auto& chunk : chunks) {
                result = std::max(result, chunk.input_frames);
            }
            return result;
        }
    };

    inline int64_t ltx_encoder_temporal_output_frames(int64_t input_frames) {
        if (input_frames <= 0) {
            return 0;
        }
        return 1 + (input_frames - 1) / 8;
    }

    inline EncoderChunkPlan make_encoder_chunk_plan(int64_t total_frames,
                                                    int first_chunk_frames,
                                                    int continuation_frames) {
        EncoderChunkPlan plan;
        plan.total_frames        = total_frames;
        plan.first_chunk_frames  = first_chunk_frames;
        plan.continuation_frames = continuation_frames;
        if (total_frames <= 0 || first_chunk_frames <= 0 || continuation_frames <= 0) {
            return plan;
        }

        EncoderTemporalState state;
        int64_t start = 0;
        while (start < total_frames) {
            const int64_t payload =
                plan.chunks.empty() ? first_chunk_frames : continuation_frames;
            const int64_t frames = std::min<int64_t>(payload, total_frames - start);
            plan.chunks.push_back(state.advance(frames));
            start += frames;
        }

        plan.exact = state.exact_at_end() &&
                     state.consumed_frames == total_frames &&
                     state.emitted_frames == ltx_encoder_temporal_output_frames(total_frames);
        return plan;
    }

    inline EncoderChunkPlan make_encoder_chunk_plan(int64_t total_frames,
                                                    int max_chunk_frames) {
        if (max_chunk_frames < 9) {
            return make_encoder_chunk_plan(total_frames,
                                           max_chunk_frames,
                                           max_chunk_frames);
        }
        const int continuation_frames = ((max_chunk_frames - 1) / 8) * 8;
        return make_encoder_chunk_plan(total_frames,
                                       continuation_frames + 1,
                                       continuation_frames);
    }

}  // namespace LTXVAE

#endif
