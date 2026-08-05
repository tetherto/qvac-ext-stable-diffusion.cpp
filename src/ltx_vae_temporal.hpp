#ifndef __SD_LTX_VAE_TEMPORAL_HPP__
#define __SD_LTX_VAE_TEMPORAL_HPP__

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace LTXVAE {

    constexpr int LTX_ENCODER_TEMPORAL_LEVELS = 3;

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
