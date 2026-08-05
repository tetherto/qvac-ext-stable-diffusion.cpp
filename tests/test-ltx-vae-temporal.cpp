#include "ltx_vae_temporal.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

    using Values = std::vector<float>;

    struct ConvState {
        std::array<float, 2> history = {};
        bool started = false;

        Values run(const Values& input) {
            if (input.empty()) {
                return {};
            }
            if (!started) {
                history = {input.front(), input.front()};
                started = true;
            }

            Values out;
            out.reserve(input.size());
            for (float value : input) {
                out.push_back(0.25f * history[0] +
                              0.5f * history[1] +
                              0.75f * value);
                history[0] = history[1];
                history[1] = value;
            }
            return out;
        }
    };

    struct DownsampleState {
        ConvState conv;
        bool started = false;
        bool has_pending = false;
        float pending_x = 0.0f;
        float pending_h = 0.0f;

        Values run(Values input) {
            if (input.empty()) {
                return {};
            }
            if (!started) {
                input.insert(input.begin(), input.front());
                started = true;
            }

            Values h = conv.run(input);
            if (has_pending) {
                input.insert(input.begin(), pending_x);
                h.insert(h.begin(), pending_h);
            }

            has_pending = (input.size() % 2) != 0;
            if (has_pending) {
                pending_x = input.back();
                pending_h = h.back();
                input.pop_back();
                h.pop_back();
            }

            Values out;
            out.reserve(input.size() / 2);
            for (size_t i = 0; i < input.size(); i += 2) {
                const float residual = 0.5f * (input[i] + input[i + 1]);
                const float branch   = 0.5f * (h[i] + h[i + 1]);
                out.push_back(residual + branch);
            }
            return out;
        }
    };

    struct ToyEncoder {
        ConvState conv_in;
        std::array<DownsampleState, 3> downsample;
        std::array<ConvState, 3> between;
        ConvState conv_out;
        size_t max_live_frames = 0;

        Values run(Values input) {
            max_live_frames = std::max(max_live_frames, input.size() + 3);
            Values x = conv_in.run(input);
            for (size_t level = 0; level < downsample.size(); ++level) {
                x = downsample[level].run(std::move(x));
                max_live_frames = std::max(max_live_frames, x.size());
                if (x.empty()) {
                    return {};
                }
                x = between[level].run(x);
            }
            return conv_out.run(x);
        }

        size_t state_slots() const {
            return (1 + 3 + 3 + 1) * 2 + 3 * 2;
        }
    };

    Values make_input(int64_t frames) {
        Values input(static_cast<size_t>(frames));
        for (int64_t i = 0; i < frames; ++i) {
            input[static_cast<size_t>(i)] =
                std::sin(static_cast<float>(i) * 0.37f) +
                static_cast<float>((i * 17) % 11) * 0.03125f;
        }
        return input;
    }

    Values run_chunked(const Values& input,
                       const LTXVAE::EncoderChunkPlan& plan,
                       size_t* state_slots = nullptr,
                       size_t* max_live_frames = nullptr) {
        ToyEncoder encoder;
        Values output;
        size_t start = 0;
        for (const auto& phase : plan.chunks) {
            const size_t end = start + static_cast<size_t>(phase.input_frames);
            Values chunk(input.begin() + static_cast<std::ptrdiff_t>(start),
                         input.begin() + static_cast<std::ptrdiff_t>(end));
            Values chunk_output = encoder.run(std::move(chunk));
            output.insert(output.end(), chunk_output.begin(), chunk_output.end());
            start = end;
        }
        assert(start == input.size());
        for (const auto& downsample : encoder.downsample) {
            assert(!downsample.has_pending);
        }
        if (state_slots != nullptr) {
            *state_slots = encoder.state_slots();
        }
        if (max_live_frames != nullptr) {
            *max_live_frames = encoder.max_live_frames;
        }
        return output;
    }

    void assert_close(const Values& expected,
                      const Values& actual,
                      float tolerance = 1e-6f) {
        assert(expected.size() == actual.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            assert(std::fabs(expected[i] - actual[i]) <= tolerance);
        }
    }

}  // namespace

int main() {
    const std::array<int64_t, 7> boundary_lengths = {1, 9, 17, 25, 33, 65, 217};
    const std::array<int64_t, 7> expected_shapes  = {1, 2, 3, 4, 5, 9, 28};
    const std::array<int, 12> chunk_sizes         = {1, 2, 3, 5, 8, 9, 16, 17, 25, 33, 49, 57};

    for (size_t case_index = 0; case_index < boundary_lengths.size(); ++case_index) {
        const int64_t frames = boundary_lengths[case_index];
        assert(LTXVAE::ltx_encoder_temporal_output_frames(frames) ==
               expected_shapes[case_index]);

        const Values input = make_input(frames);
        ToyEncoder unchunked_encoder;
        const Values unchunked = unchunked_encoder.run(input);
        assert(static_cast<int64_t>(unchunked.size()) == expected_shapes[case_index]);

        for (int chunk_size : chunk_sizes) {
            const auto plan = LTXVAE::make_encoder_chunk_plan(frames, chunk_size);
            assert(plan.exact);
            assert(plan.max_input_frames() <= chunk_size);
            const Values chunked = run_chunked(input, plan);
            assert_close(unchunked, chunked);
        }
    }

    // Small payloads exercise pending-only chunks that emit no latent frame.
    const auto phase_plan = LTXVAE::make_encoder_chunk_plan(17, 5);
    assert(phase_plan.exact);
    assert(phase_plan.chunks.size() == 4);
    assert((phase_plan.chunks[0].first_frame == std::array<bool, 3>{true, true, true}));
    assert((phase_plan.chunks[0].pending_after == std::array<bool, 3>{false, false, true}));
    assert((phase_plan.chunks[1].pending_after == std::array<bool, 3>{true, false, false}));
    assert((phase_plan.chunks[2].pending_after == std::array<bool, 3>{false, true, true}));
    assert(phase_plan.chunks[2].output_frames == 0);
    assert(phase_plan.chunks[3].input_frames == 2);
    assert((phase_plan.chunks[3].pending_after == std::array<bool, 3>{false, false, false}));
    assert(phase_plan.chunks[3].output_frames == 1);

    // Production candidates use a first payload of 8k+1 and 8k continuation
    // payloads, keeping every internal source boundary at 1 modulo 8.
    const auto production_49 = LTXVAE::make_encoder_chunk_plan(217, 49);
    assert(production_49.exact);
    assert(production_49.first_chunk_frames == 49);
    assert(production_49.continuation_frames == 48);
    assert(production_49.chunks.size() == 5);
    assert(production_49.chunks[0].input_frames == 49);
    assert(production_49.chunks[1].input_frames == 48);
    assert(production_49.chunks.back().input_frames == 24);

    const auto maximum_57 = LTXVAE::make_encoder_chunk_plan(217, 57);
    assert(maximum_57.exact);
    assert(maximum_57.first_chunk_frames == 57);
    assert(maximum_57.continuation_frames == 56);
    assert(maximum_57.chunks.size() == 4);
    assert(maximum_57.chunks.back().input_frames == 48);

    const auto aligned_oracle = LTXVAE::make_encoder_chunk_plan(217, 1, 64);
    assert(aligned_oracle.exact);
    assert(aligned_oracle.chunks.size() == 5);
    assert(aligned_oracle.chunks.front().input_frames == 1);
    assert(aligned_oracle.chunks.back().input_frames == 24);

    size_t state_65 = 0;
    size_t state_217 = 0;
    size_t peak_65 = 0;
    size_t peak_217 = 0;
    const auto bounded_65 = LTXVAE::make_encoder_chunk_plan(65, 17);
    const auto bounded_217 = LTXVAE::make_encoder_chunk_plan(217, 17);
    run_chunked(make_input(65), bounded_65, &state_65, &peak_65);
    run_chunked(make_input(217), bounded_217, &state_217, &peak_217);
    ToyEncoder full_217;
    full_217.run(make_input(217));
    assert(state_65 == state_217);
    assert(peak_65 <= 20);
    assert(peak_217 <= 20);
    assert(peak_65 == peak_217);
    assert(full_217.max_live_frames >= 220);
    assert(peak_217 * 10 < full_217.max_live_frames);

    return 0;
}
