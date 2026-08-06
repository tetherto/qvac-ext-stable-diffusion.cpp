#include "ltx_vae_temporal.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

    using Values = std::vector<float>;

    Values make_input(int64_t frames) {
        Values input(static_cast<size_t>(frames));
        for (int64_t i = 0; i < frames; ++i) {
            input[static_cast<size_t>(i)] =
                std::sin(static_cast<float>(i) * 0.31f) +
                static_cast<float>((i * 13) % 7) * 0.0625f;
        }
        return input;
    }

    void assert_close(const Values& expected,
                      const Values& actual,
                      float tolerance = 1e-6f) {
        assert(expected.size() == actual.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            assert(std::fabs(expected[i] - actual[i]) <= tolerance);
        }
    }

    struct StreamingConv {
        LTXVAE::DecoderConvTemporalState temporal;
        Values cache;
        int right_pads = 0;

        Values run(const Values& input, bool final) {
            const auto phase = temporal.advance(
                static_cast<int64_t>(input.size()),
                final);
            Values padded;
            if (phase.started_before) {
                assert(cache.size() == 2);
                padded = cache;
                padded.insert(padded.end(), input.begin(), input.end());
            } else if (!input.empty()) {
                padded.push_back(input.front());
                padded.insert(padded.end(), input.begin(), input.end());
            }
            if (final && !padded.empty()) {
                padded.push_back(padded.back());
                right_pads++;
            } else if (!final && !padded.empty()) {
                assert(padded.size() >= 2);
                cache.assign(padded.end() - 2, padded.end());
            }

            Values output;
            for (size_t i = 0; i + 2 < padded.size(); ++i) {
                output.push_back(0.25f * padded[i] +
                                 0.5f * padded[i + 1] +
                                 0.75f * padded[i + 2]);
            }
            assert(static_cast<int64_t>(output.size()) ==
                   phase.output_frames);
            if (final) {
                cache.clear();
            }
            return output;
        }
    };

    struct StreamingResidual {
        StreamingConv conv1;
        StreamingConv conv2;
        LTXVAE::DecoderResidualTemporalState temporal;
        Values pending;

        Values run(const Values& input, bool final) {
            pending.insert(pending.end(), input.begin(), input.end());
            Values branch = conv1.run(input, final);
            if (!branch.empty()) {
                for (float& value : branch) {
                    value = std::tanh(value);
                }
                branch = conv2.run(branch, final);
            }
            assert(temporal.advance(
                static_cast<int64_t>(input.size()),
                static_cast<int64_t>(branch.size()),
                final));
            assert(static_cast<int64_t>(pending.size()) >=
                   static_cast<int64_t>(branch.size()));
            for (size_t i = 0; i < branch.size(); ++i) {
                branch[i] += pending[i];
            }
            pending.erase(pending.begin(),
                          pending.begin() +
                              static_cast<std::ptrdiff_t>(branch.size()));
            assert(static_cast<int64_t>(pending.size()) ==
                   temporal.pending_frames);
            return branch;
        }
    };

    struct StreamingUpsample {
        StreamingConv conv;
        LTXVAE::DecoderUpsampleTemporalState temporal;

        Values run(const Values& input, bool final) {
            Values convolved = conv.run(input, final);
            const auto phase = temporal.advance(
                static_cast<int64_t>(convolved.size()));
            Values output;
            for (float value : convolved) {
                output.push_back(value - 0.125f);
                output.push_back(value + 0.125f);
            }
            if (phase.drop_first) {
                assert(!output.empty());
                output.erase(output.begin());
            }
            assert(static_cast<int64_t>(output.size()) ==
                   phase.output_frames);
            return output;
        }
    };

    struct ToyDecoder {
        StreamingConv conv_in;
        std::array<StreamingResidual, 4> residuals;
        std::array<StreamingUpsample, 3> upsamplers;
        StreamingConv conv_out;

        Values run(Values input, bool final) {
            Values x = conv_in.run(input, final);
            if (x.empty()) {
                return {};
            }
            for (size_t i = 0; i < residuals.size(); ++i) {
                x = residuals[i].run(x, final);
                if (x.empty()) {
                    return {};
                }
                if (i < upsamplers.size()) {
                    x = upsamplers[i].run(x, final);
                    if (x.empty()) {
                        return {};
                    }
                }
            }
            return conv_out.run(x, final);
        }

        int right_pad_count() const {
            int count = conv_in.right_pads + conv_out.right_pads;
            for (const auto& residual : residuals) {
                count += residual.conv1.right_pads +
                         residual.conv2.right_pads;
            }
            for (const auto& upsampler : upsamplers) {
                count += upsampler.conv.right_pads;
            }
            return count;
        }

        bool clean() const {
            if (conv_in.temporal.started || conv_out.temporal.started) {
                return false;
            }
            for (const auto& residual : residuals) {
                if (residual.conv1.temporal.started ||
                    residual.conv2.temporal.started ||
                    residual.temporal.pending_frames != 0 ||
                    !residual.pending.empty()) {
                    return false;
                }
            }
            for (const auto& upsampler : upsamplers) {
                if (upsampler.conv.temporal.started) {
                    return false;
                }
            }
            return true;
        }
    };

    Values run_partitioned(const Values& input,
                           const std::vector<int64_t>& parts,
                           ToyDecoder* decoder_out = nullptr) {
        ToyDecoder decoder;
        Values output;
        int64_t begin = 0;
        for (size_t i = 0; i < parts.size(); ++i) {
            const int64_t end = begin + parts[i];
            assert(end <= static_cast<int64_t>(input.size()));
            Values chunk(input.begin() + begin, input.begin() + end);
            Values emitted = decoder.run(std::move(chunk),
                                         i + 1 == parts.size());
            output.insert(output.end(), emitted.begin(), emitted.end());
            begin = end;
        }
        assert(begin == static_cast<int64_t>(input.size()));
        assert(decoder.clean());
        if (decoder_out != nullptr) {
            *decoder_out = std::move(decoder);
        }
        return output;
    }

    std::vector<int64_t> partition_from_mask(int64_t frames,
                                             uint64_t mask) {
        std::vector<int64_t> parts;
        int64_t run = 1;
        for (int64_t i = 1; i < frames; ++i) {
            if ((mask & (uint64_t{1} << (i - 1))) != 0) {
                parts.push_back(run);
                run = 1;
            } else {
                run++;
            }
        }
        parts.push_back(run);
        return parts;
    }

}  // namespace

int main() {
    // Decoder routing is backend-derived and has no caller-controlled mode.
    // Vulkan always selects exact stateful decoding; existing Metal, CUDA,
    // and other backend schedules remain on their default path.
    assert(LTXVAE::decoder_execution_mode(
               LTXVAE::DecoderBackendKind::VULKAN) ==
           LTXVAE::DecoderExecutionMode::EXACT_STATEFUL);
    assert(LTXVAE::decoder_execution_mode(
               LTXVAE::DecoderBackendKind::METAL) ==
           LTXVAE::DecoderExecutionMode::DEFAULT);
    assert(LTXVAE::decoder_execution_mode(
               LTXVAE::DecoderBackendKind::CUDA) ==
           LTXVAE::DecoderExecutionMode::DEFAULT);
    assert(LTXVAE::decoder_execution_mode(
               LTXVAE::DecoderBackendKind::OTHER) ==
           LTXVAE::DecoderExecutionMode::DEFAULT);

    // A kernel-3 noncausal convolution is identical for every partition,
    // including single-frame chunks and a final pending-only flush.
    for (int64_t frames = 1; frames <= 9; ++frames) {
        const Values input = make_input(frames);
        StreamingConv monolithic;
        const Values expected = monolithic.run(input, true);
        const uint64_t partition_count =
            uint64_t{1} << std::max<int64_t>(0, frames - 1);
        for (uint64_t mask = 0; mask < partition_count; ++mask) {
            StreamingConv streamed;
            Values actual;
            int64_t begin = 0;
            const auto parts = partition_from_mask(frames, mask);
            for (size_t i = 0; i < parts.size(); ++i) {
                Values chunk(input.begin() + begin,
                             input.begin() + begin + parts[i]);
                Values emitted = streamed.run(
                    chunk,
                    i + 1 == parts.size());
                actual.insert(actual.end(),
                              emitted.begin(),
                              emitted.end());
                begin += parts[i];
            }
            assert_close(expected, actual);
            assert(streamed.right_pads == 1);
            assert(streamed.cache.empty());
        }
    }

    const std::array<int64_t, 6> lengths = {1, 2, 3, 5, 9, 28};
    for (int64_t frames : lengths) {
        const Values input = make_input(frames);
        ToyDecoder monolithic_decoder;
        const Values expected = monolithic_decoder.run(input, true);
        assert(static_cast<int64_t>(expected.size()) ==
               LTXVAE::ltx_decoder_temporal_output_frames(frames));
        assert(monolithic_decoder.clean());

        if (frames <= 9) {
            const uint64_t partition_count =
                uint64_t{1} << std::max<int64_t>(0, frames - 1);
            for (uint64_t mask = 0; mask < partition_count; ++mask) {
                const auto actual = run_partitioned(
                    input,
                    partition_from_mask(frames, mask));
                assert_close(expected, actual);
            }
        }

        std::mt19937 rng(static_cast<uint32_t>(frames));
        for (int trial = 0; trial < 64; ++trial) {
            std::vector<int64_t> parts;
            int64_t remaining = frames;
            while (remaining > 0) {
                const int64_t part = std::min<int64_t>(
                    remaining,
                    1 + static_cast<int64_t>(rng() % 5));
                parts.push_back(part);
                remaining -= part;
            }
            assert_close(expected, run_partitioned(input, parts));
        }
    }
    assert(LTXVAE::ltx_decoder_temporal_output_frames(28) == 217);

    // Empty and partial emissions preserve residual queues and each upsampler
    // drops its special first output only once globally.
    ToyDecoder partial;
    Values first = partial.run({1.0f}, false);
    assert(first.empty());
    Values second = partial.run({2.0f}, false);
    assert(second.empty());
    Values final = partial.run({3.0f}, true);
    assert(final.size() == 17);
    assert(partial.clean());
    assert(partial.right_pad_count() == 13);
    for (const auto& upsampler : partial.upsamplers) {
        assert(upsampler.temporal.dropped_first);
    }

    // Reset/repeated runs own their caches and do not retain stale state.
    const Values repeated_input = make_input(9);
    const Values first_run = run_partitioned(repeated_input, {1, 3, 5});
    const Values second_run = run_partitioned(repeated_input, {2, 2, 2, 3});
    assert_close(first_run, second_run);

    // Capacity preflight checks every graph class and rejects before state if
    // compute, cache, writer, or replacement peak exceeds the limit.
    constexpr size_t MIB = 1024ull * 1024ull;
    std::vector<LTXVAE::DecoderCapacitySample> samples = {
        {LTXVAE::DecoderGraphPhase::INITIAL, 500 * MIB, 100 * MIB},
        {LTXVAE::DecoderGraphPhase::STEADY, 2100 * MIB, 280 * MIB},
        {LTXVAE::DecoderGraphPhase::TAIL, 1900 * MIB, 280 * MIB},
        {LTXVAE::DecoderGraphPhase::FINAL_FLUSH, 2300 * MIB, 280 * MIB},
    };
    const auto fits = LTXVAE::make_decoder_capacity_plan(
        samples,
        100 * MIB,
        900 * MIB,
        4096 * MIB,
        4096 * MIB);
    assert(fits.fits);
    assert(fits.max_compute_bytes == 2300 * MIB);
    assert(fits.max_cache_bytes == 280 * MIB);

    samples.back().compute_bytes = 4096 * MIB + 1;
    assert(!LTXVAE::make_decoder_capacity_plan(
                samples,
                0,
                900 * MIB,
                4096 * MIB,
                std::numeric_limits<size_t>::max())
                .fits);
    samples.back().compute_bytes = 2300 * MIB;
    assert(!LTXVAE::make_decoder_capacity_plan(
                samples,
                0,
                4096 * MIB + 1,
                4096 * MIB,
                std::numeric_limits<size_t>::max())
                .fits);
    samples[1].cache_bytes = 4096 * MIB + 1;
    assert(!LTXVAE::make_decoder_capacity_plan(
                samples,
                0,
                900 * MIB,
                4096 * MIB,
                std::numeric_limits<size_t>::max())
                .fits);
    samples[1].cache_bytes = 280 * MIB;
    assert(!LTXVAE::make_decoder_capacity_plan(
                samples,
                100 * MIB,
                900 * MIB,
                4096 * MIB,
                2859 * MIB)
                .fits);

    // A synthetic reset-at-chunk decoder has a boundary seam; exact stateful
    // output remains partition invariant and therefore has zero seam error.
    const Values seam_input = make_input(9);
    ToyDecoder oracle_decoder;
    const Values oracle = oracle_decoder.run(seam_input, true);
    const Values exact = run_partitioned(seam_input, {3, 3, 3});
    assert_close(oracle, exact);
    Values reset;
    for (int64_t begin = 0; begin < 9; begin += 3) {
        ToyDecoder independent;
        Values tile(seam_input.begin() + begin,
                    seam_input.begin() + begin + 3);
        Values decoded = independent.run(tile, true);
        reset.insert(reset.end(), decoded.begin(), decoded.end());
    }
    float reset_error = 0.0f;
    for (size_t i = 0; i < std::min(reset.size(), oracle.size()); ++i) {
        reset_error = std::max(reset_error,
                               std::fabs(reset[i] - oracle[i]));
    }
    assert(reset_error > 1e-3f);
    return 0;
}
