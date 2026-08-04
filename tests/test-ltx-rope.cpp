#include "model/diffusion/ltxv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t hash_float_bits(const std::vector<float>& values) {
    uint64_t hash = 1469598103934665603ULL;
    for (float value : values) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (bits >> shift) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static bool has_valid_rotation_blocks(const std::vector<float>& values) {
    if (values.size() % 4 != 0) {
        return false;
    }
    for (size_t i = 0; i < values.size(); i += 4) {
        if (values[i] != values[i + 3] || values[i + 1] != -values[i + 2]) {
            return false;
        }
        const double norm = static_cast<double>(values[i]) * values[i] +
                            static_cast<double>(values[i + 2]) * values[i + 2];
        if (std::abs(norm - 1.0) > 1e-7) {
            return false;
        }
    }
    return true;
}

int main() {
    const auto video = LTXV::build_video_rope_matrix(
        4, 3, 2, 4096, 32, 24.f, 10000.f, {20, 2048, 2048}, {8, 32, 32}, false, true);
    const auto connector = LTXV::build_1d_rope_matrix(37, 1024, 8, 10000.f, 4096.f, true);

    const uint64_t video_hash = hash_float_bits(video);
    const uint64_t connector_hash = hash_float_bits(connector);

    if (!has_valid_rotation_blocks(video) || !has_valid_rotation_blocks(connector)) {
        std::fprintf(stderr, "invalid RoPE rotation block\n");
        return 1;
    }

    // Golden hashes cover frequency generation, coordinate normalization,
    // head splitting, and binary64 sin/cos rounding to the shared F32 table.
    constexpr uint64_t expected_video_hash = 0x17060c3ec6c902ebULL;
    constexpr uint64_t expected_connector_hash = 0x89d227927a382d13ULL;
    if (video_hash != expected_video_hash || connector_hash != expected_connector_hash) {
        std::fprintf(stderr,
                     "RoPE hash mismatch: video=%016llx connector=%016llx\n",
                     static_cast<unsigned long long>(video_hash),
                     static_cast<unsigned long long>(connector_hash));
        return 1;
    }

    return 0;
}
