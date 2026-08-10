#include "ggml.h"
#include "ltx_conditioning.hpp"

#include <cmath>
#include <limits>

int main() {
    using sd::LtxReferenceValidation;
    using sd::validate_ltx_reference_parameters;

    GGML_ASSERT(validate_ltx_reference_parameters(0.f, 1.f) ==
                LtxReferenceValidation::VALID);
    GGML_ASSERT(validate_ltx_reference_parameters(1.f, 1.f) ==
                LtxReferenceValidation::VALID);
    GGML_ASSERT(validate_ltx_reference_parameters(
                    std::numeric_limits<float>::quiet_NaN(),
                    1.f) ==
                LtxReferenceValidation::NONFINITE_ATTENTION_STRENGTH);
    GGML_ASSERT(validate_ltx_reference_parameters(
                    std::numeric_limits<float>::infinity(),
                    1.f) ==
                LtxReferenceValidation::NONFINITE_ATTENTION_STRENGTH);
    GGML_ASSERT(validate_ltx_reference_parameters(-0.01f, 1.f) ==
                LtxReferenceValidation::ATTENTION_STRENGTH_OUT_OF_RANGE);
    GGML_ASSERT(validate_ltx_reference_parameters(0.5f, 0.999f) ==
                LtxReferenceValidation::UNSUPPORTED_DOWNSCALE_FACTOR);
    GGML_ASSERT(validate_ltx_reference_parameters(
                    0.5f,
                    std::numeric_limits<float>::quiet_NaN()) ==
                LtxReferenceValidation::NONFINITE_DOWNSCALE_FACTOR);

    const auto first_pass = sd::build_ltxv_video_positions(
        2, 2, 3, 2, 0, 1, 24, 32, 8, true, true);
    const auto refine_pass = sd::build_ltxv_video_positions(
        4, 4, 3, 2, 0, 1, 24, 32, 8, true, true);
    GGML_ASSERT(!first_pass.empty());
    GGML_ASSERT(!refine_pass.empty());
    GGML_ASSERT(first_pass.shape()[2] == 2 * 2 * (3 + 2));
    GGML_ASSERT(refine_pass.shape()[2] == 4 * 4 * (3 + 2));

    // Reference tokens remain first and overlap the target timeline in both
    // passes, while spatial coordinates are rebuilt for the refined grid.
    GGML_ASSERT(first_pass.index(0, 0, 0, 0) == 0.f);
    GGML_ASSERT(refine_pass.index(0, 0, 0, 0) == 0.f);
    const int64_t first_target_token  = 2 * 2 * 2;
    const int64_t refine_target_token = 4 * 4 * 2;
    GGML_ASSERT(first_pass.index(0, 0, first_target_token, 0) == 0.f);
    GGML_ASSERT(refine_pass.index(0, 0, refine_target_token, 0) == 0.f);
    GGML_ASSERT(first_pass.index(1, 2, 1, 0) == 64.f);
    GGML_ASSERT(refine_pass.index(1, 2, 3, 0) == 128.f);
    return 0;
}
