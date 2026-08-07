#include "common/common.h"
#include "ggml.h"

#include <cmath>

int main() {
    SDGenerationParams params;
    GGML_ASSERT(params.from_json_str(
        R"({"reference_attention_strength":0.25,"reference_downscale_factor":0.5})"));
    GGML_ASSERT(std::abs(params.reference_attention_strength - 0.25f) < 1e-6f);
    GGML_ASSERT(std::abs(params.reference_downscale_factor - 0.5f) < 1e-6f);

    const sd_vid_gen_params_t api_params =
        params.to_sd_vid_gen_params_t();
    GGML_ASSERT(std::abs(api_params.reference_attention_strength - 0.25f) <
                1e-6f);
    GGML_ASSERT(std::abs(api_params.reference_downscale_factor - 0.5f) <
                1e-6f);
    return 0;
}
