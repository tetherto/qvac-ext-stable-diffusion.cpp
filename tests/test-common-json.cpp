#include "common/common.h"
#include "ggml.h"

#include <cmath>
#include <limits>

namespace {

    void add_reference_payload(SDGenerationParams* params) {
        GGML_ASSERT(params != nullptr);
        params->ref_images.emplace_back(
            sd_image_t{1, 1, 3, nullptr});
    }

}  // namespace

int main() {
    SDGenerationParams request_values;
    GGML_ASSERT(request_values.from_json_str(
        R"({"reference_attention_strength":0.25,"reference_downscale_factor":1.0})"));
    add_reference_payload(&request_values);
    GGML_ASSERT(request_values.validate(VID_GEN));
    GGML_ASSERT(std::abs(request_values.reference_attention_strength - 0.25f) <
                1e-6f);
    GGML_ASSERT(std::abs(request_values.reference_downscale_factor - 1.f) <
                1e-6f);

    const sd_vid_gen_params_t api_params =
        request_values.to_sd_vid_gen_params_t();
    GGML_ASSERT(api_params.reference_images != nullptr);
    GGML_ASSERT(api_params.reference_images_count == 1);
    GGML_ASSERT(std::abs(api_params.reference_attention_strength - 0.25f) <
                1e-6f);
    GGML_ASSERT(std::abs(api_params.reference_downscale_factor - 1.f) <
                1e-6f);

    SDGenerationParams request_out_of_range;
    GGML_ASSERT(request_out_of_range.from_json_str(
        R"({"reference_attention_strength":5.0,"reference_downscale_factor":1.0})"));
    add_reference_payload(&request_out_of_range);
    GGML_ASSERT(!request_out_of_range.validate(VID_GEN));

    // Models the server copying an invalid command-line default before a
    // request supplies only ref_images.
    SDGenerationParams inherited_invalid_default;
    inherited_invalid_default.reference_attention_strength = 5.f;
    add_reference_payload(&inherited_invalid_default);
    GGML_ASSERT(!inherited_invalid_default.validate(VID_GEN));

    SDGenerationParams inherited_nan_default;
    inherited_nan_default.reference_attention_strength =
        std::numeric_limits<float>::quiet_NaN();
    add_reference_payload(&inherited_nan_default);
    GGML_ASSERT(!inherited_nan_default.validate(VID_GEN));

    SDGenerationParams inherited_infinite_downscale;
    inherited_infinite_downscale.reference_downscale_factor =
        std::numeric_limits<float>::infinity();
    add_reference_payload(&inherited_infinite_downscale);
    GGML_ASSERT(!inherited_infinite_downscale.validate(VID_GEN));

    SDGenerationParams request_unsupported_downscale;
    GGML_ASSERT(request_unsupported_downscale.from_json_str(
        R"({"reference_attention_strength":0.5,"reference_downscale_factor":0.5})"));
    add_reference_payload(&request_unsupported_downscale);
    GGML_ASSERT(!request_unsupported_downscale.validate(VID_GEN));

    // CLI paths are validated after they are loaded into the same owned
    // ref_images payload. Valid values remain accepted.
    SDGenerationParams cli_values;
    cli_values.ref_image_paths.push_back("reference.png");
    cli_values.reference_attention_strength = 0.5f;
    cli_values.reference_downscale_factor   = 1.f;
    add_reference_payload(&cli_values);
    GGML_ASSERT(cli_values.validate(VID_GEN));
    return 0;
}
