#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using coordinates = std::array<int64_t, 4>;

static constexpr size_t context_bytes  = 4 * 1024 * 1024;
static constexpr int skip_status       = 77;
static constexpr float sentinel        = -1234.0f;
static constexpr double rope_tolerance = 1e-5;

struct Fixture {
    ggml_context* context        = ggml_init({context_bytes, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;

    Fixture() {
        if (!context) {
            throw std::runtime_error("context allocation failed");
        }
    }

    ~Fixture() {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }
};

static coordinates unravel(int64_t index, const int64_t* shape) {
    coordinates result = {};
    for (size_t axis = 0; axis < result.size(); ++axis) {
        result[axis] = index % shape[axis];
        index /= shape[axis];
    }
    return result;
}

static size_t element_offset(const ggml_tensor* tensor, const coordinates& position) {
    size_t offset = tensor->view_offs;
    for (size_t axis = 0; axis < position.size(); ++axis) {
        offset += position[axis] * tensor->nb[axis];
    }
    return offset / sizeof(float);
}

static float input_value(const coordinates& position) {
    return static_cast<float>(position[0] + 11 * position[1] + 101 * position[2] + 1009 * position[3] - 29) / 32.0f;
}

static std::vector<float> make_input(const ggml_tensor* storage, const ggml_tensor* input) {
    std::vector<float> data(ggml_nelements(storage), sentinel);
    for (int64_t index = 0; index < ggml_nelements(input); ++index) {
        const auto position                      = unravel(index, input->ne);
        data.at(element_offset(input, position)) = input_value(position);
    }
    return data;
}

static double rotation_angle(int64_t pair, int64_t token) {
    return 0.17 * (pair + 1) + 0.11 * token;
}

static float rotation_value(const coordinates& position) {
    const double angle = rotation_angle(position[2], position[3]);
    if (position[0] == position[1]) {
        return static_cast<float>(std::cos(angle));
    }
    return static_cast<float>((position[0] == 0 ? 1 : -1) * std::sin(angle));
}

static std::vector<float> make_rotation(const ggml_tensor* storage, const ggml_tensor* rotation) {
    std::vector<float> data(ggml_nelements(storage), sentinel);
    for (int64_t index = 0; index < ggml_nelements(rotation); ++index) {
        const auto position                         = unravel(index, rotation->ne);
        data.at(element_offset(rotation, position)) = rotation_value(position);
    }
    return data;
}

static ggml_cgraph* allocate_graph(Fixture& test, ggml_backend_t backend, ggml_tensor* output) {
    if (!ggml_backend_supports_op(backend, output)) {
        throw std::runtime_error(std::string("required native operation unsupported: ") + ggml_op_name(output->op));
    }
    ggml_cgraph* graph = ggml_new_graph(test.context);
    ggml_build_forward_expand(graph, output);
    test.buffer = ggml_backend_alloc_ctx_tensors(test.context, backend);
    if (!test.buffer) {
        throw std::runtime_error("backend allocation failed");
    }
    return graph;
}

static std::vector<float> read_output(ggml_tensor* output) {
    std::vector<float> values(ggml_nelements(output));
    if (output->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
        return values;
    }
    std::vector<ggml_fp16_t> halves(values.size());
    ggml_backend_tensor_get(output, halves.data(), 0, ggml_nbytes(output));
    ggml_fp16_to_fp32_row(halves.data(), values.data(), static_cast<int64_t>(values.size()));
    return values;
}

static bool compare_values(const char* name, const std::vector<float>& actual, const std::vector<float>& expected, double tolerance) {
    bool ok              = actual.size() == expected.size();
    double maximum_error = 0;
    for (size_t index = 0; index < actual.size() && index < expected.size(); ++index) {
        const double error = std::abs(static_cast<double>(actual[index]) - expected[index]);
        maximum_error      = std::max(maximum_error, error);
        ok &= std::isfinite(actual[index]) && error <= tolerance * (1 + std::abs(expected[index]));
    }
    std::printf("%s: %s, values=%zu, max_abs_error=%.9g\n", name, ok ? "PASS" : "FAIL", actual.size(), maximum_error);
    return ok;
}

static bool compute_and_compare(ggml_backend_t backend, ggml_cgraph* graph, ggml_tensor* output, const char* name, const std::vector<float>& expected, double tolerance) {
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("graph execution failed");
    }
    return compare_values(name, read_output(output), expected, tolerance);
}

static float expected_rope_value(const coordinates& position, int64_t heads, bool rotate) {
    coordinates input = {position[0], position[2] % heads, position[1], position[2] / heads};
    if (!rotate) {
        return input_value(input);
    }
    const bool odd = input[0] % 2 != 0;
    input[0] -= odd;
    const double even_value = input_value(input);
    ++input[0];
    const double odd_value = input_value(input);
    const double cosine    = static_cast<float>(std::cos(rotation_angle(input[0] / 2, input[2])));
    const double sine      = static_cast<float>(std::sin(rotation_angle(input[0] / 2, input[2])));
    return static_cast<float>(odd ? even_value * sine + odd_value * cosine : even_value * cosine - odd_value * sine);
}

static std::vector<float> expected_rope(const ggml_tensor* output, int64_t heads, bool rotate) {
    std::vector<float> expected(ggml_nelements(output));
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = expected_rope_value(unravel(index, output->ne), heads, rotate);
    }
    return expected;
}

static bool test_rope(ggml_backend_t backend, bool strided, bool rotate) {
    Fixture test;
    constexpr int64_t width = 8, heads = 3, tokens = 5, batches = 2;
    ggml_tensor* storage = ggml_new_tensor_4d(test.context, GGML_TYPE_F32,
                                              strided ? heads + 2 : width, strided ? width : heads, tokens + (strided ? 1 : 0), batches);
    ggml_tensor* input   = ggml_view_4d(test.context, storage,
                                      strided ? heads : width, strided ? width : heads, tokens, batches,
                                        storage->nb[1], storage->nb[2], storage->nb[3], strided ? sizeof(float) : 0);
    if (strided) {
        input = ggml_permute(test.context, input, 1, 0, 2, 3);
    }
    ggml_tensor* positions = ggml_new_tensor_4d(test.context, GGML_TYPE_F32, 4, 3, width / 2 + 1, tokens);
    ggml_tensor* rotation  = ggml_view_4d(test.context, positions, 2, 2, width / 2, tokens,
                                          positions->nb[1], positions->nb[2], positions->nb[3], sizeof(float));
    ggml_tensor* output    = ggml_rope_flux(test.context, input, rotate ? rotation : nullptr);
    if (output->ne[0] != width || output->ne[1] != tokens || output->ne[2] != heads * batches || output->ne[3] != 1) {
        throw std::runtime_error("Flux RoPE output shape changed");
    }
    ggml_cgraph* graph = allocate_graph(test, backend, output);
    const auto data    = make_input(storage, input);
    ggml_backend_tensor_set(storage, data.data(), 0, ggml_nbytes(storage));
    const auto angles = make_rotation(positions, rotation);
    ggml_backend_tensor_set(positions, angles.data(), 0, ggml_nbytes(positions));
    const std::string name = std::string("rope/") + (strided ? "strided/" : "contiguous/") + (rotate ? "rotation" : "permute");
    return compute_and_compare(backend, graph, output, name.c_str(), expected_rope(output, heads, rotate), rotate ? rope_tolerance : 0);
}

static float expected_padding(const coordinates& output, const coordinates& shape, const coordinates& left) {
    coordinates input = {};
    for (size_t axis = 0; axis < input.size(); ++axis) {
        input[axis] = output[axis] - left[axis];
        if (input[axis] < 0 || input[axis] >= shape[axis]) {
            return 0;
        }
    }
    return input_value(input);
}

static std::vector<float> expected_padding_values(const ggml_tensor* output, const coordinates& shape, const coordinates& left) {
    std::vector<float> expected(ggml_nelements(output));
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = expected_padding(unravel(index, output->ne), shape, left);
    }
    return expected;
}

static bool test_padding(ggml_backend_t backend) {
    Fixture test;
    const coordinates shape = {5, 3, 2, 2};
    const coordinates left  = {2, 1, 1, 0};
    ggml_tensor* input      = ggml_new_tensor(test.context, GGML_TYPE_F32, 4, shape.data());
    ggml_tensor* output     = ggml_pad_ext(test.context, input, left[0], 1, left[1], 2, left[2], 0, left[3], 1);
    if (output->ne[0] != 8 || output->ne[1] != 6 || output->ne[2] != 3 || output->ne[3] != 3) {
        throw std::runtime_error("padding output shape changed");
    }
    ggml_cgraph* graph = allocate_graph(test, backend, output);
    const auto data    = make_input(input, input);
    ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
    return compute_and_compare(backend, graph, output, "padding/asymmetric-4d",
                               expected_padding_values(output, shape, left), 0);
}

static float expected_patch(const coordinates& output, const ggml_tensor* input) {
    constexpr int64_t channels = 2, kernel_width = 3, kernel_height = 2, kernel_depth = 2;
    constexpr int64_t output_depth = 5;
    const int64_t column           = output[0];
    const int64_t channel          = column / (kernel_width * kernel_height * kernel_depth);
    const coordinates pixel        = {
        output[1] * 2 - 1 + column % kernel_width,
        output[2] + 2 * (column / kernel_width % kernel_height),
        output[3] % output_depth - 1 + column / (kernel_width * kernel_height) % kernel_depth,
        output[3] / output_depth * channels + channel,
    };
    return expected_padding(pixel, {input->ne[0], input->ne[1], input->ne[2], input->ne[3]}, {0, 0, 0, 0});
}

static std::vector<float> expected_patches(const ggml_tensor* output, const ggml_tensor* input) {
    std::vector<float> expected(ggml_nelements(output));
    for (size_t index = 0; index < expected.size(); ++index) {
        float value     = expected_patch(unravel(index, output->ne), input);
        expected[index] = output->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)) : value;
    }
    return expected;
}

static bool test_im2col(ggml_backend_t backend, bool strided, ggml_type type) {
    Fixture test;
    constexpr int64_t width = 7, height = 5, depth = 4, channels = 2, batches = 2;
    ggml_tensor* storage = ggml_new_tensor_4d(test.context, GGML_TYPE_F32,
                                              width + (strided ? 2 : 0), height, depth, channels * batches);
    ggml_tensor* input   = ggml_view_4d(test.context, storage, width, height, depth, channels * batches,
                                        storage->nb[1], storage->nb[2], storage->nb[3], strided ? sizeof(float) : 0);
    ggml_tensor* kernel  = ggml_new_tensor_4d(test.context, type, 3, 2, 2, channels * 3);
    ggml_tensor* output  = ggml_im2col_3d(test.context, kernel, input, channels, 2, 1, 1, 1, 0, 1, 1, 2, 1, type);
    if (output->ne[0] != 24 || output->ne[1] != 4 || output->ne[2] != 3 || output->ne[3] != 10) {
        throw std::runtime_error("IM2COL_3D output shape changed");
    }
    ggml_cgraph* graph = allocate_graph(test, backend, output);
    const auto data    = make_input(storage, input);
    ggml_backend_tensor_set(storage, data.data(), 0, ggml_nbytes(storage));
    ggml_backend_tensor_memset(kernel, 0, 0, ggml_nbytes(kernel));
    const std::string name = std::string("im2col3d/") + (strided ? "strided/" : "contiguous/") + ggml_type_name(type);
    return compute_and_compare(backend, graph, output, name.c_str(), expected_patches(output, input), 0);
}

static bool run_cases(ggml_backend_t backend, const std::string& operation) {
    bool ok = true;
    if (operation == "rope") {
        ok &= test_rope(backend, false, false);
        ok &= test_rope(backend, false, true);
        ok &= test_rope(backend, true, false);
        ok &= test_rope(backend, true, true);
    } else if (operation == "pad") {
        ok &= test_padding(backend);
    } else if (operation == "im2col3d") {
        ok &= test_im2col(backend, false, GGML_TYPE_F32);
        ok &= test_im2col(backend, false, GGML_TYPE_F16);
        ok &= test_im2col(backend, true, GGML_TYPE_F32);
        ok &= test_im2col(backend, true, GGML_TYPE_F16);
    } else {
        throw std::runtime_error("unknown operation");
    }
    return ok;
}

static int run_test(const char* registry_name, const char* operation) {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(registry_name);
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::fprintf(stderr, "SKIP: requested backend %s is unavailable\n", registry_name);
        return skip_status;
    }
    ggml_backend_dev_t device = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t backend    = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        std::fprintf(stderr, "FAIL: requested backend initialization failed\n");
        return 1;
    }
    std::printf("backend=%s device=%s\n", registry_name, ggml_backend_dev_description(device));
    int result = 1;
    try {
        result = run_cases(backend, operation) ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
    }
    ggml_backend_free(backend);
    return result;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s CPU|CUDA|Vulkan|Metal rope|pad|im2col3d\n", argv[0]);
        return 1;
    }
    return run_test(argv[1], argv[2]);
}
