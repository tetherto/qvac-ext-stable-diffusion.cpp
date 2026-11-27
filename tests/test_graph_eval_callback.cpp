#include <array>
#include <cstdio>
#include <vector>

#include "core/ggml_extend_backend.h"

static constexpr size_t kGraphSize = 16;
static constexpr std::array<float, 4> kFactors = {2.f, 3.f, 5.f, 7.f};
static constexpr std::array<float, 4> kExpected = {2.f, 6.f, 30.f, 210.f};

struct CallbackState {
    std::vector<ggml_tensor*> nodes;
    std::vector<int> asked;
    std::vector<int> observed;
    unsigned mask = 0;
    int stop = -1;
};

static int node_index(const CallbackState& state, ggml_tensor* tensor) {
    for (size_t i = 0; i < state.nodes.size(); ++i) {
        if (state.nodes[i] == tensor) {
            return static_cast<int>(i);
        }
    }
    GGML_ABORT("Unexpected callback tensor");
}

static bool observe(ggml_tensor* tensor, bool ask, void* data) {
    auto& state = *static_cast<CallbackState*>(data);
    const int index = node_index(state, tensor);
    if (ask) {
        state.asked.push_back(index);
        return (state.mask & (1u << index)) != 0;
    }
    float value = 0.f;
    ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    GGML_ASSERT(value == kExpected[index]);
    state.observed.push_back(index);
    return index != state.stop;
}

static ggml_tensor* build_chain(ggml_context* ctx, ggml_tensor* input, CallbackState& state) {
    ggml_tensor* current = input;
    for (float factor : kFactors) {
        current = ggml_scale_inplace(ctx, current, factor);
        state.nodes.push_back(current);
    }
    return current;
}

static void check_callbacks(const CallbackState& state, int last) {
    std::vector<int> expected_asked;
    std::vector<int> expected_observed;
    for (int i = 0; i <= last; ++i) {
        expected_asked.push_back(i);
        if (state.mask & (1u << i)) {
            expected_observed.push_back(i);
        }
    }
    GGML_ASSERT(state.asked == expected_asked);
    GGML_ASSERT(state.observed == expected_observed);
}

static void run_case(ggml_backend_t backend, unsigned mask, int stop, bool callback) {
    const size_t context_size = ggml_graph_overhead_custom(kGraphSize, false) +
                                (kFactors.size() + 1) * ggml_tensor_overhead();
    ggml_context* ctx = ggml_init({context_size, nullptr, true});
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    CallbackState state;
    state.mask = mask;
    state.stop = stop;
    ggml_tensor* output = build_chain(ctx, input, state);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    const float initial = 1.f;
    ggml_backend_tensor_set(input, &initial, 0, sizeof(initial));
    const ggml_status status = sd_backend_graph_compute_with_eval_callback(
        backend, graph, callback ? observe : nullptr, &state);
    GGML_ASSERT(status == (stop < 0 ? GGML_STATUS_SUCCESS : GGML_STATUS_ABORTED));
    const int last = stop < 0 ? static_cast<int>(kFactors.size()) - 1 : stop;
    float value = 0.f;
    ggml_backend_tensor_get(input, &value, 0, sizeof(value));
    GGML_ASSERT(value == kExpected[last]);
    if (callback) {
        check_callbacks(state, last);
    } else {
        GGML_ASSERT(state.asked.empty() && state.observed.empty());
    }
    GGML_ASSERT(ggml_graph_n_nodes(graph) == static_cast<int>(kFactors.size()));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void run_empty(ggml_backend_t backend) {
    ggml_context* ctx = ggml_init({ggml_graph_overhead_custom(kGraphSize, false), nullptr, true});
    GGML_ASSERT(ctx != nullptr);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, kGraphSize, false);
    CallbackState state;
    GGML_ASSERT(sd_backend_graph_compute_with_eval_callback(backend, graph, observe, &state) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(state.asked.empty() && state.observed.empty());
    ggml_free(ctx);
}

static void run_cases(ggml_backend_t backend) {
    const unsigned all_nodes = (1u << kFactors.size()) - 1;
    run_case(backend, 0, -1, false);
    for (unsigned mask = 0; mask <= all_nodes; ++mask) {
        run_case(backend, mask, -1, true);
    }
    for (int stop = 0; stop < static_cast<int>(kFactors.size()); ++stop) {
        run_case(backend, all_nodes, stop, true);
    }
}

static ggml_backend_t init_test_backend(const char* name) {
    ggml_backend_reg_t registry = ggml_backend_reg_by_name(name);
    if (registry != nullptr && ggml_backend_reg_dev_count(registry) > 0) {
        return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
    }
    return ggml_backend_init_by_name(name, nullptr);
}

int main(int argc, char** argv) {
    ggml_backend_load_all();
    ggml_backend_t backend = argc > 1 ? init_test_backend(argv[1]) : sd_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    run_empty(backend);
    run_cases(backend);
    ggml_backend_free(backend);
    std::puts("Graph evaluation callback tests passed");
    return 0;
}
