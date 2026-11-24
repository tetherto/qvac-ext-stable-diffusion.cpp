#include "core/view_pinning.h"

const ggml_tensor* find_view_root(const ggml_tensor* tensor) {
    while (tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

bool is_computed_view(const ggml_tensor* tensor) {
    return tensor->view_src != nullptr && find_view_root(tensor)->op != GGML_OP_NONE;
}

static ggml_backend_t find_view_owner(const ggml_tensor* view,
                                      ggml_backend_t unpinned_root_backend,
                                      const PinnedNodeBackends& pinned_nodes) {
    auto root = pinned_nodes.find(find_view_root(view));
    return root == pinned_nodes.end() ? unpinned_root_backend : root->second;
}

ggml_backend_t select_pin_backend(const ggml_tensor* node,
                                  ggml_backend_t current_backend,
                                  ggml_backend_t fallback_backend,
                                  const PinnedNodeBackends& pinned_nodes) {
    if (!is_computed_view(node)) {
        return current_backend;
    }
    ggml_backend_t unpinned_root_backend = fallback_backend == nullptr ? current_backend : fallback_backend;
    return find_view_owner(node, unpinned_root_backend, pinned_nodes);
}
