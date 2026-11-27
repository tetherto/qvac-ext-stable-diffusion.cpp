#include "core/view_pinning.h"

#include <utility>

ggml_tensor* find_view_root(ggml_tensor* tensor) {
    while (tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

const ggml_tensor* find_view_root(const ggml_tensor* tensor) {
    return find_view_root(const_cast<ggml_tensor*>(tensor));
}

bool is_computed_view(const ggml_tensor* tensor) {
    return tensor->view_src != nullptr && find_view_root(tensor)->op != GGML_OP_NONE;
}

NodePinner::NodePinner(ggml_backend_t fallback_backend, NodeSupportCheck supports, NodeAssignment assign)
    : fallback_backend_(fallback_backend), supports_(std::move(supports)), assign_(std::move(assign)) {}

void NodePinner::pin(ggml_tensor* node, ggml_backend_t current_backend) {
    ggml_backend_t target = select_backend(node, current_backend);
    if (target != nullptr) {
        pin_if_supported(node, target);
    }
}

bool NodePinner::pin_if_supported(ggml_tensor* node, ggml_backend_t backend) {
    if (!supports_(backend, node)) {
        return false;
    }
    assign_(node, backend);
    pinned_nodes_[node] = backend;
    return true;
}

ggml_backend_t NodePinner::find_owner(const ggml_tensor* root) const {
    auto owner = pinned_nodes_.find(root);
    return owner == pinned_nodes_.end() ? nullptr : owner->second;
}

void NodePinner::pin_unowned_root(ggml_tensor* root) {
    if (fallback_backend_ == nullptr || root->buffer != nullptr || find_owner(root) != nullptr) {
        return;
    }
    pin_if_supported(root, fallback_backend_);
}

bool NodePinner::is_placed_by_scheduler(const ggml_tensor* root) const {
    return root->buffer != nullptr || fallback_backend_ != nullptr;
}

ggml_backend_t NodePinner::select_backend(ggml_tensor* node, ggml_backend_t current_backend) {
    if (!is_computed_view(node)) {
        return current_backend;
    }
    ggml_tensor* root = find_view_root(node);
    pin_unowned_root(root);
    ggml_backend_t owner = find_owner(root);
    if (owner != nullptr) {
        return owner;
    }
    return is_placed_by_scheduler(root) ? nullptr : current_backend;
}
