#ifndef __SD_CORE_VIEW_PINNING_H__
#define __SD_CORE_VIEW_PINNING_H__

#include <functional>
#include <unordered_map>

#include "ggml-backend.h"
#include "ggml.h"

using PinnedNodeBackends = std::unordered_map<const ggml_tensor*, ggml_backend_t>;
using NodeSupportCheck   = std::function<bool(ggml_backend_t, const ggml_tensor*)>;
using NodeAssignment     = std::function<void(ggml_tensor*, ggml_backend_t)>;

ggml_tensor* find_view_root(ggml_tensor* tensor);

const ggml_tensor* find_view_root(const ggml_tensor* tensor);

bool is_computed_view(const ggml_tensor* tensor);

class NodePinner {
public:
    NodePinner(ggml_backend_t fallback_backend, NodeSupportCheck supports, NodeAssignment assign);

    void pin(ggml_tensor* node, ggml_backend_t current_backend);

private:
    bool pin_if_supported(ggml_tensor* node, ggml_backend_t backend);
    void pin_unowned_root(ggml_tensor* root);
    ggml_backend_t find_owner(const ggml_tensor* root) const;
    bool is_placed_by_scheduler(const ggml_tensor* root) const;
    ggml_backend_t select_backend(ggml_tensor* node, ggml_backend_t current_backend);

    ggml_backend_t fallback_backend_;
    NodeSupportCheck supports_;
    NodeAssignment assign_;
    PinnedNodeBackends pinned_nodes_;
};

#endif  // __SD_CORE_VIEW_PINNING_H__
