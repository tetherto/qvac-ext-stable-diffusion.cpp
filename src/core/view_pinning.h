#ifndef __SD_CORE_VIEW_PINNING_H__
#define __SD_CORE_VIEW_PINNING_H__

#include <unordered_map>

#include "ggml-backend.h"
#include "ggml.h"

using PinnedNodeBackends = std::unordered_map<const ggml_tensor*, ggml_backend_t>;

const ggml_tensor* find_view_root(const ggml_tensor* tensor);

bool is_computed_view(const ggml_tensor* tensor);

ggml_backend_t select_pin_backend(const ggml_tensor* node,
                                  ggml_backend_t current_backend,
                                  ggml_backend_t fallback_backend,
                                  const PinnedNodeBackends& pinned_nodes);

#endif  // __SD_CORE_VIEW_PINNING_H__
