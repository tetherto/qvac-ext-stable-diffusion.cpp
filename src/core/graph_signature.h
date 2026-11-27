#ifndef __SD_CORE_GRAPH_SIGNATURE_H__
#define __SD_CORE_GRAPH_SIGNATURE_H__

#include <cstdint>
#include <functional>

#include "ggml.h"

namespace sd {
    using GraphSignature  = uint64_t;
    using TensorPredicate = std::function<bool(const ggml_tensor*)>;

    GraphSignature graph_signature(ggml_cgraph* graph, const TensorPredicate& is_external);
}

#endif
