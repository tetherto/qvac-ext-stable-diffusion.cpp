#include "core/graph_signature.h"

#include <cstring>
#include <unordered_map>

namespace sd {
    namespace {
        constexpr uint64_t HASH_OFFSET     = 14695981039346656037ULL;
        constexpr uint64_t HASH_PRIME      = 1099511628211ULL;
        constexpr uint64_t UNKNOWN_TENSOR  = ~0ULL;
        constexpr uint64_t MISSING_TENSOR  = ~0ULL - 1;
        constexpr int PARAM_WORDS          = static_cast<int>(sizeof(ggml_tensor::op_params) / sizeof(int32_t));

        using TensorPositions = std::unordered_map<const ggml_tensor*, uint64_t>;

        struct Hasher {
            uint64_t state = HASH_OFFSET;

            void add(uint64_t value) {
                state = (state ^ value) * HASH_PRIME;
            }
        };

        TensorPositions index_tensors(ggml_cgraph* graph) {
            TensorPositions positions;
            const int node_count = ggml_graph_n_nodes(graph);
            const int leaf_count = ggml_graph_n_leafs(graph);
            positions.reserve(static_cast<size_t>(node_count + leaf_count));
            for (int i = 0; i < node_count; ++i) {
                positions.emplace(ggml_graph_node(graph, i), static_cast<uint64_t>(i));
            }
            for (int i = 0; i < leaf_count; ++i) {
                positions.emplace(ggml_graph_leaf(graph, i), static_cast<uint64_t>(node_count + i));
            }
            return positions;
        }

        uint64_t position_of(const TensorPositions& positions, const ggml_tensor* tensor) {
            if (tensor == nullptr) {
                return MISSING_TENSOR;
            }
            auto found = positions.find(tensor);
            return found == positions.end() ? UNKNOWN_TENSOR : found->second;
        }

        void add_shape(Hasher& hasher, const ggml_tensor* tensor) {
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                hasher.add(static_cast<uint64_t>(tensor->ne[i]));
                hasher.add(static_cast<uint64_t>(tensor->nb[i]));
            }
        }

        void add_params(Hasher& hasher, const ggml_tensor* tensor) {
            int32_t words[PARAM_WORDS];
            std::memcpy(words, tensor->op_params, sizeof(words));
            for (int i = 0; i < PARAM_WORDS; ++i) {
                hasher.add(static_cast<uint64_t>(static_cast<uint32_t>(words[i])));
            }
        }

        void add_sources(Hasher& hasher, const TensorPositions& positions, const ggml_tensor* tensor) {
            hasher.add(position_of(positions, tensor->view_src));
            hasher.add(static_cast<uint64_t>(tensor->view_offs));
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                hasher.add(position_of(positions, tensor->src[i]));
            }
        }

        void add_tensor(Hasher& hasher, const TensorPositions& positions, const ggml_tensor* tensor, const TensorPredicate& is_external) {
            hasher.add(static_cast<uint64_t>(tensor->op));
            hasher.add(static_cast<uint64_t>(tensor->type));
            hasher.add(static_cast<uint64_t>(tensor->flags));
            hasher.add(is_external(tensor) ? 1 : 0);
            add_shape(hasher, tensor);
            add_params(hasher, tensor);
            add_sources(hasher, positions, tensor);
        }
    }

    GraphSignature graph_signature(ggml_cgraph* graph, const TensorPredicate& is_external) {
        const TensorPositions positions = index_tensors(graph);
        const int node_count            = ggml_graph_n_nodes(graph);
        const int leaf_count            = ggml_graph_n_leafs(graph);
        Hasher hasher;
        hasher.add(static_cast<uint64_t>(node_count));
        hasher.add(static_cast<uint64_t>(leaf_count));
        for (int i = 0; i < node_count; ++i) {
            add_tensor(hasher, positions, ggml_graph_node(graph, i), is_external);
        }
        for (int i = 0; i < leaf_count; ++i) {
            add_tensor(hasher, positions, ggml_graph_leaf(graph, i), is_external);
        }
        return hasher.state;
    }
}
