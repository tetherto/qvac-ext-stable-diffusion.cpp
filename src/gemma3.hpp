#ifndef __GEMMA3_HPP__
#define __GEMMA3_HPP__

#include "ggml_extend.hpp"
#include "rope.hpp"

// Gemma-3 decoder-only text encoder for LTX-2 conditioning.
//
// This is a self-contained Gemma-3 implementation (separate from the shared
// LLM path in llm.hpp) so the LTX-2 text encoder can evolve without perturbing
// Qwen/Mistral/Z-Image. It implements the Gemma-3 specifics: (1+weight)
// RMSNorm, GeGLU MLP, q/k RMSNorm, four per-layer norms, scaled token
// embeddings, and sliding-window vs global attention layers.
//
// The hidden states of every decoder layer are returned stacked on a new axis
// so the LTX multi-layer feature extractor can aggregate them.
//
// NOTE: RoPE theta, the sliding-window pattern, and the query scaling follow
// the public Gemma-3 config; validate numerically against the reference before
// claiming parity (M3).
namespace GEMMA3 {

    struct Gemma3Params {
        int64_t num_layers        = 48;    // gemma-3-12b
        int64_t hidden_size       = 3840;
        int64_t intermediate_size = 15360;
        int num_heads             = 16;
        int num_kv_heads          = 8;
        int head_dim              = 256;
        int64_t vocab_size        = 262208;
        float rms_norm_eps        = 1e-6f;
        int sliding_window        = 1024;
        int sliding_window_pattern = 6;     // every 6th layer is global
        float rope_theta_local    = 10000.0f;
        float rope_theta_global   = 1000000.0f;
    };

    // Gemma RMSNorm: y = x / rms(x) * (1 + weight)
    class GemmaRMSNorm : public UnaryBlock {
    protected:
        int64_t dim;
        float eps;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, dim);
        }

    public:
        GemmaRMSNorm(int64_t dim, float eps = 1e-6f)
            : dim(dim), eps(eps) {}

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            auto w = params["weight"];
            x      = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            // (1 + weight) * x
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, w));
            return x;
        }
    };

    // GeGLU MLP: down(gelu(gate(x)) * up(x))
    class GemmaMLP : public GGMLBlock {
    public:
        GemmaMLP(int64_t hidden_size, int64_t intermediate_size) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, false));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, false));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, false));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto gate = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);
            auto h    = ggml_ext_gelu(ctx->ggml_ctx, gate->forward(ctx, x), true);
            h         = ggml_mul(ctx->ggml_ctx, h, up->forward(ctx, x));
            return down->forward(ctx, h);
        }
    };

    class Gemma3Attention : public GGMLBlock {
    protected:
        int head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        float eps;
        float rope_theta;

    public:
        Gemma3Attention(const Gemma3Params& p, float rope_theta)
            : head_dim(p.head_dim), num_heads(p.num_heads), num_kv_heads(p.num_kv_heads), eps(p.rms_norm_eps), rope_theta(rope_theta) {
            blocks["q_proj"] = std::make_shared<Linear>(p.hidden_size, num_heads * head_dim, false);
            blocks["k_proj"] = std::make_shared<Linear>(p.hidden_size, num_kv_heads * head_dim, false);
            blocks["v_proj"] = std::make_shared<Linear>(p.hidden_size, num_kv_heads * head_dim, false);
            blocks["o_proj"] = std::make_shared<Linear>(num_heads * head_dim, p.hidden_size, false);
            blocks["q_norm"] = std::make_shared<GemmaRMSNorm>(head_dim, eps);
            blocks["k_norm"] = std::make_shared<GemmaRMSNorm>(head_dim, eps);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask) {
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto o_proj     = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);
            auto q_norm     = std::dynamic_pointer_cast<GemmaRMSNorm>(blocks["q_norm"]);
            auto k_norm     = std::dynamic_pointer_cast<GemmaRMSNorm>(blocks["k_norm"]);

            auto q = q_proj->forward(ctx, x);
            auto k = k_proj->forward(ctx, x);
            auto v = v_proj->forward(ctx, x);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.f, 0.f, 1.f, 0.f, 0.f);
            k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.f, 0.f, 1.f, 0.f, 0.f);

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);
            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, attention_mask, true, false);
            x = o_proj->forward(ctx, x);
            return x;
        }
    };

    class Gemma3Block : public GGMLBlock {
    public:
        Gemma3Block(const Gemma3Params& p, float rope_theta) {
            blocks["self_attn"]                 = std::make_shared<Gemma3Attention>(p, rope_theta);
            blocks["mlp"]                       = std::make_shared<GemmaMLP>(p.hidden_size, p.intermediate_size);
            blocks["input_layernorm"]           = std::make_shared<GemmaRMSNorm>(p.hidden_size, p.rms_norm_eps);
            blocks["post_attention_layernorm"]  = std::make_shared<GemmaRMSNorm>(p.hidden_size, p.rms_norm_eps);
            blocks["pre_feedforward_layernorm"] = std::make_shared<GemmaRMSNorm>(p.hidden_size, p.rms_norm_eps);
            blocks["post_feedforward_layernorm"]= std::make_shared<GemmaRMSNorm>(p.hidden_size, p.rms_norm_eps);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask) {
            auto self_attn = std::dynamic_pointer_cast<Gemma3Attention>(blocks["self_attn"]);
            auto mlp       = std::dynamic_pointer_cast<GemmaMLP>(blocks["mlp"]);
            auto in_ln     = std::dynamic_pointer_cast<GemmaRMSNorm>(blocks["input_layernorm"]);
            auto post_attn = std::dynamic_pointer_cast<GemmaRMSNorm>(blocks["post_attention_layernorm"]);
            auto pre_ff    = std::dynamic_pointer_cast<GemmaRMSNorm>(blocks["pre_feedforward_layernorm"]);
            auto post_ff   = std::dynamic_pointer_cast<GemmaRMSNorm>(blocks["post_feedforward_layernorm"]);
            auto ctxg      = ctx->ggml_ctx;

            auto residual = x;
            x             = in_ln->forward(ctx, x);
            x             = self_attn->forward(ctx, x, input_pos, attention_mask);
            x             = post_attn->forward(ctx, x);
            x             = ggml_add(ctxg, x, residual);

            residual = x;
            x        = pre_ff->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = post_ff->forward(ctx, x);
            x        = ggml_add(ctxg, x, residual);
            return x;
        }
    };

    class Gemma3Model : public GGMLBlock {
    protected:
        Gemma3Params p;

    public:
        Gemma3Model(const Gemma3Params& p)
            : p(p) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(p.vocab_size, p.hidden_size));
            for (int i = 0; i < p.num_layers; i++) {
                bool is_global   = ((i + 1) % p.sliding_window_pattern) == 0;
                float rope_theta = is_global ? p.rope_theta_global : p.rope_theta_local;
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Gemma3Block(p, rope_theta));
            }
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new GemmaRMSNorm(p.hidden_size, p.rms_norm_eps));
        }

        // input_ids: [n_token]
        // returns: [hidden, n_token, num_layers] stacked all-layer hidden states
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* mask_local,
                                    struct ggml_tensor* mask_global) {
            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto ctxg         = ctx->ggml_ctx;

            auto x = embed_tokens->forward(ctx, input_ids);  // [hidden, n_token, 1]
            // Gemma scales the embeddings by sqrt(hidden_size)
            x = ggml_scale(ctxg, x, sqrtf(static_cast<float>(p.hidden_size)));

            struct ggml_tensor* stacked = nullptr;
            for (int i = 0; i < p.num_layers; i++) {
                auto block       = std::dynamic_pointer_cast<Gemma3Block>(blocks["layers." + std::to_string(i)]);
                bool is_global   = ((i + 1) % p.sliding_window_pattern) == 0;
                auto mask        = is_global ? mask_global : mask_local;
                x                = block->forward(ctx, x, input_pos, mask);

                auto layer_out = ggml_reshape_3d(ctxg, x, x->ne[0], x->ne[1], 1);  // [hidden, n_token, 1]
                stacked        = (stacked == nullptr) ? layer_out : ggml_concat(ctxg, stacked, layer_out, 2);
            }
            return stacked;  // [hidden, n_token, num_layers]
        }
    };

    struct Gemma3Runner : public GGMLRunner {
        Gemma3Params params;
        Gemma3Model model;
        std::vector<int> input_pos_vec;
        std::vector<float> mask_local_vec;
        std::vector<float> mask_global_vec;

        Gemma3Runner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "text_encoders.gemma3")
            : GGMLRunner(backend, offload_params_to_cpu),
              params(infer_params(tensor_storage_map, prefix)),
              model(params) {
            LOG_INFO("Gemma-3 encoder: num_layers=%lld hidden=%lld heads=%d kv_heads=%d head_dim=%d",
                     (long long)params.num_layers, (long long)params.hidden_size, params.num_heads, params.num_kv_heads, params.head_dim);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        static Gemma3Params infer_params(const String2TensorStorage& tsm, const std::string& prefix) {
            Gemma3Params p;
            std::string base = prefix.empty() ? "" : prefix + ".";
            int max_layer    = -1;
            for (auto& pair : tsm) {
                const std::string& n = pair.first;
                if (n.compare(0, base.size(), base) != 0) {
                    continue;
                }
                std::string rel = n.substr(base.size());
                size_t pos      = rel.find("layers.");
                if (pos != std::string::npos) {
                    max_layer = std::max(max_layer, atoi(rel.c_str() + pos + 7));
                }
                if (rel == "embed_tokens.weight") {
                    p.hidden_size = pair.second.ne[0];
                    p.vocab_size  = pair.second.ne[1];
                }
                if (rel == "layers.0.mlp.gate_proj.weight") {
                    p.intermediate_size = pair.second.ne[1];
                }
                if (rel == "layers.0.self_attn.q_proj.weight") {
                    // ne[1] = num_heads*head_dim
                }
                if (rel == "layers.0.self_attn.q_norm.weight") {
                    p.head_dim = (int)pair.second.ne[0];
                }
            }
            if (max_layer >= 0) {
                p.num_layers = max_layer + 1;
            }
            // recompute num_heads from q_proj if possible
            const auto qit = tsm.find(base + "layers.0.self_attn.q_proj.weight");
            const auto kit = tsm.find(base + "layers.0.self_attn.k_proj.weight");
            if (qit != tsm.end() && p.head_dim > 0) {
                p.num_heads = (int)(qit->second.ne[1] / p.head_dim);
            }
            if (kit != tsm.end() && p.head_dim > 0) {
                p.num_kv_heads = (int)(kit->second.ne[1] / p.head_dim);
            }
            return p;
        }

        std::string get_desc() override {
            return "gemma3";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids) {
            struct ggml_cgraph* gf = new_graph_custom(MAX_GRAPH_SIZE / 2);
            input_ids              = to_backend(input_ids);

            int64_t n_tokens = input_ids->ne[0];
            input_pos_vec.resize(n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                input_pos_vec[i] = i;
            }
            auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_tokens);
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            // Determine which mask types are actually referenced so we never
            // register data for a tensor that is absent from the graph.
            bool needs_global = false;
            bool needs_local  = false;
            for (int i = 0; i < params.num_layers; i++) {
                bool is_global = ((i + 1) % params.sliding_window_pattern) == 0;
                needs_global |= is_global;
                needs_local |= !is_global;
            }

            // causal mask (global) + sliding-window causal mask (local)
            struct ggml_tensor* mask_global = nullptr;
            struct ggml_tensor* mask_local  = nullptr;
            if (needs_global) {
                mask_global_vec.resize(n_tokens * n_tokens);
            }
            if (needs_local) {
                mask_local_vec.resize(n_tokens * n_tokens);
            }
            for (int64_t q = 0; q < n_tokens; q++) {
                for (int64_t k = 0; k < n_tokens; k++) {
                    bool causal_ok = k <= q;
                    bool window_ok = causal_ok && (q - k) < params.sliding_window;
                    if (needs_global) {
                        mask_global_vec[q * n_tokens + k] = causal_ok ? 0.f : -INFINITY;
                    }
                    if (needs_local) {
                        mask_local_vec[q * n_tokens + k] = window_ok ? 0.f : -INFINITY;
                    }
                }
            }
            if (needs_global) {
                mask_global = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(mask_global, mask_global_vec.data());
            }
            if (needs_local) {
                mask_local = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(mask_local, mask_local_vec.data());
            }

            auto runner_ctx = get_context();
            auto out        = model.forward(&runner_ctx, input_ids, input_pos, mask_local, mask_global);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* input_ids,
                     struct ggml_tensor** output,
                     struct ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(input_ids);
            };
            return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }
    };

}  // namespace GEMMA3

#endif  // __GEMMA3_HPP__
