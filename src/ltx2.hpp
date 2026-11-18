#ifndef __LTX2_HPP__
#define __LTX2_HPP__

#include "common_block.hpp"
#include "rope.hpp"

// LTX-2.3 video-stream DiT (AVTransformer3DModel, video half only).
//
// M1 declared the GGMLBlock tree (parameters only) so the model could be
// loaded and bound on CPU. M2 adds the forward (denoising) pass: patchify,
// 3D RoPE, AdaLN-single modulation, gated self/cross attention, the
// video-embeddings connector and the output projection.
//
// Confirmed architecture (Lightricks/LTX-2.3, model_version 2.3.0):
//   num_layers=48, num_heads=32, head_dim=128 (dim 4096), in_channels=128,
//   caption_channels=3840, cross_attention_dim=4096, qk_norm=rms_norm,
//   gated attention (to_gate_logits), FFN 4096->16384 (gelu-approx),
//   8-layer embeddings connector with 128 learnable registers.
//
// NOTE: The exact assignment of the 9 per-block modulation channels and the
// 2 "prompt" modulation channels, the RoPE axis split, and the non-affine
// norm choices below follow the PixArt/LTX-Video AdaLN-single convention and
// should be validated numerically against the Diffusers LTX-2 reference
// before claiming PSNR/SSIM parity (M3).
namespace LTX2 {

    struct Ltx2Params {
        int num_layers           = 48;
        int num_heads            = 32;
        int head_dim             = 128;
        int dim                  = 4096;  // num_heads * head_dim
        int in_channels          = 128;
        int cross_attention_dim  = 4096;
        int ffn_dim              = 16384;
        int connector_num_layers = 8;
        int connector_registers  = 128;
        int timestep_freq_dim    = 256;
        float eps                = 1e-6f;
        float rope_theta         = 10000.0f;
    };

    // 3D RoPE axis split for (t, h, w). The three parts sum to head_dim; the
    // temporal axis gets ~1/4 of the budget, height/width split the rest.
    __STATIC_INLINE__ std::vector<int> ltx2_rope_axes_dim(int head_dim) {
        int t_dim = (head_dim / 4) & ~1;
        int rem   = head_dim - t_dim;
        int h_dim = (rem / 2) & ~1;
        int w_dim = head_dim - t_dim - h_dim;
        return {t_dim, h_dim, w_dim};
    }

    // Non-affine layer norm (no learnable weight/bias), matching LTX/PixArt
    // AdaLN blocks where the modulation supplies the scale/shift.
    __STATIC_INLINE__ struct ggml_tensor* ltx2_norm(struct ggml_context* ctx, struct ggml_tensor* x, float eps) {
        return ggml_norm(ctx, x, eps);
    }

    // x: [dim, n_token, N], scale/shift: [dim, 1, 1] -> x * (1 + scale) + shift
    __STATIC_INLINE__ struct ggml_tensor* ltx2_modulate(struct ggml_context* ctx,
                                                        struct ggml_tensor* x,
                                                        struct ggml_tensor* scale,
                                                        struct ggml_tensor* shift) {
        x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x = ggml_add(ctx, x, shift);
        return x;
    }

    // q/k/v + gated output projection with rms qk-norm; matches the LTX-2
    // `attn1`/`attn2` layout (to_q, to_k, to_v, to_out.0, q_norm, k_norm,
    // to_gate_logits).
    class GatedAttention : public GGMLBlock {
    protected:
        int num_heads;
        int head_dim;
        bool use_rope;

    public:
        GatedAttention(int dim, int ctx_dim, int num_heads, float eps, bool use_rope)
            : num_heads(num_heads), head_dim(dim / num_heads), use_rope(use_rope) {
            blocks["to_q"]           = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["to_k"]           = std::shared_ptr<GGMLBlock>(new Linear(ctx_dim, dim));
            blocks["to_v"]           = std::shared_ptr<GGMLBlock>(new Linear(ctx_dim, dim));
            blocks["to_out.0"]       = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["q_norm"]         = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            blocks["k_norm"]         = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            blocks["to_gate_logits"] = std::shared_ptr<GGMLBlock>(new Linear(dim, num_heads));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe) {
            // x:       [dim, n_token, N]
            // context: [ctx_dim, n_ctx, N] (== x for self-attention)
            // pe:      [n_token, head_dim/2, 2, 2] or nullptr
            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out   = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto to_gate  = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);

            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];

            auto q = q_norm->forward(ctx, to_q->forward(ctx, x));
            auto k = k_norm->forward(ctx, to_k->forward(ctx, context));
            auto v = to_v->forward(ctx, context);

            struct ggml_tensor* attn;
            if (use_rope && pe != nullptr) {
                q    = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);
                k    = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_token, N);
                v    = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, n_token, N);
                attn = Rope::attention(ctx, q, k, v, pe, nullptr);  // [dim, n_token, N]
            } else {
                attn = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, ctx->flash_attn_enabled);
            }

            // per-head gate: sigmoid(to_gate_logits(x)) modulates each head.
            auto gate = ggml_sigmoid(ctx->ggml_ctx, to_gate->forward(ctx, x));   // [num_heads, n_token, N]
            gate      = ggml_reshape_4d(ctx->ggml_ctx, gate, 1, num_heads, n_token, N);
            attn      = ggml_reshape_4d(ctx->ggml_ctx, attn, head_dim, num_heads, n_token, N);
            attn      = ggml_mul(ctx->ggml_ctx, attn, gate);
            attn      = ggml_reshape_3d(ctx->ggml_ctx, attn, head_dim * num_heads, n_token, N);

            return to_out->forward(ctx, attn);
        }
    };

    // gelu-approximate FFN: net.0.proj (dim->inner), net.2 (inner->dim).
    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int dim, int inner) {
            blocks["net.0.proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, inner));
            blocks["net.2"]      = std::shared_ptr<GGMLBlock>(new Linear(inner, dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["net.0.proj"]);
            auto out  = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);
            x         = proj->forward(ctx, x);
            x         = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            x         = out->forward(ctx, x);
            return x;
        }
    };

    // adaln_single / prompt_adaln_single: a timestep embedder MLP plus a final
    // projection producing the modulation table. forward returns both the
    // conditioning embedding ([dim]) and the modulation table ([out_dim]).
    class AdaLnSingle : public GGMLBlock {
    public:
        int freq_dim;

        AdaLnSingle(int freq_dim, int dim, int out_dim)
            : freq_dim(freq_dim) {
            blocks["emb.timestep_embedder.linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(freq_dim, dim));
            blocks["emb.timestep_embedder.linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["linear"]                         = std::shared_ptr<GGMLBlock>(new Linear(dim, out_dim));
        }

        // timestep: [N]
        // returns: { embedded [dim, N], modulation [out_dim, N] }
        std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(GGMLRunnerContext* ctx, struct ggml_tensor* timestep) {
            auto l1     = std::dynamic_pointer_cast<Linear>(blocks["emb.timestep_embedder.linear_1"]);
            auto l2     = std::dynamic_pointer_cast<Linear>(blocks["emb.timestep_embedder.linear_2"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, freq_dim);  // [freq_dim, N]
            e      = l1->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = l2->forward(ctx, e);  // [dim, N] embedded timestep

            auto mod = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, e));  // [out_dim, N]
            return {e, mod};
        }
    };

    // One video DiT block: gated self-attn (attn1), gated text cross-attn
    // (attn2), gelu FFN, and two raw modulation tables.
    class Ltx2TransformerBlock : public GGMLBlock {
    protected:
        int dim;
        float eps;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"]        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 9);
            params["prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
        }

    public:
        Ltx2TransformerBlock(const Ltx2Params& p)
            : dim(p.dim), eps(p.eps) {
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new GatedAttention(p.dim, p.dim, p.num_heads, p.eps, /*use_rope*/ true));
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new GatedAttention(p.dim, p.cross_attention_dim, p.num_heads, p.eps, /*use_rope*/ false));
            blocks["ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(p.dim, p.ffn_dim));
        }

        // x:        [dim, n_token, N]
        // mod:      [dim, 9, N]   (global adaln modulation, already reshaped)
        // prompt:   [dim, 2, N]   (global prompt modulation, already reshaped)
        // context:  [cross_attention_dim, n_ctx, N]
        // pe:       [n_token, head_dim/2, 2, 2]
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* mod,
                                    struct ggml_tensor* prompt,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe) {
            auto attn1 = std::dynamic_pointer_cast<GatedAttention>(blocks["attn1"]);
            auto attn2 = std::dynamic_pointer_cast<GatedAttention>(blocks["attn2"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            auto ctxg = ctx->ggml_ctx;

            // per-block modulation = global modulation + block scale_shift_table
            auto e = ggml_add(ctxg, mod, params["scale_shift_table"]);  // [dim, 9, N]
            auto es = ggml_ext_chunk(ctxg, e, 9, 1);                    // 9 x [dim, 1, N]

            auto pe_mod = ggml_add(ctxg, prompt, params["prompt_scale_shift_table"]);  // [dim, 2, N]
            auto ps     = ggml_ext_chunk(ctxg, pe_mod, 2, 1);                           // 2 x [dim, 1, N]

            // self-attention (modulated)
            auto y = ltx2_modulate(ctxg, ltx2_norm(ctxg, x, eps), es[1], es[0]);
            y      = attn1->forward(ctx, y, y, pe);
            x      = ggml_add(ctxg, x, ggml_mul(ctxg, y, es[2]));

            // text cross-attention (modulated query stream + modulated prompt)
            auto xc  = ltx2_modulate(ctxg, ltx2_norm(ctxg, x, eps), es[4], es[3]);
            auto ctc = ltx2_modulate(ctxg, context, ps[1], ps[0]);
            auto c   = attn2->forward(ctx, xc, ctc, nullptr);
            x        = ggml_add(ctxg, x, ggml_mul(ctxg, c, es[5]));

            // feed-forward (modulated)
            auto f = ltx2_modulate(ctxg, ltx2_norm(ctxg, x, eps), es[7], es[6]);
            f      = ff->forward(ctx, f);
            x      = ggml_add(ctxg, x, ggml_mul(ctxg, f, es[8]));

            return x;
        }
    };

    // One connector block: self-attn + FFN (no modulation tables).
    class Ltx2ConnectorBlock : public GGMLBlock {
    public:
        float eps;

        Ltx2ConnectorBlock(const Ltx2Params& p)
            : eps(p.eps) {
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new GatedAttention(p.dim, p.dim, p.num_heads, p.eps, /*use_rope*/ false));
            blocks["ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(p.dim, p.ffn_dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto attn = std::dynamic_pointer_cast<GatedAttention>(blocks["attn1"]);
            auto ff   = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
            auto ctxg = ctx->ggml_ctx;
            x         = ggml_add(ctxg, x, attn->forward(ctx, ltx2_norm(ctxg, x, eps), ltx2_norm(ctxg, x, eps), nullptr));
            x         = ggml_add(ctxg, x, ff->forward(ctx, ltx2_norm(ctxg, x, eps)));
            return x;
        }
    };

    // video_embeddings_connector: learnable registers + N 1d transformer blocks.
    class Ltx2Connector : public GGMLBlock {
    protected:
        int dim;
        int num_registers;
        int num_layers;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["learnable_registers"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, num_registers);
        }

    public:
        Ltx2Connector(const Ltx2Params& p)
            : dim(p.dim), num_registers(p.connector_registers), num_layers(p.connector_num_layers) {
            for (int i = 0; i < p.connector_num_layers; i++) {
                blocks["transformer_1d_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Ltx2ConnectorBlock(p));
            }
        }

        // context: [dim, n_ctx, 1] -> [dim, num_registers + n_ctx, 1]
        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* context) {
            auto ctxg = ctx->ggml_ctx;

            auto regs = ggml_reshape_3d(ctxg, params["learnable_registers"], dim, num_registers, 1);
            auto x    = ggml_concat(ctxg, regs, context, 1);  // [dim, num_registers + n_ctx, 1]

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<Ltx2ConnectorBlock>(blocks["transformer_1d_blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
            }
            return x;
        }
    };

    // Top-level video DiT.
    class Ltx2 : public GGMLBlock {
    protected:
        int dim    = 4096;
        float eps  = 1e-6f;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
        }

    public:
        Ltx2Params params_;

        Ltx2(const Ltx2Params& p)
            : dim(p.dim), eps(p.eps), params_(p) {
            blocks["patchify_proj"]              = std::shared_ptr<GGMLBlock>(new Linear(p.in_channels, p.dim));
            blocks["proj_out"]                   = std::shared_ptr<GGMLBlock>(new Linear(p.dim, p.in_channels));
            blocks["adaln_single"]               = std::shared_ptr<GGMLBlock>(new AdaLnSingle(p.timestep_freq_dim, p.dim, 9 * p.dim));
            blocks["prompt_adaln_single"]        = std::shared_ptr<GGMLBlock>(new AdaLnSingle(p.timestep_freq_dim, p.dim, 2 * p.dim));
            blocks["video_embeddings_connector"] = std::shared_ptr<GGMLBlock>(new Ltx2Connector(p));
            for (int i = 0; i < p.num_layers; i++) {
                blocks["transformer_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Ltx2TransformerBlock(p));
            }
        }

        // x:        [W, H, T, C] latent
        // timestep: [N]
        // context:  [cross_attention_dim, n_ctx, N] text features
        // pe:       [n_token, head_dim/2, 2, 2]
        // returns:  [W, H, T, C] latent prediction
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe) {
            auto ctxg = ctx->ggml_ctx;

            auto patchify   = std::dynamic_pointer_cast<Linear>(blocks["patchify_proj"]);
            auto proj_out   = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            auto adaln      = std::dynamic_pointer_cast<AdaLnSingle>(blocks["adaln_single"]);
            auto prompt_ada = std::dynamic_pointer_cast<AdaLnSingle>(blocks["prompt_adaln_single"]);
            auto connector  = std::dynamic_pointer_cast<Ltx2Connector>(blocks["video_embeddings_connector"]);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];
            int64_t C = x->ne[3];
            int64_t N = 1;

            // patchify: [W,H,T,C] -> tokens [C, n_token, N] -> [dim, n_token, N]
            auto tokens = ggml_cont(ctxg, ggml_permute(ctxg, x, 1, 2, 3, 0));  // [C, W, H, T]
            tokens      = ggml_reshape_3d(ctxg, tokens, C, W * H * T, N);       // [C, n_token, N]
            auto h      = patchify->forward(ctx, tokens);                       // [dim, n_token, N]

            // timestep modulation tables
            auto ada       = adaln->forward(ctx, timestep);          // embedded [dim,N], mod [9*dim,N]
            auto embedded  = ada.first;
            auto mod       = ggml_reshape_3d(ctxg, ada.second, dim, 9, N);
            auto prompt    = prompt_ada->forward(ctx, timestep).second;
            prompt         = ggml_reshape_3d(ctxg, prompt, dim, 2, N);

            // text connector
            context = connector->forward(ctx, context);  // [dim, n_reg + n_ctx, N]

            for (int i = 0; i < params_.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<Ltx2TransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                h          = block->forward(ctx, h, mod, prompt, context, pe);
            }

            // final norm + global scale_shift_table modulation (N == 1)
            auto final_mod = ggml_add(ctxg, params["scale_shift_table"], embedded);  // [dim, 2]
            auto fs        = ggml_ext_chunk(ctxg, final_mod, 2, 1);
            h              = ltx2_modulate(ctxg, ltx2_norm(ctxg, h, eps), fs[1], fs[0]);
            h              = proj_out->forward(ctx, h);  // [C, n_token, N]

            // unpatchify back to [W, H, T, C]
            h = ggml_reshape_4d(ctxg, h, C, W, H, T);
            h = ggml_cont(ctxg, ggml_permute(ctxg, h, 3, 0, 1, 2));  // [W, H, T, C]
            return h;
        }
    };

    struct Ltx2Runner : public GGMLRunner {
        std::string desc = "ltx2_dit";
        Ltx2Params params;
        Ltx2 dit;
        std::vector<float> pe_vec;

        Ltx2Runner(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "")
            : GGMLRunner(backend, offload_params_to_cpu),
              params(infer_params(tensor_storage_map, prefix)),
              dit(params) {
            LOG_INFO("LTX-2 DiT: num_layers=%d dim=%d heads=%d in_ch=%d connector_layers=%d",
                     params.num_layers, params.dim, params.num_heads, params.in_channels, params.connector_num_layers);
            dit.init(params_ctx, tensor_storage_map, prefix);
        }

        // Infer the model geometry from tensor shapes so reduced-size synthetic
        // checkpoints load as well as the full LTX-2.3 weights. Falls back to the
        // confirmed LTX-2.3 defaults when a tensor is absent.
        static Ltx2Params infer_params(const String2TensorStorage& tsm, const std::string& prefix) {
            Ltx2Params p;
            std::string base = prefix.empty() ? "" : prefix + ".";

            int max_block     = -1;
            int max_connector = -1;
            for (auto& pair : tsm) {
                const std::string& n = pair.first;
                if (n.compare(0, base.size(), base) != 0) {
                    continue;
                }
                std::string rel = n.substr(base.size());
                max_block       = std::max(max_block, parse_index(rel, "transformer_blocks."));
                max_connector   = std::max(max_connector, parse_index(rel, "video_embeddings_connector.transformer_1d_blocks."));
            }
            if (max_block >= 0) {
                p.num_layers = max_block + 1;
            }
            if (max_connector >= 0) {
                p.connector_num_layers = max_connector + 1;
            }

            // dim / in_channels from patchify_proj.weight [ne0=in, ne1=out]
            const TensorStorage* patch = find(tsm, base + "patchify_proj.weight");
            if (patch != nullptr && patch->n_dims >= 2) {
                p.in_channels = (int)patch->ne[0];
                p.dim         = (int)patch->ne[1];
            }
            // ffn_dim from a transformer block ff
            const TensorStorage* ff = find(tsm, base + "transformer_blocks.0.ff.net.0.proj.weight");
            if (ff != nullptr && ff->n_dims >= 2) {
                p.ffn_dim = (int)ff->ne[1];
            }
            // cross_attention_dim from attn2.to_k.weight [ne0=ctx_dim]
            const TensorStorage* xk = find(tsm, base + "transformer_blocks.0.attn2.to_k.weight");
            if (xk != nullptr && xk->n_dims >= 1) {
                p.cross_attention_dim = (int)xk->ne[0];
            }
            // num_heads from to_gate_logits.weight [ne1=num_heads]
            const TensorStorage* gate = find(tsm, base + "transformer_blocks.0.attn1.to_gate_logits.weight");
            if (gate != nullptr && gate->n_dims >= 2) {
                p.num_heads = (int)gate->ne[1];
            }
            if (p.num_heads > 0) {
                p.head_dim = p.dim / p.num_heads;
            }
            // registers from connector learnable_registers [ne1=num_registers]
            const TensorStorage* reg = find(tsm, base + "video_embeddings_connector.learnable_registers");
            if (reg != nullptr && reg->n_dims >= 2) {
                p.connector_registers = (int)reg->ne[1];
            }
            // timestep embedder input dim
            const TensorStorage* te = find(tsm, base + "adaln_single.emb.timestep_embedder.linear_1.weight");
            if (te != nullptr && te->n_dims >= 1) {
                p.timestep_freq_dim = (int)te->ne[0];
            }
            return p;
        }

        static int parse_index(const std::string& rel, const std::string& tag) {
            size_t pos = rel.find(tag);
            if (pos == std::string::npos) {
                return -1;
            }
            return atoi(rel.c_str() + pos + tag.size());
        }

        static const TensorStorage* find(const String2TensorStorage& tsm, const std::string& name) {
            auto it = tsm.find(name);
            return it == tsm.end() ? nullptr : &it->second;
        }

        std::string get_desc() override {
            return desc;
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            dit.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context) {
            struct ggml_cgraph* gf = new_graph_custom(MAX_GRAPH_SIZE / 2);

            x         = to_backend(x);
            timesteps = to_backend(timesteps);
            context   = to_backend(context);

            // 3D RoPE positions for the latent grid (patch size 1x1x1).
            std::vector<int> axes_dim = ltx2_rope_axes_dim(params.head_dim);
            int axes_dim_sum          = axes_dim[0] + axes_dim[1] + axes_dim[2];
            pe_vec                    = Rope::gen_wan_pe(static_cast<int>(x->ne[2]),
                                                        static_cast<int>(x->ne[1]),
                                                        static_cast<int>(x->ne[0]),
                                                        1, 1, 1, 1,
                                                        static_cast<int>(params.rope_theta),
                                                        axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx         = get_context();
            struct ggml_tensor* out = dit.forward(&runner_ctx, x, timesteps, context, pe);

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     struct ggml_tensor** output     = nullptr,
                     struct ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context);
            };
            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }
    };

}  // namespace LTX2

#endif  // __LTX2_HPP__
