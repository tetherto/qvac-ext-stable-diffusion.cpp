#ifndef __LTX2_HPP__
#define __LTX2_HPP__

#include "common_block.hpp"

// LTX-2.3 video-stream DiT (AVTransformer3DModel, video half only).
//
// This is the M1 scaffolding: the GGMLBlock tree declares exactly the video
// tensors produced by script/convert_ltx2_to_gguf.py so the model can be
// loaded and its params allocated on CPU. Forward passes (denoising) are added
// in M2; the blocks here intentionally declare parameters only.
//
// Confirmed architecture (Lightricks/LTX-2.3, model_version 2.3.0):
//   num_layers=48, num_heads=32, head_dim=128 (dim 4096), in_channels=128,
//   caption_channels=3840, cross_attention_dim=4096, qk_norm=rms_norm,
//   gated attention (to_gate_logits), FFN 4096->16384 (gelu-approx),
//   8-layer embeddings connector with 128 learnable registers.
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
    };

    // q/k/v + gated output projection with rms qk-norm; matches the LTX-2
    // `attn1`/`attn2` layout (to_q, to_k, to_v, to_out.0, q_norm, k_norm,
    // to_gate_logits).
    class GatedAttention : public GGMLBlock {
    public:
        GatedAttention(int dim, int ctx_dim, int num_heads, float eps) {
            blocks["to_q"]           = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["to_k"]           = std::shared_ptr<GGMLBlock>(new Linear(ctx_dim, dim));
            blocks["to_v"]           = std::shared_ptr<GGMLBlock>(new Linear(ctx_dim, dim));
            blocks["to_out.0"]       = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["q_norm"]         = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            blocks["k_norm"]         = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            blocks["to_gate_logits"] = std::shared_ptr<GGMLBlock>(new Linear(dim, num_heads));
        }
    };

    // gelu-approximate FFN: net.0.proj (dim->inner), net.2 (inner->dim).
    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int dim, int inner) {
            blocks["net.0.proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, inner));
            blocks["net.2"]      = std::shared_ptr<GGMLBlock>(new Linear(inner, dim));
        }
    };

    // adaln_single / prompt_adaln_single: a timestep embedder MLP plus a final
    // projection producing the modulation table.
    class AdaLnSingle : public GGMLBlock {
    public:
        AdaLnSingle(int freq_dim, int dim, int out_dim) {
            blocks["emb.timestep_embedder.linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(freq_dim, dim));
            blocks["emb.timestep_embedder.linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["linear"]                         = std::shared_ptr<GGMLBlock>(new Linear(dim, out_dim));
        }
    };

    // One video DiT block: self-attn (attn1), text cross-attn (attn2), FFN, and
    // two raw modulation tables.
    class Ltx2TransformerBlock : public GGMLBlock {
    protected:
        int dim;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"]        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 9);
            params["prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
        }

    public:
        Ltx2TransformerBlock(const Ltx2Params& p)
            : dim(p.dim) {
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new GatedAttention(p.dim, p.dim, p.num_heads, p.eps));
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new GatedAttention(p.dim, p.cross_attention_dim, p.num_heads, p.eps));
            blocks["ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(p.dim, p.ffn_dim));
        }
    };

    // One connector block: self-attn + FFN (no modulation tables).
    class Ltx2ConnectorBlock : public GGMLBlock {
    public:
        Ltx2ConnectorBlock(const Ltx2Params& p) {
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new GatedAttention(p.dim, p.dim, p.num_heads, p.eps));
            blocks["ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(p.dim, p.ffn_dim));
        }
    };

    // video_embeddings_connector: learnable registers + N 1d transformer blocks.
    class Ltx2Connector : public GGMLBlock {
    protected:
        int dim;
        int num_registers;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["learnable_registers"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, num_registers);
        }

    public:
        Ltx2Connector(const Ltx2Params& p)
            : dim(p.dim), num_registers(p.connector_registers) {
            for (int i = 0; i < p.connector_num_layers; i++) {
                blocks["transformer_1d_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Ltx2ConnectorBlock(p));
            }
        }
    };

    // Top-level video DiT.
    class Ltx2 : public GGMLBlock {
    protected:
        int dim;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
        }

    public:
        Ltx2(const Ltx2Params& p)
            : dim(p.dim) {
            blocks["patchify_proj"]              = std::shared_ptr<GGMLBlock>(new Linear(p.in_channels, p.dim));
            blocks["proj_out"]                   = std::shared_ptr<GGMLBlock>(new Linear(p.dim, p.in_channels));
            blocks["adaln_single"]               = std::shared_ptr<GGMLBlock>(new AdaLnSingle(p.timestep_freq_dim, p.dim, 9 * p.dim));
            blocks["prompt_adaln_single"]        = std::shared_ptr<GGMLBlock>(new AdaLnSingle(p.timestep_freq_dim, p.dim, 2 * p.dim));
            blocks["video_embeddings_connector"] = std::shared_ptr<GGMLBlock>(new Ltx2Connector(p));
            for (int i = 0; i < p.num_layers; i++) {
                blocks["transformer_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Ltx2TransformerBlock(p));
            }
        }
    };

    struct Ltx2Runner : public GGMLRunner {
        std::string desc = "ltx2_dit";
        Ltx2Params params;
        Ltx2 dit;

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
    };

}  // namespace LTX2

#endif  // __LTX2_HPP__
