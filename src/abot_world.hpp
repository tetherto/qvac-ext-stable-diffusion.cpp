#ifndef __ABOT_WORLD_HPP__
#define __ABOT_WORLD_HPP__

// ABot-World causal walk — recompute formulation.
//
// ABot-World generates video block-by-block (num_frame_per_block latent frames
// per step) with keyboard-action conditioning. The reference implementation
// keeps a rolling KV cache; this port reproduces the exact same math by
// recomputing attention over the full visible sequence each denoise step:
//
//   sequence = [ ref tokens | history frames (clean, t=0) | block frames (t) ]
//
// with a per-row mask: tokens of frame f attend the ref tokens plus all frames
// of blocks <= block(f) that lie inside f's trailing local attention window
// (local_attn_size frames). Attention within a block is full (the causal
// boundary is between blocks). History frames re-derive exactly the K/V the
// reference wrote into its cache (clean latents at context_noise=0 modulation,
// each frame using its own historical action features), including under
// eviction, because each frame's row mask reproduces the window that frame saw
// when it was generated.
//
// RoPE uses window-local temporal ids (the reference's relative-RoPE scheme,
// ids clamped to local_attn_size-1 <= 20) and the reference's negative-time
// ids for ref slots. The 4-step distilled schedule (denoising_step_list warped
// by timestep_shift) runs flow-matching updates:
//   x0 = xt - sigma(t) * flow,  xt' = (1 - sigma(t')) * x0 + sigma(t') * eps
// where sigma(t) = t / 1000 for the warped step values.
//
// Semantics derived from the reference (github.com/amap-cvlab/ABot-World):
// pipeline/causal_inference.py generate_next_block, wan/modules/causal_model.py
// (_forward_inference, ref token prep, relative RoPE), utils/wan_wrapper.py
// (flow<->x0 conversion), utils/scheduler.py (FlowMatchScheduler).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <functional>
#include <random>

#include "ggml_extend.hpp"
#include "model.h"
#include "rope.hpp"
#include "vae.hpp"  // must precede tae.hpp (TinyVideoAutoEncoder's base)
#include "tae.hpp"
#include "wan.hpp"

namespace ABOT {

struct AbotWorldConfig {
    int num_frame_per_block = 3;
    int local_attn_size     = 8;      // latent-frame window (test config; deployed default 21)
    int rope_temporal_clamp = 20;     // reference clamps window-local temporal ids to <= 20
    float context_noise_t   = 0.0f;   // history frames' timestep
    std::vector<float> denoise_steps = {1000.0f, 937.5f, 833.3333333f, 625.0f};
    int act_in_dim           = 32;    // 8 keys x 4 (repeat_interleave)
    int act_downscale_factor = 16;
};

static inline float abot_sigma_of_t(float t) {
    return t / 1000.0f;  // warped steps satisfy t = sigma' * 1000 exactly
}

// One latent frame of walk state.
struct AbotFrame {
    sd::Tensor<float> latent;  // [W_l, H_l, C] (ggml order) — clean/final latent
    uint8_t action_mask = 0;   // 8-key bitmask active when this frame was generated
};

struct AbotScenePack {
    sd::Tensor<float> prompt_embeds;        // [4096, 512] per token row-major -> tensor shape {4096, 512, 1}
    sd::Tensor<float> first_frame_latents;  // {W_l, H_l, C, 1}
    sd::Tensor<float> ref_latents;          // {32, 32, C, K} (T=1 squeezed)
    sd::Tensor<float> ref_mask;             // {K}
    int ref_slots = 0;

    // Minimal safetensors reader (F32 tensors only — the scene pack is fp32).
    // Torch shapes map to ggml ne reversed; trailing singleton dims dropped.
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            LOG_ERROR("scene pack: cannot open '%s'", path.c_str());
            return false;
        }
        uint64_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 8);
        std::string header(hlen, '\0');
        f.read(header.data(), static_cast<std::streamsize>(hlen));
        const uint64_t data_base = 8 + hlen;

        auto fetch = [&](const std::string& name, sd::Tensor<float>& out,
                         std::vector<int64_t>& torch_shape) -> bool {
            size_t p = header.find("\"" + name + "\"");
            if (p == std::string::npos) {
                return false;
            }
            size_t sp = header.find("\"shape\"", p);
            size_t sb = header.find('[', sp), se = header.find(']', sb);
            torch_shape.clear();
            for (size_t q = sb + 1; q < se;) {
                while (q < se && (header[q] == ' ' || header[q] == ',')) q++;
                if (q >= se) break;
                torch_shape.push_back(strtoll(header.c_str() + q, nullptr, 10));
                while (q < se && header[q] != ',') q++;
            }
            size_t op = header.find("\"data_offsets\"", p);
            size_t ob = header.find('[', op);
            uint64_t off0 = strtoull(header.c_str() + ob + 1, nullptr, 10);
            int64_t n = 1;
            for (auto s : torch_shape) n *= s;
            std::vector<int64_t> ne;  // reversed, no trailing 1s beyond 4
            for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it) {
                if (*it != 1 || ne.size() < 1) ne.push_back(*it);
            }
            while (ne.size() < 1) ne.push_back(1);
            out.resize(ne);
            if (out.numel() != n) {
                out.resize({n});
            }
            f.seekg(static_cast<std::streamoff>(data_base + off0));
            f.read(reinterpret_cast<char*>(out.data()), n * static_cast<int64_t>(sizeof(float)));
            return f.good();
        };

        std::vector<int64_t> sh;
        if (!fetch("prompt_embeds", prompt_embeds, sh)) {          // [1,512,4096]
            return false;
        }
        prompt_embeds.resize({sh[2], sh[1], 1, 1});
        if (!fetch("first_frame_latents", first_frame_latents, sh)) {  // [1,1,C,H,W]
            return false;
        }
        first_frame_latents.resize({sh[4], sh[3], sh[2], 1});
        if (fetch("ref_latents", ref_latents, sh)) {               // [1,K,C,1,32,32]
            ref_slots = static_cast<int>(sh[1]);
            // stored per-slot [C,1,32,32]; conv needs ne {32,32,K,C}:
            // torch order is (K,C,32,32) contiguous -> need transpose to {W,H,K,C}
            sd::Tensor<float> re({sh[5], sh[4], sh[1], sh[2]});
            const int64_t K = sh[1], C = sh[2], HW = sh[4] * sh[5];
            for (int64_t k = 0; k < K; k++) {
                for (int64_t c = 0; c < C; c++) {
                    memcpy(re.data() + (c * K + k) * HW,
                           ref_latents.data() + (k * C + c) * HW,
                           static_cast<size_t>(HW) * sizeof(float));
                }
            }
            ref_latents = std::move(re);
            std::vector<int64_t> msh;
            if (!fetch("ref_mask", ref_mask, msh)) {
                return false;
            }
            ref_mask.resize({ref_slots});
        }
        return true;
    }
};

// Wan subclass: same weights/blocks, causal recompute graph.
class AbotWan : public WAN::Wan {
public:
    AbotWan() = default;
    explicit AbotWan(WAN::WanParams params)
        : WAN::Wan(params) {}

    // Layout conventions mirror WAN::Wan::forward_orig:
    // x_all:      ne {W_l, H_l, F_vis, C}  latents (history clean + block xt)
    // act_planes: ne {W_l/ds, H_l/ds, act_in_dim*ds*ds, F_vis}? NO — adapter input
    //             is the pixel-unshuffled action image at latent grid:
    //             ne {W_l, H_l, act_in_dim*ds*ds/ (16/ latent factor) ...} — the
    //             host builds it at ne {w_in, h_in, C_unsh, F_vis} such that the
    //             adapter conv (k2 s2) lands on the token grid (w_len, h_len).
    // t_frames:   ne {F_vis + 1} per-frame timesteps, extra last row = 0 (ref)
    // context:    ne {4096, 512, 1} prompt embeds
    // ref_latents: ne {32, 32, K, C} or nullptr (T dim = slots, temporal k=1)
    // ref_mask:   ne {K} or nullptr
    // pe:         packed rope table for [ref | video] token sequence
    // attn_mask:  ne {L, L} additive f32 (built host-side)
    // token_frame_rows: ne {L} int32 -> row in the timestep table (ref rows -> F_vis)
    // returns flow tokens for the LAST block_frames frames: {pt*ph*pw*out_dim, block_tokens, 1}
    ggml_tensor* forward_causal(GGMLRunnerContext* ctx,
                                ggml_tensor* x_all,
                                ggml_tensor* act_planes,
                                ggml_tensor* t_frames,
                                ggml_tensor* context,
                                ggml_tensor* ref_latents,
                                ggml_tensor* ref_mask,
                                ggml_tensor* pe,
                                ggml_tensor* attn_mask,
                                ggml_tensor* token_frame_rows,
                                int block_frames,
                                int64_t& out_h_len,
                                int64_t& out_w_len) {
        auto gctx = ctx->ggml_ctx;

        auto patch_embedding  = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);
        auto text_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
        auto text_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);
        auto time_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
        auto time_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
        auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);
        auto head             = std::dynamic_pointer_cast<WAN::Head>(blocks["head"]);
        auto adapter          = std::dynamic_pointer_cast<WAN::ActControlAdapter>(blocks["act_control_adapter"]);

        const int64_t F_vis = x_all->ne[2];

        // ── video tokens: patchify (+ action adapter), forward_orig layout chain ──
        ggml_tensor* x = patch_embedding->forward(ctx, x_all);  // {w_len, h_len, F_vis, dim}
        out_w_len      = x->ne[0];
        out_h_len      = x->ne[1];
        const int64_t fsl = out_w_len * out_h_len;  // tokens per frame
        x = ggml_reshape_3d(gctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3], 1);  // {w*h*F, dim, 1}
        x = ggml_ext_cont(gctx, ggml_ext_torch_permute(gctx, x, 1, 0, 2, 3));       // {dim, w*h*F, 1}

        if (adapter != nullptr && act_planes != nullptr) {
            ggml_tensor* a = adapter->forward(ctx, act_planes);  // {w_len, h_len, dim, F_vis}
            a              = ggml_reshape_3d(gctx, a, a->ne[0] * a->ne[1], a->ne[2], a->ne[3]);  // {w*h, dim, F}
            a              = ggml_ext_cont(gctx, ggml_ext_torch_permute(gctx, a, 1, 0, 2, 3));   // {dim, w*h, F}
            a              = ggml_reshape_3d(gctx, a, a->ne[0], a->ne[1] * a->ne[2], 1);         // {dim, w*h*F, 1}
            x              = ggml_add(gctx, x, a);  // act_context_scale = 1.0
        }

        // ── ref tokens (masked after conv so conv bias is cancelled on empty slots) ──
        if (ref_latents != nullptr) {
            ggml_tensor* r = patch_embedding->forward(ctx, ref_latents);  // {16, 16, K, dim}
            ggml_tensor* m = ggml_reshape_4d(gctx, ref_mask, 1, 1, ref_mask->ne[0], 1);
            r              = ggml_mul(gctx, r, m);
            r = ggml_reshape_3d(gctx, r, r->ne[0] * r->ne[1] * r->ne[2], r->ne[3], 1);  // {16*16*K, dim, 1}
            r = ggml_ext_cont(gctx, ggml_ext_torch_permute(gctx, r, 1, 0, 2, 3));       // {dim, n_ref, 1}
            x = ggml_concat(gctx, r, x, 1);                                              // ref first
        }
        const int64_t L = x->ne[1];

        // ── per-frame modulation table -> per-token via get_rows ──
        // rows: F_vis frame timesteps + 1 extra row (t = 0) used by ref tokens
        ggml_tensor* e_tbl = ggml_ext_timestep_embedding(gctx, t_frames, params.freq_dim);  // [freq_dim, F_vis+1]
        e_tbl              = time_embedding_0->forward(ctx, e_tbl);
        e_tbl              = ggml_silu_inplace(gctx, e_tbl);
        e_tbl              = time_embedding_2->forward(ctx, e_tbl);  // [dim, F_vis+1]  (this is `e`)
        ggml_tensor* e0_tbl = ggml_silu(gctx, e_tbl);
        e0_tbl              = time_projection_1->forward(ctx, e0_tbl);  // [6*dim, F_vis+1]

        ggml_tensor* e0_tok = ggml_get_rows(gctx, e0_tbl, token_frame_rows);            // [6*dim, L]
        e0_tok              = ggml_reshape_4d(gctx, e0_tok, params.dim, 6, L, 1);        // [N=1, L, 6, dim]

        // ── context ──
        ggml_tensor* c = text_embedding_0->forward(ctx, context);
        c              = ggml_ext_gelu(gctx, c);
        c              = text_embedding_2->forward(ctx, c);  // [dim, 512, 1]

        // ── transformer ──
        for (int i = 0; i < params.num_layers; i++) {
            auto block = std::dynamic_pointer_cast<WAN::WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);
            x          = block->forward(ctx, x, e0_tok, pe, c, 0, attn_mask);
        }

        // ── head over the trailing block tokens only ──
        const int64_t block_tokens = static_cast<int64_t>(block_frames) * fsl;
        ggml_tensor* xb            = ggml_ext_slice(gctx, x, 1, L - block_tokens, L);  // [dim, block_tokens, 1]
        // per-frame e for the block frames: rows F_vis-block_frames .. F_vis-1
        ggml_tensor* e_blk = ggml_ext_slice(gctx, e_tbl, 1, F_vis - block_frames, F_vis);  // [dim, block_frames]
        e_blk              = ggml_reshape_3d(gctx, e_blk, params.dim, block_frames, 1);     // [N=1, T, dim]
        ggml_tensor* out   = head->forward(ctx, xb, e_blk);  // [pt*ph*pw*out_dim, block_tokens, 1]
        return out;
    }
};

// Runner: owns the AbotWan blocks + walk state, builds one graph per denoise
// forward, and runs the host-side 4-step block loop.
struct AbotWorldRunner : public GGMLRunner {
    AbotWorldConfig cfg;
    WAN::WanParams wan_params;
    AbotWan wan;
    AbotScenePack scene;

    // walk state
    std::vector<AbotFrame> history;  // finalized latent frames (clean)
    int64_t lat_w = 0, lat_h = 0, lat_c = 0;

    std::vector<float> pe_vec;

    AbotWorldRunner(ggml_backend_t backend,
                    ggml_backend_t params_backend,
                    const String2TensorStorage& tensor_storage_map,
                    const std::string prefix,
                    const AbotWorldConfig& config = {})
        : GGMLRunner(backend, params_backend), cfg(config) {
        wan_params.model_type = "t2v";
        wan_params.dim        = 3072;
        wan_params.eps        = 1e-06f;
        wan_params.ffn_dim    = 14336;
        wan_params.freq_dim   = 256;
        wan_params.in_dim     = 48;
        wan_params.num_heads  = 24;
        wan_params.out_dim    = 48;
        wan_params.text_len   = 512;
        wan_params.num_layers = 30;
        wan_params.abot_world = true;
        wan                   = AbotWan(wan_params);
        wan.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "ABot-World-5B (causal walk)";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
        wan.get_param_tensors(tensors, prefix);
    }

    // ---- host-side helpers ----

    // Per-token 3-axis rope ids: ref slots at negative constant times (the
    // reference's _build_ref_freqs), video frames at absolute frame ids (the
    // reference's REL_ROPE_CACHE fast path with base 0; valid for walks well
    // below the re-base threshold of ~256 frames).
    std::vector<float> build_pe(int n_ref_slots, int ref_grid, int F_vis, int h_len, int w_len) {
        std::vector<std::vector<float>> ids;
        const int tokens_per_slot = ref_grid * ref_grid;
        const int t_step          = std::max(tokens_per_slot, 256);
        for (int k = 0; k < n_ref_slots; k++) {
            float t_id = -static_cast<float>((n_ref_slots - k) * t_step);
            for (int hh = 0; hh < ref_grid; hh++) {
                for (int ww = 0; ww < ref_grid; ww++) {
                    ids.push_back({t_id, static_cast<float>(hh), static_cast<float>(ww)});
                }
            }
        }
        for (int f = 0; f < F_vis; f++) {
            for (int hh = 0; hh < h_len; hh++) {
                for (int ww = 0; ww < w_len; ww++) {
                    ids.push_back({static_cast<float>(f), static_cast<float>(hh), static_cast<float>(ww)});
                }
            }
        }
        return Rope::embed_nd(ids, 1, static_cast<float>(wan_params.theta), wan_params.axes_dim);
    }

    // Additive mask {L(k), L(q)}: video rows of block b attend ref + frames of
    // blocks <= b inside b's trailing window; ref rows attend ref + block 0.
    std::vector<float> build_mask(int n_ref, int F_vis, int fsl, int block_frames) {
        const int64_t L = n_ref + static_cast<int64_t>(F_vis) * fsl;
        std::vector<float> mask(static_cast<size_t>(L) * L, -INFINITY);
        auto frame_of = [&](int64_t tok) -> int {  // -1 for ref tokens
            return tok < n_ref ? -1 : static_cast<int>((tok - n_ref) / fsl);
        };
        for (int64_t q = 0; q < L; q++) {
            int fq = frame_of(q);
            int bq = fq < 0 ? 0 : fq / cfg.num_frame_per_block;
            // trailing window (in frames) as seen when block bq was generated
            int hi = (bq + 1) * cfg.num_frame_per_block;  // frames [0, hi)
            int lo = std::max(0, hi - cfg.local_attn_size);
            for (int64_t k = 0; k < L; k++) {
                int fk       = frame_of(k);
                bool allowed = false;
                if (fk < 0) {
                    allowed = true;  // ref keys visible to everyone
                } else if (fq < 0) {
                    allowed = fk < cfg.num_frame_per_block;  // ref rows formed with block 0
                } else {
                    allowed = fk >= lo && fk < hi;
                }
                if (allowed) {
                    mask[static_cast<size_t>(q) * L + k] = 0.0f;
                }
            }
        }
        return mask;
    }

    // Pixel-unshuffled action planes for one frame: the reference broadcasts the
    // 8-key vector to 8 channels, repeat_interleaves x4 -> 32 channels constant
    // over HxW, then PixelUnshuffle(16): output channel c corresponds to input
    // channel c / (16*16) -> value = key[(c / 256) / 4].
    void fill_act_plane(float* dst, uint8_t action_mask, int w_in, int h_in, int c_unsh) {
        for (int c = 0; c < c_unsh; c++) {
            int key   = (c / (cfg.act_downscale_factor * cfg.act_downscale_factor)) / 4;
            float val = (action_mask >> key) & 1 ? 1.0f : 0.0f;
            float* p  = dst + static_cast<size_t>(c) * w_in * h_in;
            std::fill(p, p + static_cast<size_t>(w_in) * h_in, val);
        }
    }

    // One denoise forward over the full visible sequence; returns x0 for the
    // current block (host applies the flow->x0 conversion).
    // xt_block: {W,H,C} x block_frames latents (current noisy block)
    // t_value: timestep for the block frames (frame 0 of walk pinned to 0 by caller)
    sd::Tensor<float> forward_step(const std::vector<const float*>& frame_latents,  // F_vis pointers {W*H*C}
                                   const std::vector<uint8_t>& frame_actions,
                                   const std::vector<float>& frame_timesteps,  // F_vis values
                                   int block_frames,
                                   int n_threads) {
        const int F_vis = static_cast<int>(frame_latents.size());
        const int ds    = cfg.act_downscale_factor;
        const int w_in  = static_cast<int>(lat_w) / ds * ds;  // == lat_w (multiple of 16 grid)
        // adapter input grid: unshuffle by ds over the PIXEL grid == latent grid here:
        // reference act image is [32, H_px, W_px] with H_px = lat_h*16? No: the
        // adapter output must land on (h_len, w_len) = (lat_h/2, lat_w/2); conv is
        // k2 s2, so its input grid is (lat_h, lat_w) with C = 32*ds*ds = 8192.
        const int c_unsh = cfg.act_in_dim * ds * ds;

        // ---- build host buffers ----
        int h_len = static_cast<int>(lat_h) / 2;
        int w_len = static_cast<int>(lat_w) / 2;
        int fsl   = h_len * w_len;
        int n_ref = scene.ref_slots * 16 * 16;

        sd::Tensor<float> x_all({lat_w, lat_h, F_vis, lat_c});
        for (int f = 0; f < F_vis; f++) {
            // {W,H,T,C}: strides — for each channel c, plane at [.., f, c]
            for (int64_t c = 0; c < lat_c; c++) {
                float* dst       = x_all.data() + (c * F_vis + f) * lat_w * lat_h;
                const float* src = frame_latents[f] + c * lat_w * lat_h;
                memcpy(dst, src, static_cast<size_t>(lat_w) * lat_h * sizeof(float));
            }
        }

        sd::Tensor<float> act({lat_w, lat_h, c_unsh, F_vis});
        for (int f = 0; f < F_vis; f++) {
            fill_act_plane(act.data() + static_cast<size_t>(f) * c_unsh * lat_w * lat_h,
                           frame_actions[f], static_cast<int>(lat_w), static_cast<int>(lat_h), c_unsh);
        }

        sd::Tensor<float> tvec({F_vis + 1});
        for (int f = 0; f < F_vis; f++) {
            tvec.data()[f] = frame_timesteps[f];
        }
        tvec.data()[F_vis] = 0.0f;  // ref/modulation row

        std::vector<int32_t> rows(static_cast<size_t>(n_ref) + static_cast<size_t>(F_vis) * fsl);
        for (int i = 0; i < n_ref; i++) {
            rows[i] = F_vis;
        }
        for (int f = 0; f < F_vis; f++) {
            std::fill(rows.begin() + n_ref + static_cast<size_t>(f) * fsl,
                      rows.begin() + n_ref + static_cast<size_t>(f + 1) * fsl, f);
        }

        pe_vec                  = build_pe(scene.ref_slots, 16, F_vis, h_len, w_len);
        std::vector<float> mask = build_mask(n_ref, F_vis, fsl, block_frames);
        const int64_t L         = n_ref + static_cast<int64_t>(F_vis) * fsl;

        // ---- graph ----
        int64_t out_h = 0, out_w = 0;
        auto get_graph = [&]() -> ggml_cgraph* {
            ggml_cgraph* gf     = ggml_new_graph_custom(compute_ctx, WAN::WAN_GRAPH_SIZE, false);
            ggml_tensor* x_in   = to_backend_input(x_all);
            ggml_tensor* act_in = to_backend_input(act);
            ggml_tensor* t_in   = to_backend_input(tvec);
            ggml_tensor* ctx_in = to_backend_input(scene.prompt_embeds);
            ggml_tensor* ref_in = nullptr;
            ggml_tensor* rm_in  = nullptr;
            if (scene.ref_slots > 0) {
                ref_in = to_backend_input(scene.ref_latents);
                rm_in  = to_backend_input(scene.ref_mask);
            }
            ggml_tensor* pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2,
                                                 wan_params.axes_dim_sum / 2,
                                                 static_cast<int64_t>(pe_vec.size()) / wan_params.axes_dim_sum / 2);
            set_backend_tensor_data(pe, pe_vec.data());
            ggml_tensor* mk = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, L, L);
            set_backend_tensor_data(mk, mask.data());
            ggml_tensor* rw = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, static_cast<int64_t>(rows.size()));
            set_backend_tensor_data(rw, rows.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = wan.forward_causal(&runner_ctx, x_in, act_in, t_in, ctx_in,
                                                  ref_in, rm_in, pe, mk, rw, block_frames, out_h, out_w);
            ggml_build_forward_expand(gf, out);
            return gf;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
        if (!result.has_value()) {
            return {};
        }
        return std::move(*result);
    }

private:
    template <typename T>
    ggml_tensor* to_backend_input(sd::Tensor<T>& t) {
        auto shape = t.shape();
        ggml_tensor* g = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                            shape.size() > 0 ? shape[0] : 1,
                                            shape.size() > 1 ? shape[1] : 1,
                                            shape.size() > 2 ? shape[2] : 1,
                                            shape.size() > 3 ? shape[3] : 1);
        set_backend_tensor_data(g, t.data());
        return g;
    }
};

// Seed-stable normal RNG (xorshift128+ / Box-Muller) so free walks reproduce
// across platforms regardless of libstdc++/libc++ std::normal_distribution.
struct AbotRng {
    uint64_t s0, s1;
    bool have_spare = false;
    float spare     = 0.0f;
    explicit AbotRng(uint64_t seed = 42) {
        s0 = seed ^ 0x9E3779B97F4A7C15ULL;
        s1 = (seed << 1) | 1;
        for (int i = 0; i < 16; i++) {
            next_u64();
        }
    }
    uint64_t next_u64() {
        uint64_t x = s0, y = s1;
        s0         = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }
    float normal() {
        if (have_spare) {
            have_spare = false;
            return spare;
        }
        auto uniform = [&]() -> float {
            return static_cast<float>((next_u64() >> 11) * (1.0 / 9007199254740992.0));
        };
        float u1 = std::max(uniform(), 1e-12f), u2 = uniform();
        float r = std::sqrt(-2.0f * std::log(u1)), a = 6.28318530718f * u2;
        spare      = r * std::sin(a);
        have_spare = true;
        return r * std::cos(a);
    }
    void fill(float* dst, size_t n) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = normal();
        }
    }
};

// Interactive walk session: DiT walk core + taehv pixel decode, one keyboard
// action per generated block. Self-contained (loads its own weights); this is
// the object behind the public sd_abot_session_* C API.
class AbotWalkSession {
public:
    AbotWorldConfig cfg;
    AbotRng rng;
    int n_threads = 8;

    // optional noise override for deterministic validation: called with the
    // denoise step index (-1 = initial block noise, 0..2 = renoise eps);
    // return false to fall back to the RNG.
    std::function<bool(int block, int step, float* dst, size_t n)> noise_override;

    std::unique_ptr<AbotWorldRunner> runner;
    std::shared_ptr<TinyVideoAutoEncoder> tae;

    // finalized walk state (per-frame ggml {W,H,C} latents, torch [C,H,W] flat)
    std::vector<std::vector<float>> history;
    std::vector<uint8_t> history_actions;

    bool load(ggml_backend_t runtime_backend,
              ggml_backend_t params_backend,
              ggml_backend_t vae_backend,
              ggml_backend_t vae_params_backend,
              const std::string& dit_path,
              const std::string& taehv_path,
              const std::string& scene_path,
              const AbotWorldConfig& config,
              uint64_t seed,
              int threads) {
        cfg       = config;
        rng       = AbotRng(seed);
        n_threads = threads;

        ModelLoader ml;
        if (!ml.init_from_file(dit_path, "model.diffusion_model.")) {
            LOG_ERROR("abot session: cannot open DiT '%s'", dit_path.c_str());
            return false;
        }
        runner = std::make_unique<AbotWorldRunner>(runtime_backend, params_backend,
                                                   ml.get_tensor_storage_map(),
                                                   "model.diffusion_model.", cfg);
        if (!runner->alloc_params_buffer()) {
            return false;
        }
        std::map<std::string, ggml_tensor*> tensors;
        runner->get_param_tensors(tensors, "model.diffusion_model");
        if (!ml.load_tensors(tensors, {}, n_threads)) {
            LOG_ERROR("abot session: DiT tensor load failed");
            return false;
        }
        if (!runner->scene.load(scene_path)) {
            return false;
        }
        auto ffl      = runner->scene.first_frame_latents.shape();
        runner->lat_w = ffl[0];
        runner->lat_h = ffl[1];
        runner->lat_c = ffl.size() > 2 ? ffl[2] : 1;

        ModelLoader tl;
        if (!tl.init_from_file(taehv_path, "tae.")) {  // same prefixing as new_sd_ctx's taesd path
            LOG_ERROR("abot session: cannot open taehv '%s'", taehv_path.c_str());
            return false;
        }
        tae = std::make_shared<TinyVideoAutoEncoder>(vae_backend, vae_params_backend,
                                                     tl.get_tensor_storage_map(),
                                                     "decoder", true, VERSION_ABOT_WORLD);
        if (!tae->alloc_params_buffer()) {
            return false;
        }
        std::map<std::string, ggml_tensor*> tae_tensors;
        tae->get_param_tensors(tae_tensors, "tae");
        if (!tl.load_tensors(tae_tensors, {}, n_threads)) {
            LOG_ERROR("abot session: taehv tensor load failed");
            return false;
        }
        LOG_INFO("abot session ready: latent %dx%dx%d, ref slots %d, window %d",
                 (int)runner->lat_w, (int)runner->lat_h, (int)runner->lat_c,
                 runner->scene.ref_slots, cfg.local_attn_size);
        return true;
    }

    int64_t frame_elems() const {
        return runner->lat_w * runner->lat_h * runner->lat_c;
    }

    // Generate the next latent block under `action_mask` (bits: W,A,S,D,I,J,K,L).
    // Appends num_frame_per_block clean frames to history; returns them.
    bool step_latents(uint8_t action_mask, std::vector<std::vector<float>>& out_frames) {
        const int Fb     = cfg.num_frame_per_block;
        const size_t fel = static_cast<size_t>(frame_elems());
        const bool first = history.empty();
        const int block  = static_cast<int>(history.size()) / Fb;
        const float* ff  = runner->scene.first_frame_latents.data();

        std::vector<float> xt(static_cast<size_t>(Fb) * fel);
        if (!noise_override || !noise_override(block, -1, xt.data(), xt.size())) {
            rng.fill(xt.data(), xt.size());
            if (first) {
                memcpy(xt.data(), ff, fel * sizeof(float));  // pin frame 0 to the scene
            }
        }

        for (size_t s = 0; s < cfg.denoise_steps.size(); s++) {
            const float t_cur = cfg.denoise_steps[s];
            std::vector<const float*> frames;
            std::vector<uint8_t> facts;
            std::vector<float> fts;
            for (size_t h = 0; h < history.size(); h++) {
                frames.push_back(history[h].data());
                facts.push_back(history_actions[h]);
                fts.push_back(cfg.context_noise_t);
            }
            for (int f = 0; f < Fb; f++) {
                frames.push_back(xt.data() + static_cast<size_t>(f) * fel);
                facts.push_back(action_mask);
                fts.push_back(first && f == 0 ? 0.0f : t_cur);
            }

            auto flow = runner->forward_step(frames, facts, fts, Fb, n_threads);
            if (flow.empty()) {
                LOG_ERROR("abot session: forward failed (block %d step %zu)", block, s);
                return false;
            }

            const int H = static_cast<int>(runner->lat_h), W = static_cast<int>(runner->lat_w);
            const int C = static_cast<int>(runner->lat_c);
            const int h_len = H / 2, w_len = W / 2;
            std::vector<float> x0(static_cast<size_t>(Fb) * fel);
            for (int f = 0; f < Fb; f++) {
                float sigma     = abot_sigma_of_t(fts[history.size() + f]);
                const float* xf = xt.data() + static_cast<size_t>(f) * fel;
                float* of       = x0.data() + static_cast<size_t>(f) * fel;
                for (int hh = 0; hh < h_len; hh++) {
                    for (int ww = 0; ww < w_len; ww++) {
                        const float* tok = flow.data() +
                                           ((static_cast<size_t>(f) * h_len + hh) * w_len + ww) * (4 * C);
                        for (int ph = 0; ph < 2; ph++) {
                            for (int pw = 0; pw < 2; pw++) {
                                for (int c = 0; c < C; c++) {
                                    size_t idx = (static_cast<size_t>(c) * H + (hh * 2 + ph)) * W + (ww * 2 + pw);
                                    of[idx]    = xf[idx] - sigma * tok[(ph * 2 + pw) * C + c];
                                }
                            }
                        }
                    }
                }
            }
            if (s + 1 < cfg.denoise_steps.size()) {
                const float sn = abot_sigma_of_t(cfg.denoise_steps[s + 1]);
                std::vector<float> eps(static_cast<size_t>(Fb) * fel);
                if (!noise_override || !noise_override(block, static_cast<int>(s), eps.data(), eps.size())) {
                    rng.fill(eps.data(), eps.size());
                }
                for (size_t i = 0; i < xt.size(); i++) {
                    xt[i] = (1.0f - sn) * x0[i] + sn * eps[i];
                }
            } else {
                xt = std::move(x0);
            }
            if (first) {
                memcpy(xt.data(), ff, fel * sizeof(float));  // keep frame 0 pinned
            }
        }

        out_frames.clear();
        for (int f = 0; f < Fb; f++) {
            out_frames.emplace_back(xt.begin() + static_cast<long long>(f) * fel,
                                    xt.begin() + static_cast<long long>(f + 1) * fel);
            history.push_back(out_frames.back());
            history_actions.push_back(action_mask);
        }
        return true;
    }

    // Decode the newest block to pixels with a decode_overlap-latent-frame
    // prefix from the previous block (taehv here is stateless per call; the
    // overlap replaces the reference's streaming decoder cache and its warmup
    // frames are dropped). A 3-frame overlap matches the reference streaming
    // decode at >33 dB PSNR on every frame (mean 46 dB) in local validation.
    // Returns {W_px, H_px, T_px, 3} floats in [0, 1].
    sd::Tensor<float> decode_last_block(int decode_overlap = 3) {
        const int Fb    = cfg.num_frame_per_block;
        const size_t fel = static_cast<size_t>(frame_elems());
        const int total = static_cast<int>(history.size());
        if (total < Fb) {
            return {};
        }
        const int lead = std::min(decode_overlap, total - Fb);
        const int T    = Fb + lead;
        const int64_t W = runner->lat_w, H = runner->lat_h, C = runner->lat_c;

        // wan layout {W, H, T, C}: per-channel frame planes
        sd::Tensor<float> z({W, H, T, C});
        for (int f = 0; f < T; f++) {
            const float* src = history[static_cast<size_t>(total - T + f)].data();
            for (int64_t c = 0; c < C; c++) {
                memcpy(z.data() + (c * T + f) * W * H, src + c * W * H,
                       static_cast<size_t>(W) * H * sizeof(float));
            }
        }
        sd_tiling_params_t no_tiling = {};
        sd::Tensor<float> px         = tae->decode(n_threads, z, no_tiling, true);
        if (px.empty()) {
            LOG_ERROR("abot session: taehv decode failed");
            return {};
        }
        // px {W_px, H_px, T_px, 3}: the taehv decoder emits [0, 1] pixels
        // directly (taesd convention). Drop warmup/overlap frames: stateless
        // decode of T latents yields 4T-3 px frames; the last 4*Fb belong to
        // the current block (or 4*Fb-3 when there is no lead block).
        const int64_t Wp = px.shape()[0], Hp = px.shape()[1], Tp = px.shape()[2];
        const int64_t keep  = lead > 0 ? 4 * Fb : Tp;
        const int64_t start = Tp - keep;
        sd::Tensor<float> out({Wp, Hp, keep, 3});
        for (int64_t ch = 0; ch < 3; ch++) {
            for (int64_t t = 0; t < keep; t++) {
                const float* sp = px.data() + (ch * Tp + start + t) * Wp * Hp;
                float* dp       = out.data() + (ch * keep + t) * Wp * Hp;
                for (int64_t i = 0; i < Wp * Hp; i++) {
                    dp[i] = std::min(std::max(sp[i], 0.0f), 1.0f);
                }
            }
        }
        return out;
    }

    // Convenience: step + decode to 8-bit RGB frames (row-major, interleaved).
    bool step(uint8_t action_mask,
              std::vector<std::vector<uint8_t>>& rgb_frames,
              int64_t& px_w,
              int64_t& px_h) {
        std::vector<std::vector<float>> latents;
        if (!step_latents(action_mask, latents)) {
            return false;
        }
        sd::Tensor<float> px = decode_last_block();
        if (px.empty()) {
            return false;
        }
        px_w = px.shape()[0];
        px_h = px.shape()[1];
        const int64_t T = px.shape()[2];
        rgb_frames.assign(static_cast<size_t>(T), {});
        for (int64_t t = 0; t < T; t++) {
            auto& fr = rgb_frames[static_cast<size_t>(t)];
            fr.resize(static_cast<size_t>(px_w) * px_h * 3);
            for (int64_t ch = 0; ch < 3; ch++) {
                const float* sp = px.data() + (ch * T + t) * px_w * px_h;
                for (int64_t i = 0; i < px_w * px_h; i++) {
                    fr[static_cast<size_t>(i) * 3 + ch] =
                        static_cast<uint8_t>(sp[i] * 255.0f + 0.5f);
                }
            }
        }
        return true;
    }
};

}  // namespace ABOT

#endif  // __ABOT_WORLD_HPP__
