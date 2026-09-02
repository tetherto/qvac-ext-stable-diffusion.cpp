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

#include <array>
#include <functional>
#include <random>
#include <thread>

#include "core/ggml_extend.hpp"
#include "model.h"
#include "model_manager.h"
#include "model/common/rope.hpp"
#include "model/vae/vae.hpp"  // must precede tae.hpp (TinyVideoAutoEncoder's base)
#include "model/vae/tae.hpp"
#include "model/diffusion/wan.hpp"

namespace ABOT {

struct AbotWorldConfig {
    int num_frame_per_block = 3;
    int local_attn_size     = 8;      // latent-frame window (test config; deployed default 21)
    int rope_temporal_clamp = 20;     // reference clamps window-local temporal ids to <= 20
    float context_noise_t   = 0.0f;   // history frames' timestep
    std::vector<float> denoise_steps = {1000.0f, 937.5f, 833.3333333f, 625.0f};
    int act_in_dim           = 32;    // 8 keys x 4 (repeat_interleave)
    int act_downscale_factor = 16;
    // Frames of finalized history fed into each block's graph, in addition to
    // walk block 0 (always pinned: ref rows re-derive their K/V against it).
    // 0 = local_attn_size (bounded memory: the current block's window is exact,
    // older frames' K/V lose context beyond the kept set - a second-order
    // effect; for walks <= local_attn_size + num_frame_per_block frames the
    // kept set equals the full history, so short walks are unchanged).
    // < 0 = unbounded (previous behavior: full-history recompute, VRAM grows
    // every block until OOM).
    int history_keep = 0;
    // Per-layer history KV cache: capture finalized-block K/V once instead of
    // recomputing them every denoise step. Requires local_attn_size -
    // num_frame_per_block <= AbotWorldRunner::kv_ring_slots (validated at
    // session load; the ring is compile-time sized).
    bool kv_cache = false;
    // Per-stage timing logs ([prof] lines). ABOT_PROF=1 also enables, so the
    // switch stays reachable without an API change in field debugging.
    bool profile = false;
};

static inline bool abot_prof_env() {
    static const bool v = std::getenv("ABOT_PROF") != nullptr;
    return v;
}

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
    // prompt rows before the zeroed padding (diagnostic; 0 = not computed)
    int64_t text_rows_live = 0;
    // false = text-only scene: block-0 frame 0 is generated from noise
    // instead of being pinned to first_frame_latents
    bool has_first_frame = true;

    // Minimal safetensors reader (F32 tensors only — the scene pack is fp32).
    // Torch shapes map to ggml ne reversed; trailing singleton dims dropped.
    // Packs are portable artifacts (downloaded reference packs, packs shared
    // between machines), so every field read from the file is validated as
    // untrusted input: bounded header, npos-checked parsing, positive dims
    // with an overflow-safe element cap, per-tensor rank/layout checks, and
    // data ranges checked against the file size.
    bool load(const std::string& path) {
        // scene-pack headers are ~1 KB; the element cap is far above the
        // largest legitimate tensor (prompt_embeds, 2M floats) but bounds a
        // hostile allocation to 1 GiB
        constexpr uint64_t MAX_HEADER  = 16ull * 1024 * 1024;
        constexpr int64_t MAX_ELEMENTS = int64_t(1) << 28;
        constexpr int64_t MAX_REF_SLOTS = 64;

        std::ifstream f(path, std::ios::binary);
        if (!f) {
            LOG_ERROR("scene pack: cannot open '%s'", path.c_str());
            return false;
        }
        f.seekg(0, std::ios::end);
        const int64_t file_size = static_cast<int64_t>(f.tellg());
        f.seekg(0);
        uint64_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 8);
        if (f.gcount() != 8 || hlen == 0 || hlen > MAX_HEADER ||
            static_cast<int64_t>(8 + hlen) > file_size) {
            LOG_ERROR("scene pack: invalid header length in '%s'", path.c_str());
            return false;
        }
        std::string header(hlen, '\0');
        f.read(header.data(), static_cast<std::streamsize>(hlen));
        if (static_cast<uint64_t>(f.gcount()) != hlen) {
            LOG_ERROR("scene pack: truncated header in '%s'", path.c_str());
            return false;
        }
        const uint64_t data_base = 8 + hlen;

        // false = absent OR malformed; `bad` distinguishes (malformed packs
        // must fail the load even for optional tensors)
        bool bad   = false;
        auto fetch = [&](const std::string& name, sd::Tensor<float>& out,
                         std::vector<int64_t>& torch_shape) -> bool {
            size_t p = header.find("\"" + name + "\"");
            if (p == std::string::npos) {
                return false;
            }
            const size_t sp = header.find("\"shape\"", p);
            const size_t sb = sp == std::string::npos ? std::string::npos : header.find('[', sp);
            const size_t se = sb == std::string::npos ? std::string::npos : header.find(']', sb);
            if (se == std::string::npos) {
                LOG_ERROR("scene pack: tensor '%s' has no shape", name.c_str());
                bad = true;
                return false;
            }
            torch_shape.clear();
            for (size_t q = sb + 1; q < se;) {
                while (q < se && (header[q] == ' ' || header[q] == ',')) q++;
                if (q >= se) break;
                torch_shape.push_back(strtoll(header.c_str() + q, nullptr, 10));
                while (q < se && header[q] != ',') q++;
            }
            const size_t op = header.find("\"data_offsets\"", p);
            const size_t ob = op == std::string::npos ? std::string::npos : header.find('[', op);
            if (ob == std::string::npos) {
                LOG_ERROR("scene pack: tensor '%s' has no data_offsets", name.c_str());
                bad = true;
                return false;
            }
            const uint64_t off0 = strtoull(header.c_str() + ob + 1, nullptr, 10);
            int64_t n           = 1;
            for (auto s : torch_shape) {
                if (s <= 0 || n > MAX_ELEMENTS / s) {
                    LOG_ERROR("scene pack: tensor '%s' has an invalid shape", name.c_str());
                    bad = true;
                    return false;
                }
                n *= s;
            }
            if (off0 > static_cast<uint64_t>(file_size) ||
                data_base + off0 + static_cast<uint64_t>(n) * sizeof(float) >
                    static_cast<uint64_t>(file_size)) {
                LOG_ERROR("scene pack: tensor '%s' data exceeds the file", name.c_str());
                bad = true;
                return false;
            }
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
            if (!f.good()) {
                LOG_ERROR("scene pack: tensor '%s' read failed", name.c_str());
                bad = true;
                return false;
            }
            return true;
        };
        // rank + leading-singleton layout check per named tensor: the resizes
        // below index fixed positions and reinterpret the buffer contiguously
        auto expect = [&](const char* name, const std::vector<int64_t>& sh,
                          size_t rank, std::initializer_list<size_t> one_dims) -> bool {
            if (sh.size() != rank) {
                LOG_ERROR("scene pack: tensor '%s' has rank %zu, expected %zu",
                          name, sh.size(), rank);
                return false;
            }
            for (size_t d : one_dims) {
                if (sh[d] != 1) {
                    LOG_ERROR("scene pack: tensor '%s' dim %zu must be 1", name, d);
                    return false;
                }
            }
            return true;
        };

        std::vector<int64_t> sh;
        if (!fetch("prompt_embeds", prompt_embeds, sh) ||          // [1,512,4096]
            !expect("prompt_embeds", sh, 3, {0})) {
            return false;
        }
        prompt_embeds.resize({sh[2], sh[1], 1, 1});
        // Real prompt rows: the producer zeroes everything past the last token
        // (the reference's `u[v:] = 0`), so trailing all-zero rows are padding.
        // Reported because a pack whose padding is NOT zeroed conditions the
        // walk on pad-token embeddings and degrades generation from the first
        // block - a silent failure that is otherwise only visible in the
        // output pixels.
        {
            const int64_t emb = prompt_embeds.shape()[0];
            const int64_t rows = prompt_embeds.shape()[1];
            const float* pd = prompt_embeds.data();
            int64_t live = 0;
            for (int64_t r = rows - 1; r >= 0; r--) {
                bool nonzero = false;
                for (int64_t i = 0; i < emb; i++) {
                    if (pd[r * emb + i] != 0.0f) {
                        nonzero = true;
                        break;
                    }
                }
                if (nonzero) {
                    live = r + 1;
                    break;
                }
            }
            text_rows_live = live;
            if (live == rows) {
                LOG_WARN(
                    "scene pack: all %lld prompt rows are non-zero - padding is not zeroed, so the "
                    "walk is conditioned on pad embeddings (expect washed-out output; the pack "
                    "producer is missing the reference's zero-padding step)",
                    (long long)rows);
            } else {
                LOG_INFO("scene pack: prompt rows %lld live / %lld", (long long)live, (long long)rows);
            }
        }
        if (!fetch("first_frame_latents", first_frame_latents, sh) ||  // [1,1,C,H,W]
            !expect("first_frame_latents", sh, 5, {0, 1})) {
            return false;
        }
        first_frame_latents.resize({sh[4], sh[3], sh[2], 1});
        // optional first_frame_mask: 0 = text-only scene (do NOT pin block-0
        // frame 0; it is generated from noise). Missing = 1 (back-compat).
        {
            sd::Tensor<float> ffm;
            std::vector<int64_t> msh;
            if (fetch("first_frame_mask", ffm, msh) && ffm.numel() >= 1) {
                has_first_frame = ffm.data()[0] > 0.5f;
            } else if (bad) {
                return false;
            }
        }
        if (fetch("ref_latents", ref_latents, sh)) {               // [1,K,C,1,32,32]
            if (!expect("ref_latents", sh, 6, {0, 3}) || sh[1] > MAX_REF_SLOTS) {
                return false;
            }
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
            if (!fetch("ref_mask", ref_mask, msh) || ref_mask.numel() != ref_slots) {
                LOG_ERROR("scene pack: ref_mask missing or does not match ref_latents");
                return false;
            }
            ref_mask.resize({ref_slots});
        } else if (bad) {
            return false;
        }
        return true;
    }
};

// Wan subclass: same weights/blocks, causal recompute graph.
class AbotWan : public WAN::Wan {
public:
    AbotWan() = default;
    explicit AbotWan(WAN::WanConfig params)
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
    // Optional KV-cache hooks (ABot walk):
    //  - kv_ctx_provider(layer, &k_ctx, &v_ctx): supplies cached context K/V for
    //    the layer ({d_head, T_ctx, n_head} / {T_ctx, d_head, n_head}); when set,
    //    the layer runs forward_cached and attends [ctx | rows].
    //  - kv_capture_sink(layer, k_cur, v_cur): receives this graph's roped K /
    //    per-token V of the visible rows for persisting into the cache.
    // Both null -> the original full-recompute path, bit-for-bit.
    using KvCtxProvider = std::function<void(int, ggml_tensor**, ggml_tensor**)>;
    using KvCaptureSink = std::function<void(int, ggml_tensor*, ggml_tensor*)>;

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
                                int64_t& out_w_len,
                                const KvCtxProvider& kv_ctx_provider = nullptr,
                                const KvCaptureSink& kv_capture_sink = nullptr) {
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
        ggml_tensor* e_tbl = ggml_ext_timestep_embedding(gctx, t_frames, config.freq_dim);  // [freq_dim, F_vis+1]
        e_tbl              = time_embedding_0->forward(ctx, e_tbl);
        e_tbl              = ggml_silu_inplace(gctx, e_tbl);
        e_tbl              = time_embedding_2->forward(ctx, e_tbl);  // [dim, F_vis+1]  (this is `e`)
        ggml_tensor* e0_tbl = ggml_silu(gctx, e_tbl);
        e0_tbl              = time_projection_1->forward(ctx, e0_tbl);  // [6*dim, F_vis+1]

        ggml_tensor* e0_tok = ggml_get_rows(gctx, e0_tbl, token_frame_rows);            // [6*dim, L]
        e0_tok              = ggml_reshape_4d(gctx, e0_tok, config.dim, 6, L, 1);        // [N=1, L, 6, dim]

        // ── context ──
        ggml_tensor* c = text_embedding_0->forward(ctx, context);
        c              = ggml_ext_gelu(gctx, c);
        c              = text_embedding_2->forward(ctx, c);  // [dim, 512, 1]

        // ── transformer ──
        for (int i = 0; i < config.num_layers; i++) {
            auto block = std::dynamic_pointer_cast<WAN::WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);
            if (kv_ctx_provider == nullptr && kv_capture_sink == nullptr) {
                x = block->forward(ctx, x, e0_tok, pe, c, 0, attn_mask);
                continue;
            }

            ggml_tensor* k_ctx = nullptr;
            ggml_tensor* v_ctx = nullptr;
            if (kv_ctx_provider != nullptr) {
                kv_ctx_provider(i, &k_ctx, &v_ctx);
            }
            ggml_tensor* k_cur = nullptr;
            ggml_tensor* v_cur = nullptr;
            x = block->forward_cached(ctx,
                                      x,
                                      e0_tok,
                                      pe,
                                      c,
                                      0,
                                      attn_mask,
                                      k_ctx,
                                      v_ctx,
                                      kv_capture_sink != nullptr ? &k_cur : nullptr,
                                      kv_capture_sink != nullptr ? &v_cur : nullptr);
            if (kv_capture_sink != nullptr) {
                kv_capture_sink(i, k_cur, v_cur);
            }
        }

        // ── head over the trailing block tokens only ──
        const int64_t block_tokens = static_cast<int64_t>(block_frames) * fsl;
        ggml_tensor* xb            = ggml_ext_slice(gctx, x, 1, L - block_tokens, L);  // [dim, block_tokens, 1]
        // per-frame e for the block frames: rows F_vis-block_frames .. F_vis-1
        ggml_tensor* e_blk = ggml_ext_slice(gctx, e_tbl, 1, F_vis - block_frames, F_vis);  // [dim, block_frames]
        e_blk              = ggml_reshape_3d(gctx, e_blk, config.dim, block_frames, 1);     // [N=1, T, dim]
        ggml_tensor* out   = head->forward(ctx, xb, e_blk);  // [pt*ph*pw*out_dim, block_tokens, 1]
        return out;
    }
};

// Runner: owns the AbotWan blocks + walk state, builds one graph per denoise
// forward, and runs the host-side 4-step block loop.
struct AbotWorldRunner : public GGMLRunner {
    AbotWorldConfig cfg;
    WAN::WanConfig wan_params;
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
                    const AbotWorldConfig& config = {},
                    std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager), cfg(config) {
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
    // reference's _build_ref_freqs), video frames at their ABSOLUTE frame ids
    // (the reference's REL_ROPE_CACHE fast path with base 0; valid for walks
    // well below the re-base threshold of ~256 frames). With a trimmed history
    // the visible frames are non-contiguous; absolute ids keep every pairwise
    // rotary distance identical to the full-history graph.
    std::vector<float> build_pe(int n_ref_slots, int ref_grid,
                                const std::vector<int64_t>& frame_abs_ids, int h_len, int w_len) {
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
        for (int64_t abs_f : frame_abs_ids) {
            for (int hh = 0; hh < h_len; hh++) {
                for (int ww = 0; ww < w_len; ww++) {
                    ids.push_back({static_cast<float>(abs_f), static_cast<float>(hh), static_cast<float>(ww)});
                }
            }
        }
        return Rope::embed_nd(ids, 1, static_cast<float>(wan_params.theta), wan_params.axes_dim);
    }

    // Additive mask {L(k), L(q)}: video rows of block b attend ref + frames of
    // blocks <= b inside b's trailing window; ref rows attend ref + block 0.
    // Frames are identified by their ABSOLUTE walk ids (frame_abs_ids per
    // visible slot) so the window rules stay correct with a trimmed history;
    // with the full history present the mask is identical to the untrimmed one.
    std::vector<float> build_mask(int n_ref, const std::vector<int64_t>& frame_abs_ids, int fsl) {
        const int F_vis = static_cast<int>(frame_abs_ids.size());
        const int64_t L = n_ref + static_cast<int64_t>(F_vis) * fsl;
        std::vector<float> mask(static_cast<size_t>(L) * L, -INFINITY);
        auto frame_of = [&](int64_t tok) -> int {  // visible slot; -1 for ref tokens
            return tok < n_ref ? -1 : static_cast<int>((tok - n_ref) / fsl);
        };
        for (int64_t q = 0; q < L; q++) {
            int fq         = frame_of(q);
            int64_t abs_fq = fq < 0 ? 0 : frame_abs_ids[fq];
            int64_t bq     = fq < 0 ? 0 : abs_fq / cfg.num_frame_per_block;
            // trailing window (in absolute frames) as seen when block bq was generated
            int64_t hi = (bq + 1) * cfg.num_frame_per_block;  // frames [0, hi)
            int64_t lo = std::max<int64_t>(0, hi - cfg.local_attn_size);
            for (int64_t k = 0; k < L; k++) {
                int fk       = frame_of(k);
                bool allowed = false;
                if (fk < 0) {
                    allowed = true;  // ref keys visible to everyone
                } else if (fq < 0) {
                    allowed = frame_abs_ids[fk] < cfg.num_frame_per_block;  // ref rows formed with block 0
                } else {
                    int64_t abs_fk = frame_abs_ids[fk];
                    allowed        = abs_fk >= lo && abs_fk < hi;
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
                                   const std::vector<int64_t>& frame_abs_ids,  // F_vis absolute walk frame ids
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

        pe_vec                  = build_pe(scene.ref_slots, 16, frame_abs_ids, h_len, w_len);
        std::vector<float> mask = build_mask(n_ref, frame_abs_ids, fsl);
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

        // keep the compute buffer and its graph allocator across steps: the walk
        // runs 5-6 graphs per block forever, and the defaults would free and
        // re-reserve multi-GB of VRAM on every one of them (ggml re-reserves
        // automatically when the graph shape changes between denoise and append)
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
        if (!result.has_value()) {
            return {};
        }
        return std::move(*result);
    }

    // ── KV-cache walk path ───────────────────────────────────────────────────
    // Cache layout (fixed): per layer i,
    //   abot.kv.l{i}.{k,v}.base : refs + walk block 0 (n_ref + Fb*fsl tokens)
    //   abot.kv.l{i}.{k,v}.r{s} : one finalized frame per ring slot, s in
    //                             [0, ring_slots) rotating oldest-first
    // K stored {d_head, T, n_head} (roped), V stored {T, d_head, n_head}.
    // The mask column layout every cached graph uses:
    //   [ base(refs + block0) | r0..r{R-1} | current rows ]
    static constexpr int kv_ring_slots = 5;  // window(8) - fpb(3): the current
                                             // block's exact trailing window

    enum class KvMode {
        INIT_CAPTURE,  // rows = [refs | clean block0]; captures base, no read
        DENOISE,       // rows = current noisy block; reads cache
        APPEND,        // rows = current clean block (t=0); reads cache, captures ring
    };

    static std::string kv_name(int layer, bool is_k, const std::string& slot) {
        char buf[64];
        snprintf(buf, sizeof(buf), "abot.kv.l%02d.%s.%s", layer, is_k ? "k" : "v", slot.c_str());
        return buf;
    }

    // Additive mask {T_kv, L_q} for the cached graphs. Columns follow the fixed
    // cache layout; rows are the current block's tokens. Window rules match
    // build_mask (absolute frame ids); unwritten ring slots (abs < 0) and
    // out-of-window frames get -INF.
    std::vector<float> build_mask_kv(int n_ref, int fsl,
                                     const std::array<int64_t, kv_ring_slots>& ring_abs,
                                     const std::vector<int64_t>& cur_abs) {
        const int Fb       = cfg.num_frame_per_block;
        const int64_t L_q  = static_cast<int64_t>(cur_abs.size()) * fsl;
        const int64_t T_kv = n_ref + static_cast<int64_t>(Fb + kv_ring_slots + cur_abs.size()) * fsl;
        std::vector<float> mask(static_cast<size_t>(T_kv) * L_q, -INFINITY);
        // per-column absolute frame id; -1 = ref (always visible), -2 = empty
        std::vector<int64_t> col_abs(static_cast<size_t>(T_kv));
        int64_t c = 0;
        for (int i = 0; i < n_ref; i++) {
            col_abs[c++] = -1;
        }
        for (int f = 0; f < Fb; f++) {  // pinned block 0
            std::fill_n(col_abs.begin() + c, fsl, static_cast<int64_t>(f));
            c += fsl;
        }
        for (int s = 0; s < kv_ring_slots; s++) {
            std::fill_n(col_abs.begin() + c, fsl, ring_abs[static_cast<size_t>(s)] < 0 ? -2 : ring_abs[static_cast<size_t>(s)]);
            c += fsl;
        }
        for (int64_t a : cur_abs) {
            std::fill_n(col_abs.begin() + c, fsl, a);
            c += fsl;
        }
        for (int64_t q = 0; q < L_q; q++) {
            const int64_t aq = cur_abs[static_cast<size_t>(q / fsl)];
            const int64_t bq = aq / Fb;
            const int64_t hi = (bq + 1) * Fb;
            const int64_t lo = std::max<int64_t>(0, hi - cfg.local_attn_size);
            float* row       = mask.data() + static_cast<size_t>(q) * T_kv;
            for (int64_t k = 0; k < T_kv; k++) {
                const int64_t ak = col_abs[static_cast<size_t>(k)];
                if (ak == -1 || (ak >= lo && ak < hi)) {
                    row[k] = 0.0f;
                }
            }
        }
        return mask;
    }

    // One cached-walk forward. DENOISE returns flow tokens for the block;
    // INIT_CAPTURE/APPEND return the (discarded) flow but persist K/V.
    sd::Tensor<float> forward_step_kv(KvMode mode,
                                      const std::vector<const float*>& frame_latents,  // Fb pointers
                                      uint8_t action_mask,
                                      const std::vector<float>& frame_timesteps,       // Fb values
                                      const std::vector<int64_t>& frame_abs_ids,       // Fb values
                                      const std::array<int64_t, kv_ring_slots>& ring_abs,
                                      const std::vector<int>& ring_write_slots,        // APPEND: slot per frame
                                      int n_threads) {
        const int Fb    = cfg.num_frame_per_block;
        const int F_cur = static_cast<int>(frame_latents.size());
        const int ds    = cfg.act_downscale_factor;
        const int c_unsh = cfg.act_in_dim * ds * ds;
        int h_len       = static_cast<int>(lat_h) / 2;
        int w_len       = static_cast<int>(lat_w) / 2;
        int fsl         = h_len * w_len;
        const bool with_refs = mode == KvMode::INIT_CAPTURE;
        // ref tokens present as graph ROWS only in the init pass; as mask/cache
        // COLUMNS they are always part of the cached base
        const int n_ref_cols = scene.ref_slots * 16 * 16;
        int n_ref            = with_refs ? n_ref_cols : 0;

        sd::Tensor<float> x_all({lat_w, lat_h, F_cur, lat_c});
        for (int f = 0; f < F_cur; f++) {
            for (int64_t ch = 0; ch < lat_c; ch++) {
                float* dst       = x_all.data() + (ch * F_cur + f) * lat_w * lat_h;
                const float* src = frame_latents[static_cast<size_t>(f)] + ch * lat_w * lat_h;
                memcpy(dst, src, static_cast<size_t>(lat_w) * lat_h * sizeof(float));
            }
        }
        sd::Tensor<float> act({lat_w, lat_h, c_unsh, F_cur});
        for (int f = 0; f < F_cur; f++) {
            fill_act_plane(act.data() + static_cast<size_t>(f) * c_unsh * lat_w * lat_h,
                           action_mask, static_cast<int>(lat_w), static_cast<int>(lat_h), c_unsh);
        }
        sd::Tensor<float> tvec({F_cur + 1});
        for (int f = 0; f < F_cur; f++) {
            tvec.data()[f] = frame_timesteps[static_cast<size_t>(f)];
        }
        tvec.data()[F_cur] = 0.0f;

        std::vector<int32_t> rows(static_cast<size_t>(n_ref) + static_cast<size_t>(F_cur) * fsl);
        for (int i = 0; i < n_ref; i++) {
            rows[i] = F_cur;
        }
        for (int f = 0; f < F_cur; f++) {
            std::fill(rows.begin() + n_ref + static_cast<size_t>(f) * fsl,
                      rows.begin() + n_ref + static_cast<size_t>(f + 1) * fsl, f);
        }

        const bool prof       = cfg.profile || abot_prof_env();
        const int64_t prof_t0 = prof ? ggml_time_ms() : 0;

        pe_vec = build_pe(with_refs ? scene.ref_slots : 0, 16, frame_abs_ids, h_len, w_len);
        std::vector<float> mask;
        if (mode == KvMode::INIT_CAPTURE) {
            mask = build_mask(n_ref, frame_abs_ids, fsl);  // original full block-0 mask
        } else {
            mask = build_mask_kv(n_ref_cols, fsl, ring_abs, frame_abs_ids);
        }
        const int64_t prof_t1 = prof ? ggml_time_ms() : 0;
        const int64_t L_q  = n_ref + static_cast<int64_t>(F_cur) * fsl;
        const int64_t T_kv = static_cast<int64_t>(mask.size()) / L_q;

        // zero fill for unwritten ring slots (shared per graph)
        sd::Tensor<float> zero_k({static_cast<int64_t>(wan_params.dim) / wan_params.num_heads,
                                  static_cast<int64_t>(fsl), wan_params.num_heads});
        sd::Tensor<float> zero_v({static_cast<int64_t>(fsl),
                                  static_cast<int64_t>(wan_params.dim) / wan_params.num_heads,
                                  wan_params.num_heads});
        std::fill_n(zero_k.data(), zero_k.numel(), 0.0f);
        std::fill_n(zero_v.data(), zero_v.numel(), 0.0f);

        int64_t out_h = 0, out_w = 0;
        auto get_graph = [&]() -> ggml_cgraph* {
            ggml_cgraph* gf     = ggml_new_graph_custom(compute_ctx, WAN::WAN_GRAPH_SIZE, false);
            ggml_tensor* x_in   = to_backend_input(x_all);
            ggml_tensor* act_in = to_backend_input(act);
            ggml_tensor* t_in   = to_backend_input(tvec);
            ggml_tensor* ctx_in = to_backend_input(scene.prompt_embeds);
            ggml_tensor* ref_in = nullptr;
            ggml_tensor* rm_in  = nullptr;
            if (with_refs && scene.ref_slots > 0) {
                ref_in = to_backend_input(scene.ref_latents);
                rm_in  = to_backend_input(scene.ref_mask);
            }
            ggml_tensor* pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2,
                                                 wan_params.axes_dim_sum / 2,
                                                 static_cast<int64_t>(pe_vec.size()) / wan_params.axes_dim_sum / 2);
            set_backend_tensor_data(pe, pe_vec.data());
            ggml_tensor* mk = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, T_kv, L_q);
            set_backend_tensor_data(mk, mask.data());
            ggml_tensor* rw = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, static_cast<int64_t>(rows.size()));
            set_backend_tensor_data(rw, rows.data());

            ggml_tensor* zk_in = nullptr;
            ggml_tensor* zv_in = nullptr;
            if (mode != KvMode::INIT_CAPTURE) {
                zk_in = to_backend_input(zero_k);
                zv_in = to_backend_input(zero_v);
            }

            auto runner_ctx = get_context();

            AbotWan::KvCtxProvider provider = nullptr;
            if (mode != KvMode::INIT_CAPTURE) {
                provider = [&, zk_in, zv_in](int layer, ggml_tensor** k_ctx, ggml_tensor** v_ctx) {
                    ggml_tensor* k_all = runner_ctx.get_cache_tensor(kv_name(layer, true, "base"));
                    ggml_tensor* v_all = runner_ctx.get_cache_tensor(kv_name(layer, false, "base"));
                    GGML_ASSERT(k_all != nullptr && v_all != nullptr);
                    for (int s = 0; s < kv_ring_slots; s++) {
                        ggml_tensor* ks = nullptr;
                        ggml_tensor* vs = nullptr;
                        if (ring_abs[static_cast<size_t>(s)] >= 0) {
                            ks = runner_ctx.get_cache_tensor(kv_name(layer, true, "r" + std::to_string(s)));
                            vs = runner_ctx.get_cache_tensor(kv_name(layer, false, "r" + std::to_string(s)));
                        }
                        k_all = ggml_concat(compute_ctx, k_all, ks != nullptr ? ks : zk_in, 1);
                        v_all = ggml_concat(compute_ctx, v_all, vs != nullptr ? vs : zv_in, 0);
                    }
                    *k_ctx = k_all;
                    *v_ctx = v_all;
                };
            }

            AbotWan::KvCaptureSink sink = nullptr;
            std::vector<std::pair<std::string, ggml_tensor*>>* captures = &pending_kv_captures;
            captures->clear();
            if (mode == KvMode::INIT_CAPTURE) {
                sink = [&, captures](int layer, ggml_tensor* k_cur, ggml_tensor* v_cur) {
                    captures->push_back({kv_name(layer, true, "base"), k_cur});
                    captures->push_back({kv_name(layer, false, "base"), v_cur});
                };
            } else if (mode == KvMode::APPEND) {
                sink = [&, captures](int layer, ggml_tensor* k_cur, ggml_tensor* v_cur) {
                    for (int f = 0; f < F_cur; f++) {
                        const std::string slot = "r" + std::to_string(ring_write_slots[static_cast<size_t>(f)]);
                        ggml_tensor* kf = ggml_ext_cont(compute_ctx,
                                                        ggml_ext_slice(compute_ctx, k_cur, 1, f * fsl, (f + 1) * fsl));
                        ggml_tensor* vf = ggml_ext_cont(compute_ctx,
                                                        ggml_ext_slice(compute_ctx, v_cur, 0, f * fsl, (f + 1) * fsl));
                        captures->push_back({kv_name(layer, true, slot), kf});
                        captures->push_back({kv_name(layer, false, slot), vf});
                    }
                };
            }

            ggml_tensor* out = wan.forward_causal(&runner_ctx, x_in, act_in, t_in, ctx_in,
                                                  ref_in, rm_in, pe, mk, rw, Fb, out_h, out_w,
                                                  provider, sink);
            ggml_build_forward_expand(gf, out);
            for (auto& cap : *captures) {
                // Captured K/V are also consumed by the attention itself, so
                // the graph allocator may hand their memory to later nodes once
                // those consumers ran; the cache persist only happens after the
                // full graph. Materialize views into their own storage and pin
                // every capture as a graph output so its bytes survive to the
                // post-compute cache copy.
                ggml_tensor* pinned = cap.second;
                if (pinned->view_src != nullptr) {
                    pinned = ggml_cont(compute_ctx, pinned);
                }
                ggml_set_output(pinned);
                ggml_build_forward_expand(gf, pinned);
                this->cache(cap.first, pinned);
            }
            return gf;
        };

        // keep the compute buffer and its graph allocator across steps: the walk
        // runs 5-6 graphs per block forever, and the defaults would free and
        // re-reserve multi-GB of VRAM on every one of them (ggml re-reserves
        // automatically when the graph shape changes between denoise and append)
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
        if (prof) {
            const int64_t prof_t2 = ggml_time_ms();
            const char* mode_s    = mode == KvMode::INIT_CAPTURE ? "init" : mode == KvMode::APPEND ? "append" : "denoise";
            LOG_INFO("[prof] kv %s: hostprep=%lldms compute=%lldms",
                     mode_s, (long long)(prof_t1 - prof_t0), (long long)(prof_t2 - prof_t1));
        }
        if (!result.has_value()) {
            return {};
        }
        return std::move(*result);
    }

    std::vector<std::pair<std::string, ggml_tensor*>> pending_kv_captures;

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
struct AbotTinyVideoAutoEncoder : public TinyVideoAutoEncoder {
    using TinyVideoAutoEncoder::TinyVideoAutoEncoder;
};

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
    std::shared_ptr<AbotTinyVideoAutoEncoder> tae;
    // Declared after the runners on purpose (mirrors StableDiffusionGGML):
    // ~ModelManager force-frees the param storage blocks and writes through
    // the registered ggml tensors (state->tensor->buffer = nullptr), which
    // live in the runner/tae contexts above. Members destroy in reverse
    // declaration order and the runners hold only weak_ptr refs to the
    // manager, so this ordering runs ~ModelManager first, while every
    // registered tensor is still alive.
    std::shared_ptr<ModelManager> model_manager;

    // finalized walk state (per-frame ggml {W,H,C} latents, torch [C,H,W] flat)
    std::vector<std::vector<float>> history;
    std::vector<uint8_t> history_actions;

    // KV-cache walk state (cfg.kv_cache): ring of finalized-frame K/V slots
    bool kv_enabled = false;
    // ggml backends are not thread-safe: the decode/append overlap is only
    // legal when DiT and taehv run on distinct backend instances (e.g.
    // "diffusion=cuda0,vae=cuda1"); on a shared instance (single-GPU Metal)
    // concurrent graph submission wedges the command queue.
    bool kv_decode_overlap_safe = false;
    std::array<int64_t, AbotWorldRunner::kv_ring_slots> kv_ring_abs{};
    int kv_ring_next = 0;
    // deferred cache append for the newest block (run after/parallel to decode)
    bool kv_append_pending = false;
    std::vector<std::vector<float>> kv_append_frames;
    uint8_t kv_append_action = 0;

    // Terminal failure state. A failed step is not retryable: the RNG has
    // advanced, and history / ring metadata / backend cache tensors may sit at
    // different logical block numbers (backend cache writes can be partially
    // applied when a graph fails, so rollback is not reliable). The session
    // poisons itself instead of returning as a reusable object; callers must
    // free it and create a new one.
    bool failed = false;

    bool fail_session(const char* reason) {
        if (!failed) {
            LOG_ERROR("abot session: terminal failure (%s); free and recreate the session", reason);
        }
        failed            = true;
        kv_append_pending = false;
        return false;
    }

    // Run the deferred cache-append pass for the newest finalized block.
    bool kv_run_append() {
        if (!kv_append_pending) {
            return true;
        }
        const int Fb        = cfg.num_frame_per_block;
        const int64_t total = static_cast<int64_t>(history.size());
        std::vector<const float*> frames;
        std::vector<float> fts;
        std::vector<int64_t> abs_ids;
        std::vector<int> write_slots;
        int next_ring_slot = kv_ring_next;
        for (int f = 0; f < Fb; f++) {
            frames.push_back(kv_append_frames[static_cast<size_t>(f)].data());
            fts.push_back(cfg.context_noise_t);
            abs_ids.push_back(total - Fb + f);
            write_slots.push_back(next_ring_slot);
            next_ring_slot = (next_ring_slot + 1) % AbotWorldRunner::kv_ring_slots;
        }
        // the append pass reads the PRE-append ring (slots being overwritten
        // this block are already outside every current row's window), so
        // kv_ring_abs must not be touched until the graph has succeeded
        auto flow = runner->forward_step_kv(AbotWorldRunner::KvMode::APPEND,
                                            frames, kv_append_action, fts, abs_ids,
                                            kv_ring_abs, write_slots, n_threads);
        if (flow.empty()) {
            return fail_session("kv append pass failed");
        }
        // commit host-side bookkeeping only after successful execution
        kv_ring_next = next_ring_slot;
        for (int f = 0; f < Fb; f++) {
            kv_ring_abs[static_cast<size_t>(write_slots[static_cast<size_t>(f)])] = abs_ids[static_cast<size_t>(f)];
        }
        kv_append_pending = false;
        return true;
    }

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

        model_manager = std::make_shared<ModelManager>();
        model_manager->set_n_threads(n_threads);
        ModelLoader& model_loader = model_manager->loader();
        if (!model_loader.init_from_file(dit_path, "model.diffusion_model.")) {
            LOG_ERROR("abot session: cannot open DiT '%s'", dit_path.c_str());
            return false;
        }
        runner = std::make_unique<AbotWorldRunner>(runtime_backend,
                                                    params_backend,
                                                    model_loader.get_tensor_storage_map(),
                                                    "model.diffusion_model",
                                                    cfg,
                                                    model_manager);  // no trailing dot: GGMLBlock::init appends its own

#ifdef SD_ABOT_FLASH_ATTN_DEBUG
        // Debug-only flash attention for the walk graph (ABOT_FLASH_ATTN=1 in
        // a build compiled with SD_ABOT_FLASH_ATTN_DEBUG). The masked
        // self-attention path supports it (2D additive mask), but it is
        // KNOWN BROKEN: parity passes for block 0 and collapses for any block
        // with history (cosine ~0.2 vs goldens on CUDA) - the flash path
        // mishandles the walk's history mask. Deliberately not reachable in
        // production builds until that is fixed.
        if (const char* fa = std::getenv("ABOT_FLASH_ATTN"); fa != nullptr && fa[0] == '1') {
            runner->set_flash_attention_enabled(true);
            LOG_WARN("abot session: DEBUG flash attention enabled - output is wrong for any block with history");
        }
#endif
        // Opt-in per-layer KV cache for the walk (params.kv_cache): history
        // K/V are captured once per finalized block instead of recomputed
        // every denoise step, so steady-state block graphs carry only the
        // current block's rows (~3.7x fewer frame-passes per block).
        if (cfg.kv_cache) {
            // The ring physically holds kv_ring_slots history frames beyond
            // the current block. A larger window would not just be masked -
            // its frames would be structurally absent as K/V columns, so the
            // walk would silently attend less context than configured.
            if (cfg.local_attn_size - cfg.num_frame_per_block > AbotWorldRunner::kv_ring_slots) {
                LOG_ERROR(
                    "abot session: kv_cache requires local_attn_size - num_frame_per_block <= %d "
                    "(the compiled KV ring size); got local_attn_size=%d, num_frame_per_block=%d. "
                    "Reduce the window or disable the KV cache.",
                    AbotWorldRunner::kv_ring_slots, cfg.local_attn_size, cfg.num_frame_per_block);
                return false;
            }
            kv_enabled = true;
            kv_ring_abs.fill(-1);
            kv_decode_overlap_safe = vae_backend != runtime_backend;
            LOG_INFO("abot session: KV cache enabled (ring %d frames, decode overlap %s)",
                     AbotWorldRunner::kv_ring_slots,
                     kv_decode_overlap_safe ? "on" : "off: shared DiT/taehv backend");
        }
        if (!model_loader.init_from_file(taehv_path, "tae.")) {  // same prefixing as new_sd_ctx's taesd path
            LOG_ERROR("abot session: cannot open taehv '%s'", taehv_path.c_str());
            return false;
        }
        tae = std::make_shared<AbotTinyVideoAutoEncoder>(vae_backend,
                                                          model_loader.get_tensor_storage_map(),
                                                          "decoder",
                                                          true,
                                                          VERSION_ABOT_WORLD,
                                                          model_manager);
        if (!model_manager->register_runner_params("ABot-World DiT",
                                                    *runner,
                                                    "model.diffusion_model",
                                                    ModelManager::ResidencyMode::ParamBackend,
                                                    runtime_backend,
                                                    params_backend) ||
            !model_manager->register_runner_params("ABot-World TAE",
                                                    *tae,
                                                    ModelManager::ResidencyMode::ParamBackend,
                                                    vae_backend,
                                                    vae_params_backend) ||
            !model_manager->validate_registered_tensors() ||
            !model_manager->load_all_params_eagerly()) {
            LOG_ERROR("abot session: model parameter registration or loading failed");
            return false;
        }
        if (!runner->scene.load(scene_path)) {
            return false;
        }
        auto ffl      = runner->scene.first_frame_latents.shape();
        runner->lat_w = ffl[0];
        runner->lat_h = ffl[1];
        runner->lat_c = ffl.size() > 2 ? ffl[2] : 1;
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
    // A false return is terminal (see fail_session): recreate the session.
    bool step_latents(uint8_t action_mask, std::vector<std::vector<float>>& out_frames) {
        if (failed) {
            LOG_ERROR("abot session: step called after terminal failure; recreate the session");
            return false;
        }
        const int Fb     = cfg.num_frame_per_block;
        const size_t fel = static_cast<size_t>(frame_elems());
        const bool first = history.empty();
        const int block  = static_cast<int>(history.size()) / Fb;
        const float* ff  = runner->scene.first_frame_latents.data();

        // text-only scenes (first_frame_mask = 0) generate block-0 frame 0
        // from noise instead of pinning it to the scene's first-frame latent
        const bool pin_first = first && runner->scene.has_first_frame;

        std::vector<float> xt(static_cast<size_t>(Fb) * fel);
        if (!noise_override || !noise_override(block, -1, xt.data(), xt.size())) {
            rng.fill(xt.data(), xt.size());
            if (pin_first) {
                memcpy(xt.data(), ff, fel * sizeof(float));  // pin frame 0 to the scene
            }
        }

        // Flush a deferred KV append (normally overlapped with decode in step())
        if (kv_enabled && !kv_run_append()) {
            return false;
        }

        // KV-cache fast path: blocks after the first attend cached context K/V,
        // so each denoise graph carries only the current block's rows. Block 0
        // (and the non-KV mode) uses the recompute graph below.
        const bool kv_fast = kv_enabled && !first;

        // Bounded history: feed walk block 0 (pinned - ref rows re-derive their
        // K/V against it, and its own window is just refs + itself, so it stays
        // exact) plus the trailing `keep` finalized frames (the current block's
        // full attention window). The union covers the whole history for short
        // walks, so goldens/parity are unaffected; beyond that the graph size
        // is constant instead of growing every block (which OOM'd a 32 GiB GPU
        // at block 5 at 832x480 F16). history_keep < 0 restores full history.
        std::vector<size_t> kept;
        const int keep = cfg.history_keep == 0 ? cfg.local_attn_size : cfg.history_keep;
        if (!kv_fast) {
            if (keep < 0) {
                for (size_t h = 0; h < history.size(); h++) {
                    kept.push_back(h);
                }
            } else {
                const size_t tail = history.size() > static_cast<size_t>(keep)
                                        ? history.size() - static_cast<size_t>(keep)
                                        : 0;
                for (size_t h = 0; h < history.size(); h++) {
                    if (h < static_cast<size_t>(Fb) || h >= tail) {
                        kept.push_back(h);
                    }
                }
            }
        }

        for (size_t s = 0; s < cfg.denoise_steps.size(); s++) {
            const float t_cur = cfg.denoise_steps[s];
            std::vector<const float*> frames;
            std::vector<uint8_t> facts;
            std::vector<float> fts;
            std::vector<int64_t> abs_ids;
            if (!kv_fast) {
                for (size_t h : kept) {
                    frames.push_back(history[h].data());
                    facts.push_back(history_actions[h]);
                    fts.push_back(cfg.context_noise_t);
                    abs_ids.push_back(static_cast<int64_t>(h));
                }
            }
            for (int f = 0; f < Fb; f++) {
                frames.push_back(xt.data() + static_cast<size_t>(f) * fel);
                facts.push_back(action_mask);
                fts.push_back(pin_first && f == 0 ? 0.0f : t_cur);
                abs_ids.push_back(static_cast<int64_t>(history.size()) + f);
            }

            auto flow = kv_fast
                            ? runner->forward_step_kv(AbotWorldRunner::KvMode::DENOISE,
                                                      frames, action_mask, fts, abs_ids,
                                                      kv_ring_abs, {}, n_threads)
                            : runner->forward_step(frames, facts, fts, abs_ids, Fb, n_threads);
            if (flow.empty()) {
                LOG_ERROR("abot session: forward failed (block %d step %zu)", block, s);
                return fail_session("denoise forward failed");
            }

            const int H = static_cast<int>(runner->lat_h), W = static_cast<int>(runner->lat_w);
            const int C = static_cast<int>(runner->lat_c);
            const int h_len = H / 2, w_len = W / 2;
            std::vector<float> x0(static_cast<size_t>(Fb) * fel);
            for (int f = 0; f < Fb; f++) {
                float sigma     = abot_sigma_of_t(fts[fts.size() - static_cast<size_t>(Fb) + f]);
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
            if (pin_first) {
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

        if (kv_enabled) {
            if (first) {
                // initial cache fill: refs + clean block 0 at t = 0
                std::vector<const float*> cframes;
                std::vector<float> cfts;
                std::vector<int64_t> cabs;
                for (int f = 0; f < Fb; f++) {
                    cframes.push_back(out_frames[static_cast<size_t>(f)].data());
                    cfts.push_back(cfg.context_noise_t);
                    cabs.push_back(f);
                }
                auto flow = runner->forward_step_kv(AbotWorldRunner::KvMode::INIT_CAPTURE,
                                                    cframes, action_mask, cfts, cabs,
                                                    kv_ring_abs, {}, n_threads);
                if (flow.empty()) {
                    // history is already committed for this block; without the
                    // base cache a later step would take the kv_fast path into
                    // a missing-tensor assert - poison instead
                    return fail_session("kv initial capture failed");
                }
            } else {
                // defer the ring append so step() can overlap it with decode
                kv_append_pending = true;
                kv_append_frames  = out_frames;
                kv_append_action  = action_mask;
            }
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
        const bool prof              = cfg.profile || abot_prof_env();
        const int64_t dec_t0         = prof ? ggml_time_ms() : 0;
        sd::Tensor<float> px         = tae->decode(n_threads, z, no_tiling, true);
        if (prof) {
            LOG_INFO("[prof] taehv decode (%d latents -> px): %lldms", T, (long long)(ggml_time_ms() - dec_t0));
        }
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
    // A false return is terminal (see fail_session): recreate the session.
    bool step(uint8_t action_mask,
              std::vector<std::vector<uint8_t>>& rgb_frames,
              int64_t& px_w,
              int64_t& px_h) {
        if (failed) {
            LOG_ERROR("abot session: step called after terminal failure; recreate the session");
            return false;
        }
        std::vector<std::vector<float>> latents;
        if (!step_latents(action_mask, latents)) {
            return false;
        }
        sd::Tensor<float> px;
        if (kv_enabled && kv_append_pending) {
            if (kv_decode_overlap_safe) {
                // overlap the taehv decode (vae backend; ideally a second GPU
                // via "vae=cuda1") with the KV cache-append pass (DiT backend)
                std::thread decode_thread([&]() { px = decode_last_block(); });
                const bool append_ok = kv_run_append();
                decode_thread.join();
                if (!append_ok) {
                    return false;
                }
            } else {
                // shared backend instance: serialize (see kv_decode_overlap_safe)
                if (!kv_run_append()) {
                    return false;
                }
                px = decode_last_block();
            }
        } else {
            px = decode_last_block();
        }
        if (px.empty()) {
            // history advanced but the block's frames are lost; the RNG has
            // moved on either way, so the walk can no longer be reproduced
            return fail_session("frame decode failed");
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

// ── Scene-pack creation ──────────────────────────────────────────────────────
// Builds the fixed world a session walks in, natively, from a prompt + first
// frame image — the offline step the reference performs with the full PyTorch
// pipeline (extract_scene.py). Components and their reference equivalents:
//   umT5-XXL encode  -> prompt_embeds [1, 512, 4096]
//       (WanTextEncoder: HF umt5 tokenizer, seq_len 512, attention-masked
//        encode, embeddings zeroed past the real tokens — reproduced by
//        T5CLIPEmbedder(is_umt5, use_mask) with zero_out_masked = true)
//   Wan2.2 VAE encode -> first_frame_latents [1, 1, 48, H/16, W/16]
//       (image scaled to cover + center-cropped to (width, height), mapped to
//        [-1, 1], encoded, then (z - mean) / std per channel — reproduced by
//        WanVAERunner encode + vae_to_diffusion_latents)
//   ref_latents/ref_mask -> zero-filled slots (the reference's documented
//        fallback when no reference images are provided)
// Output: a scene.safetensors (float32) identical in layout to the reference
// scene packs, loadable by AbotScenePack::load.

struct AbotSceneWriter {
    // Minimal safetensors writer (F32, C-order torch shapes) — the mirror of
    // AbotScenePack's reader.
    struct Entry {
        std::string name;
        std::vector<int64_t> torch_shape;
        const float* data;
        int64_t numel;
    };

    static bool write(const std::string& path, const std::vector<Entry>& entries) {
        std::string header = "{";
        uint64_t offset    = 0;
        for (size_t i = 0; i < entries.size(); i++) {
            const auto& e = entries[i];
            if (i > 0) {
                header += ",";
            }
            header += "\"" + e.name + "\":{\"dtype\":\"F32\",\"shape\":[";
            for (size_t d = 0; d < e.torch_shape.size(); d++) {
                if (d > 0) {
                    header += ",";
                }
                header += std::to_string(e.torch_shape[d]);
            }
            uint64_t nbytes = static_cast<uint64_t>(e.numel) * sizeof(float);
            header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
                      std::to_string(offset + nbytes) + "]}";
            offset += nbytes;
        }
        header += "}";
        // pad header to 8-byte alignment with spaces (safetensors convention)
        while (header.size() % 8 != 0) {
            header += " ";
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) {
            LOG_ERROR("scene writer: cannot open '%s'", path.c_str());
            return false;
        }
        uint64_t hlen = header.size();
        f.write(reinterpret_cast<const char*>(&hlen), 8);
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
        for (const auto& e : entries) {
            f.write(reinterpret_cast<const char*>(e.data),
                    static_cast<std::streamsize>(e.numel * sizeof(float)));
        }
        return f.good();
    }
};

struct AbotSceneCreateConfig {
    std::string prompt;      // encoded verbatim (the reference demo prefixes "| unknown | ")
    int width      = 832;    // pixel size; must be a multiple of 32 (16x VAE, 2x patch)
    int height     = 480;
    int ref_slots  = 5;      // zero-filled reference slots
    int ref_grid   = 32;     // latent grid per reference slot
    int latent_c   = 48;     // Wan2.2 VAE channels
};

// The builder itself (umT5 + Wan-VAE encodes) lives in stable-diffusion.cpp
// behind sd_abot_scene_create(): it needs the conditioner machinery that only
// that translation unit includes.

}  // namespace ABOT

#endif  // __ABOT_WORLD_HPP__
