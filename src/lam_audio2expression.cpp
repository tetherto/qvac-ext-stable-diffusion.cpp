#include "lam_audio2expression.hpp"

#include <cmath>
#include <cstring>
#include <thread>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

namespace {

constexpr const char* kArch = "lam-audio2exp";

uint32_t
ggufGetU32Or(struct gguf_context* g, const std::string& key, uint32_t dflt) {
  const int64_t idx = gguf_find_key(g, key.c_str());
  if (idx < 0 || gguf_get_kv_type(g, idx) != GGUF_TYPE_UINT32) {
    return dflt;
  }
  return gguf_get_val_u32(g, idx);
}

float
ggufGetF32Or(struct gguf_context* g, const std::string& key, float dflt) {
  const int64_t idx = gguf_find_key(g, key.c_str());
  if (idx < 0 || gguf_get_kv_type(g, idx) != GGUF_TYPE_FLOAT32) {
    return dflt;
  }
  return gguf_get_val_f32(g, idx);
}

// conv1d that preserves the kernel's precision: the stock ggml_conv_1d
// always runs im2col in F16, which breaks exact-f32 parity. This builds the
// same im2col + mul_mat pair with the dst type matching the kernel.
// kernel: [k, cIn, cOut], data: [t, cIn] → [tOut, cOut].
struct ggml_tensor*
conv1d(struct ggml_context* ctx, struct ggml_tensor* kernel,
       struct ggml_tensor* data, int stride, int padding) {
  struct ggml_tensor* cols =
      ggml_im2col(ctx, kernel, data, stride, 0, padding, 0, 1, 0,
                  /*is_2D*/ false, kernel->type); // [k*cIn, tOut, 1]
  struct ggml_tensor* out = ggml_mul_mat(
      ctx,
      ggml_reshape_2d(ctx, cols, cols->ne[0], cols->ne[1] * cols->ne[2]),
      ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1],
                      kernel->ne[2])); // [tOut, cOut]
  return out;
}

// Broadcast a 1-D bias [c] over the ne0 (time) axis of a [t, c] tensor.
struct ggml_tensor*
addBiasTimeMajor(struct ggml_context* ctx, struct ggml_tensor* x,
                 struct ggml_tensor* bias) {
  return ggml_add(ctx, x, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
}

// LayerNorm over ne0 (features) of a feature-major [c, t] tensor.
struct ggml_tensor*
layerNorm(struct ggml_context* ctx, struct ggml_tensor* x,
          struct ggml_tensor* w, struct ggml_tensor* b, float eps) {
  struct ggml_tensor* cur = ggml_norm(ctx, x, eps);
  cur = ggml_mul(ctx, cur, w);
  return ggml_add(ctx, cur, b);
}

} // namespace

LamAudio2Expression::~LamAudio2Expression() {
  if (weightBuffer_ != nullptr) {
    ggml_backend_buffer_free(weightBuffer_);
  }
  if (weightCtx_ != nullptr) {
    ggml_free(weightCtx_);
  }
  if (ownsBackend_ && backend_ != nullptr) {
    ggml_backend_free(backend_);
  }
}

struct ggml_tensor*
LamAudio2Expression::weight(const std::string& name) {
  auto it = weights_.find(name);
  return it == weights_.end() ? nullptr : it->second;
}

int64_t
LamAudio2Expression::frameCount(int64_t nSamples) const {
  return (nSamples * hparams_.fps + hparams_.sampleRate - 1) /
         hparams_.sampleRate;
}

int64_t
LamAudio2Expression::convOutLen(int64_t nSamples) const {
  int64_t len = nSamples;
  for (size_t i = 0; i < hparams_.feKernels.size(); ++i) {
    len = (len - hparams_.feKernels[i]) / hparams_.feStrides[i] + 1;
  }
  return len;
}

bool
LamAudio2Expression::load(const std::string& ggufPath, ggml_backend_t backend, int n_threads) {
  if (backend != nullptr) {
    backend_ = backend;
    ownsBackend_ = false;
  } else {
    backend_ = ggml_backend_cpu_init();
    ownsBackend_ = true;
    if (backend_ != nullptr) {
      const unsigned hw = std::thread::hardware_concurrency();
      const int threads = n_threads > 0
                              ? n_threads
                              : static_cast<int>(hw > 2 ? hw - 2 : 1);
      ggml_backend_cpu_set_n_threads(backend_, threads);
    }
  }
  if (backend_ == nullptr) {
    lastError_ = "failed to initialise backend";
    return false;
  }

  struct ggml_context* metaCtx = nullptr;
  struct gguf_init_params params = {/*no_alloc*/ true, /*ctx*/ &metaCtx};
  struct gguf_context* gguf = gguf_init_from_file(ggufPath.c_str(), params);
  if (gguf == nullptr) {
    lastError_ = "failed to open GGUF: " + ggufPath;
    return false;
  }

  const std::string arch = [&] {
    const int64_t idx = gguf_find_key(gguf, "general.architecture");
    return idx >= 0 ? std::string(gguf_get_val_str(gguf, idx)) : std::string();
  }();
  if (arch != kArch) {
    lastError_ = "unexpected architecture '" + arch + "'";
    gguf_free(gguf);
    ggml_free(metaCtx);
    return false;
  }

  const std::string pfx = std::string(kArch) + ".";
  hparams_.sampleRate = ggufGetU32Or(gguf, pfx + "sample_rate", 16000);
  hparams_.fps = ggufGetU32Or(gguf, pfx + "fps", 30);
  hparams_.nCoeffs = ggufGetU32Or(gguf, pfx + "n_coeffs", 52);
  hparams_.nIdentity = ggufGetU32Or(gguf, pfx + "n_identity", 12);
  hparams_.identityFeatDim = ggufGetU32Or(gguf, pfx + "identity_feat_dim", 64);
  hparams_.hiddenDim = ggufGetU32Or(gguf, pfx + "hidden_dim", 512);
  hparams_.windowFrames = ggufGetU32Or(gguf, pfx + "window_frames", 64);
  hparams_.layerNormEps = ggufGetF32Or(gguf, pfx + "layer_norm_eps", 1e-5F);
  hparams_.encLayers = ggufGetU32Or(gguf, pfx + "enc.n_layers", 12);
  hparams_.encHeads = ggufGetU32Or(gguf, pfx + "enc.n_heads", 12);
  hparams_.encHidden = ggufGetU32Or(gguf, pfx + "enc.hidden", 768);
  hparams_.encFfn = ggufGetU32Or(gguf, pfx + "enc.ffn", 3072);
  hparams_.posConvKernel = ggufGetU32Or(gguf, pfx + "enc.pos_conv_kernel", 128);
  hparams_.posConvGroups = ggufGetU32Or(gguf, pfx + "enc.pos_conv_groups", 16);

  const int64_t namesIdx = gguf_find_key(gguf, (pfx + "coeff_names").c_str());
  if (namesIdx >= 0 && gguf_get_kv_type(gguf, namesIdx) == GGUF_TYPE_ARRAY &&
      gguf_get_arr_type(gguf, namesIdx) == GGUF_TYPE_STRING) {
    const size_t n = gguf_get_arr_n(gguf, namesIdx);
    hparams_.coeffNames.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      hparams_.coeffNames.emplace_back(gguf_get_arr_str(gguf, namesIdx, i));
    }
  }

  // Copy tensor metadata into our own context, then load the data through
  // the backend buffer (works for CPU and GPU backends alike).
  const int64_t nTensors = gguf_get_n_tensors(gguf);
  const size_t ctxSize =
      (static_cast<size_t>(nTensors) + 1) * ggml_tensor_overhead();
  struct ggml_init_params wparams = {ctxSize, nullptr, /*no_alloc*/ true};
  weightCtx_ = ggml_init(wparams);

  for (struct ggml_tensor* meta = ggml_get_first_tensor(metaCtx);
       meta != nullptr; meta = ggml_get_next_tensor(metaCtx, meta)) {
    struct ggml_tensor* dst = ggml_dup_tensor(weightCtx_, meta);
    ggml_set_name(dst, ggml_get_name(meta));
    weights_[ggml_get_name(meta)] = dst;
  }

  weightBuffer_ = ggml_backend_alloc_ctx_tensors(weightCtx_, backend_);
  if (weightBuffer_ == nullptr) {
    lastError_ = "failed to allocate weight buffer";
    gguf_free(gguf);
    ggml_free(metaCtx);
    return false;
  }

  FILE* file = fopen(ggufPath.c_str(), "rb");
  if (file == nullptr) {
    lastError_ = "failed to reopen GGUF: " + ggufPath;
    gguf_free(gguf);
    ggml_free(metaCtx);
    return false;
  }
  const size_t dataOffset = gguf_get_data_offset(gguf);
  std::vector<uint8_t> readBuf;
  bool ok = true;
  for (int64_t i = 0; i < nTensors; ++i) {
    const char* name = gguf_get_tensor_name(gguf, i);
    struct ggml_tensor* dst = weights_[name];
    const size_t offset = dataOffset + gguf_get_tensor_offset(gguf, i);
    const size_t nbytes = ggml_nbytes(dst);
    readBuf.resize(nbytes);
    if (fseek(file, static_cast<long>(offset), SEEK_SET) != 0 ||
        fread(readBuf.data(), 1, nbytes, file) != nbytes) {
      lastError_ = std::string("failed to read tensor ") + name;
      ok = false;
      break;
    }
    ggml_backend_tensor_set(dst, readBuf.data(), 0, nbytes);
  }
  fclose(file);
  gguf_free(gguf);
  ggml_free(metaCtx);
  return ok;
}

bool
LamAudio2Expression::run(const std::vector<float>& pcm, uint32_t idIdx,
                         std::vector<float>& framesOut,
                         std::map<std::string, std::vector<float>>* taps) {
  const auto& hp = hparams_;
  const int64_t nSamples = static_cast<int64_t>(pcm.size());
  const int64_t frames = frameCount(nSamples);
  const int64_t tConv = convOutLen(nSamples);
  if (nSamples == 0 || frames < 2 || tConv < 2) {
    lastError_ = "input too short";
    return false;
  }
  if (idIdx >= hp.nIdentity) {
    lastError_ = "identity index out of range";
    return false;
  }

  // Generous node bound: the encoder dominates (~30 nodes/layer) plus the
  // grouped positional conv (~5 nodes/group).
  const size_t graphNodes = 2048;
  const size_t ctxSize = graphNodes * ggml_tensor_overhead() +
                         ggml_graph_overhead_custom(graphNodes, false) +
                         (1U << 16U);
  struct ggml_init_params gparams = {ctxSize, nullptr, /*no_alloc*/ true};
  struct ggml_context* ctx = ggml_init(gparams);
  struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, graphNodes, false);

  const auto tap = [&](struct ggml_tensor* t, const char* name) {
    ggml_set_name(t, name);
    if (taps != nullptr) {
      // Output-flag tapped tensors so the graph allocator does not reuse
      // their buffers before we read them back.
      ggml_set_output(t);
      ggml_build_forward_expand(graph, t);
    }
    return t;
  };

  // ---- inputs -----------------------------------------------------------
  struct ggml_tensor* pcmIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nSamples, 1);
  ggml_set_name(pcmIn, "input_pcm");
  ggml_set_input(pcmIn);

  struct ggml_tensor* idIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.nIdentity, 1);
  ggml_set_name(idIn, "input_id");
  ggml_set_input(idIn);

  // 50 Hz → 30 fps linear interpolation as a [tConv, frames] matrix
  // (align_corners=true), filled at set-input time below.
  struct ggml_tensor* interpW =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tConv, frames);
  ggml_set_name(interpW, "input_interp");
  ggml_set_input(interpW);

  // ---- feature extractor (time-major [t, c]) ----------------------------
  struct ggml_tensor* cur = pcmIn;
  for (uint32_t i = 0; i < hp.feKernels.size(); ++i) {
    cur = conv1d(ctx, weight("fe.conv" + std::to_string(i) + ".weight"), cur,
                 hp.feStrides[i], 0);
    if (i == 0) {
      // GroupNorm(512, 512) == per-channel norm over time; time is ne0 here.
      cur = ggml_norm(ctx, cur, hp.layerNormEps);
      cur = ggml_mul(ctx, cur,
                     ggml_reshape_2d(ctx, weight("fe.gn.weight"), 1,
                                     weight("fe.gn.weight")->ne[0]));
      cur = addBiasTimeMajor(ctx, cur, weight("fe.gn.bias"));
    }
    cur = ggml_gelu_erf(ctx, cur);
  }
  // [tConv, 512] time-major (byte-compatible with the PyTorch (512, tConv) dump)
  cur = tap(cur, "fe_out");

  // ---- 50→30 fps interpolation ------------------------------------------
  cur = ggml_mul_mat(ctx, interpW, ggml_cont(ctx, cur)); // [frames, 512]
  tap(cur, "interp_out");

  // ---- wav2vec2 feature projection (feature-major [c, t]) ---------------
  cur = ggml_cont(ctx, ggml_transpose(ctx, cur)); // [512, frames]
  cur = layerNorm(ctx, cur, weight("fp.ln.weight"), weight("fp.ln.bias"),
                  hp.layerNormEps);
  cur = ggml_mul_mat(ctx, weight("fp.proj.weight"), cur); // [768, frames]
  cur = ggml_add(ctx, cur, weight("fp.proj.bias"));
  tap(cur, "fp_out");

  // ---- positional conv embedding (grouped conv, time-major) -------------
  {
    struct ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, cur)); // [t, 768]
    struct ggml_tensor* kern = weight("enc.pos_conv.weight"); // [128, 48, 768]
    const int64_t chPerGroup = hp.encHidden / hp.posConvGroups; // 48
    struct ggml_tensor* pos = nullptr;
    for (uint32_t g = 0; g < hp.posConvGroups; ++g) {
      struct ggml_tensor* xg = ggml_cont(
          ctx, ggml_view_2d(ctx, xt, xt->ne[0], chPerGroup, xt->nb[1],
                            g * chPerGroup * xt->nb[1]));
      struct ggml_tensor* kg = ggml_cont(
          ctx, ggml_view_3d(ctx, kern, kern->ne[0], kern->ne[1], chPerGroup,
                            kern->nb[1], kern->nb[2],
                            g * chPerGroup * kern->nb[2]));
      struct ggml_tensor* cg =
          conv1d(ctx, kg, xg, 1, static_cast<int>(hp.posConvKernel / 2));
      pos = pos == nullptr ? cg : ggml_concat(ctx, pos, cg, 1);
    }
    // Even kernel + pad k/2 yields t+1 outputs; drop the trailing one
    // (Wav2Vec2SamePadLayer), then bias + GELU.
    pos = ggml_view_2d(ctx, pos, frames, hp.encHidden, pos->nb[1], 0);
    pos = addBiasTimeMajor(ctx, pos, weight("enc.pos_conv.bias"));
    pos = ggml_gelu_erf(ctx, pos);
    pos = ggml_cont(ctx, ggml_transpose(ctx, pos)); // [768, t]
    tap(pos, "pos_conv_out");
    cur = ggml_add(ctx, cur, pos);
  }
  cur = layerNorm(ctx, cur, weight("enc.ln.weight"), weight("enc.ln.bias"),
                  hp.layerNormEps);
  tap(cur, "enc_pre_ln");

  // ---- transformer encoder (post-norm) -----------------------------------
  const int64_t headDim = hp.encHidden / hp.encHeads;
  const float attnScale = 1.0F / std::sqrt(static_cast<float>(headDim));
  for (uint32_t il = 0; il < hp.encLayers; ++il) {
    const std::string blk = "enc.blk" + std::to_string(il) + ".";
    struct ggml_tensor* residual = cur;

    struct ggml_tensor* q =
        ggml_add(ctx, ggml_mul_mat(ctx, weight(blk + "attn_q.weight"), cur),
                 weight(blk + "attn_q.bias"));
    q = ggml_scale(ctx, q, attnScale);
    struct ggml_tensor* k =
        ggml_add(ctx, ggml_mul_mat(ctx, weight(blk + "attn_k.weight"), cur),
                 weight(blk + "attn_k.bias"));
    struct ggml_tensor* v =
        ggml_add(ctx, ggml_mul_mat(ctx, weight(blk + "attn_v.weight"), cur),
                 weight(blk + "attn_v.bias"));

    q = ggml_cont(ctx, ggml_permute(
        ctx, ggml_reshape_3d(ctx, q, headDim, hp.encHeads, frames), 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(
        ctx, ggml_reshape_3d(ctx, k, headDim, hp.encHeads, frames), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(
        ctx, ggml_reshape_3d(ctx, v, headDim, hp.encHeads, frames), 1, 2, 0, 3));

    struct ggml_tensor* kq = ggml_soft_max(ctx, ggml_mul_mat(ctx, k, q));
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq); // [headDim, t, heads]
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx, kqv, hp.encHidden, frames);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, weight(blk + "attn_o.weight"), cur),
                   weight(blk + "attn_o.bias"));

    cur = ggml_add(ctx, cur, residual);
    cur = layerNorm(ctx, cur, weight(blk + "ln1.weight"),
                    weight(blk + "ln1.bias"), hp.layerNormEps);

    struct ggml_tensor* ffn =
        ggml_add(ctx, ggml_mul_mat(ctx, weight(blk + "ffn_up.weight"), cur),
                 weight(blk + "ffn_up.bias"));
    ffn = ggml_gelu_erf(ctx, ffn);
    ffn = ggml_add(ctx, ggml_mul_mat(ctx, weight(blk + "ffn_down.weight"), ffn),
                   weight(blk + "ffn_down.bias"));
    cur = ggml_add(ctx, cur, ffn);
    cur = layerNorm(ctx, cur, weight(blk + "ln2.weight"),
                    weight(blk + "ln2.bias"), hp.layerNormEps);
    tap(cur, ("enc_layer_" + std::to_string(il)).c_str());
  }

  // ---- LAM head -----------------------------------------------------------
  cur = ggml_add(ctx, ggml_mul_mat(ctx, weight("head.proj.weight"), cur),
                 weight("head.proj.bias")); // [512, t]
  tap(cur, "lam_proj");

  // identity one-hot → 64-dim feature, broadcast over time
  struct ggml_tensor* idFeat = ggml_mul_mat(
      ctx,
      ggml_reshape_2d(ctx, weight("head.id_mlp.weight"), hp.nIdentity,
                      hp.identityFeatDim),
      idIn); // [64, 1]
  idFeat = ggml_add(ctx, idFeat,
                    ggml_reshape_2d(ctx, weight("head.id_mlp.bias"),
                                    hp.identityFeatDim, 1));
  idFeat = ggml_repeat(
      ctx, idFeat,
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.identityFeatDim, frames));

  // features live on ne0 in this layout → concat on dim 0: [576, t]
  cur = ggml_concat(ctx, cur, idFeat, 0);

  // ConvNormRelu (k=3, s=1, p=1) with LayerNorm over channels.
  // residualMode: 0 none, 1 identity, 2 conv (head.first0.res).
  const auto convNormRelu = [&](struct ggml_tensor* x, const std::string& name,
                                int residualMode) {
    struct ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [t, c]
    struct ggml_tensor* out =
        conv1d(ctx, weight(name + ".conv.weight"), xt, 1, 1);
    out = addBiasTimeMajor(ctx, out, weight(name + ".conv.bias"));
    out = ggml_cont(ctx, ggml_transpose(ctx, out)); // [c, t]
    out = layerNorm(ctx, out, weight(name + ".ln.weight"),
                    weight(name + ".ln.bias"), hp.layerNormEps);
    if (residualMode == 1) {
      out = ggml_add(ctx, out, x);
    } else if (residualMode == 2) {
      struct ggml_tensor* res =
          conv1d(ctx, weight("head.first0.res.weight"), xt, 1, 1);
      res = addBiasTimeMajor(ctx, res, weight("head.first0.res.bias"));
      out = ggml_add(ctx, out, ggml_cont(ctx, ggml_transpose(ctx, res)));
    }
    return ggml_relu(ctx, out);
  };

  cur = tap(convNormRelu(cur, "head.first0", 2), "ident_cnr_0");
  cur = tap(convNormRelu(cur, "head.first1", 1), "ident_cnr_1");
  cur = tap(convNormRelu(cur, "head.first2", 1), "ident_cnr_2");
  cur = tap(convNormRelu(cur, "head.dec0", 0), "dec_cnr_0");
  cur = tap(convNormRelu(cur, "head.dec1", 0), "dec_cnr_1");
  cur = tap(convNormRelu(cur, "head.dec2", 0), "dec_cnr_2");

  cur = ggml_add(ctx, ggml_mul_mat(ctx, weight("head.out.weight"), cur),
                 weight("head.out.bias")); // [52, t]
  tap(cur, "expr_logits");
  struct ggml_tensor* expr = ggml_sigmoid(ctx, cur);
  ggml_set_name(expr, "expr");
  ggml_set_output(expr);
  ggml_build_forward_expand(graph, expr);

  // ---- allocate, set inputs, compute -------------------------------------
  ggml_gallocr_t alloc =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (!ggml_gallocr_alloc_graph(alloc, graph)) {
    lastError_ = "failed to allocate compute graph";
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return false;
  }

  ggml_backend_tensor_set(pcmIn, pcm.data(), 0, pcm.size() * sizeof(float));

  std::vector<float> idOnehot(hp.nIdentity, 0.0F);
  idOnehot[idIdx] = 1.0F;
  ggml_backend_tensor_set(idIn, idOnehot.data(), 0,
                          idOnehot.size() * sizeof(float));

  // align_corners=true linear interpolation weights
  std::vector<float> interp(static_cast<size_t>(tConv) * frames, 0.0F);
  const double scale =
      static_cast<double>(tConv - 1) / static_cast<double>(frames - 1);
  for (int64_t j = 0; j < frames; ++j) {
    const double pos = static_cast<double>(j) * scale;
    const auto i0 = static_cast<int64_t>(pos);
    const int64_t i1 = i0 + 1 < tConv ? i0 + 1 : tConv - 1;
    const auto w = static_cast<float>(pos - static_cast<double>(i0));
    interp[j * tConv + i0] += 1.0F - w;
    interp[j * tConv + i1] += w;
  }
  ggml_backend_tensor_set(interpW, interp.data(), 0,
                          interp.size() * sizeof(float));

  const ggml_status status = ggml_backend_graph_compute(backend_, graph);
  if (status != GGML_STATUS_SUCCESS) {
    lastError_ = "graph compute failed";
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return false;
  }

  const auto readTensor = [&](struct ggml_tensor* t) {
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
  };

  // expr is [52, frames] with ne0 contiguous → already frame-major.
  framesOut = readTensor(expr);

  if (taps != nullptr) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
      struct ggml_tensor* node = ggml_graph_node(graph, i);
      const std::string name = ggml_get_name(node);
      if (name.rfind("fe_out", 0) == 0 || name.rfind("interp_out", 0) == 0 ||
          name.rfind("fp_out", 0) == 0 || name.rfind("pos_conv_out", 0) == 0 ||
          name.rfind("enc_pre_ln", 0) == 0 ||
          name.rfind("enc_layer_", 0) == 0 || name.rfind("lam_proj", 0) == 0 ||
          name.rfind("ident_cnr_", 0) == 0 || name.rfind("dec_cnr_", 0) == 0 ||
          name.rfind("expr_logits", 0) == 0 || name == "expr") {
        (*taps)[name] = readTensor(node);
      }
    }
  }

  ggml_gallocr_free(alloc);
  ggml_free(ctx);
  return true;
}

