// Tests for the CoreML sidecars (SD_COREML builds).
//
// Unit tests run the tiny models from tests/coreml/make_fixtures.py through CoreMLUNet, CoreMLUNetModel,
// CoreMLTAEDecoder and the FLUX.2 sidecar (flux_coreml_*). Pipeline tests run only when SD_TEST_SDXS_DIR points at the SDXS artifacts from
// script/coreml/export_artifacts.py (sdxs-f16.gguf, sdxs_unet_fp16.mlmodelc, sdxs_tae_decoder.mlmodelc).
//
//   python tests/coreml/make_fixtures.py build/coreml-fixtures
//   build/bin/test-coreml build/coreml-fixtures

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "ggml_tensor_utils.h"
#include "src/coreml_tae.h"
#include "src/coreml_unet.h"
#include "src/flux_coreml.h"
#include "stable-diffusion.h"

namespace {

int g_failures = 0;
std::string g_log;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                          \
        }                                                                          \
    } while (0)

void log_cb(sd_log_level_t level, const char* text, void*) {
    g_log += text;
    if (level >= SD_LOG_WARN) {
        fputs(text, stderr);
    }
}

bool log_has(const char* text) {
    return g_log.find(text) != std::string::npos;
}

constexpr int64_t C = 4, H = 8, W = 8, T = 5, D = 6;
constexpr int64_t A = 7, P = 3, K = 2;  // y values; text_embeds and time_ids values (see make_fixtures.py)

std::vector<float> ramp(size_t n, float scale, float offset) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = offset + scale * (float)((i * 37) % 101) / 101.0f;
    }
    return v;
}

// What the fixture UNet computes (see make_fixtures.py), for row-major [N, C, H, W] and [N, T, D] inputs.
std::vector<float> reference_unet(const std::vector<float>& x, const std::vector<float>& t, const std::vector<float>& ctx, int64_t n) {
    std::vector<float> out(x.size());
    for (int64_t b = 0; b < n; b++) {
        double c = 0;
        for (int64_t i = 0; i < T; i++) {
            for (int64_t j = 0; j < D; j++) {
                c += ctx[(b * T + i) * D + j] * (0.1 * (i + 1) + 0.01 * (j + 1));
            }
        }
        for (int64_t k = 0; k < C * H * W; k++) {
            out[b * C * H * W + k] = 0.5f * x[b * C * H * W + k] + (float)(c + 0.001 * t[b]);
        }
    }
    return out;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = a.size() == b.size() ? 0.0f : INFINITY;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        m = std::fmax(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

void test_unet_variants(const std::string& dir) {
    const int64_t x_shape[4]   = {1, C, H, W};
    const int64_t ctx_shape[3] = {1, T, D};
    auto x                     = ramp(C * H * W, 2.0f, -1.0f);
    auto ctx                   = ramp(T * D, 1.0f, -0.5f);
    std::vector<float> t       = {999.0f};
    auto expected              = reference_unet(x, t, ctx, 1);

    // fp32 inputs, fp16 inputs (converted by the runtime), and a [N, D, 1, T] context (transposed by the runtime).
    for (auto [name, tol] : {std::pair<const char*, float>{"unet_b1", 1e-4f},
                             {"unet_b1_fp16", 2e-2f},
                             {"unet_b1_channels_first", 1e-4f}}) {
        auto unet = CoreMLUNet::load(dir + "/" + name + ".mlpackage");
        CHECK(unet != nullptr);
        if (!unet) {
            continue;
        }
        std::vector<float> out(x.size(), NAN);
        CHECK(unet->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, nullptr, 0, out.data()));
        float err = max_abs_diff(out, expected);
        printf("  %s: max abs error %.2e\n", name, err);
        CHECK(err < tol);
        CHECK(unet->batch_size() == 1);
        CHECK(unet->prediction_type().empty());
    }
}

void test_unet_shape_rejection(const std::string& dir) {
    auto unet = CoreMLUNet::load(dir + "/unet_b1.mlpackage");
    CHECK(unet != nullptr);
    if (!unet) {
        return;
    }
    std::vector<float> x(C * 16 * 16), ctx(T * D), out(C * 16 * 16), t = {1.0f};
    const int64_t big_x[4]     = {1, C, 16, 16};
    const int64_t ctx_shape[3] = {1, T, D};
    CHECK(!unet->predict(x.data(), big_x, t.data(), ctx.data(), ctx_shape, nullptr, 0, out.data()));
    const int64_t x_shape[4]  = {1, C, H, W};
    const int64_t long_ctx[3] = {1, T + 1, D};
    std::vector<float> long_data((T + 1) * D);
    CHECK(!unet->predict(x.data(), x_shape, t.data(), long_data.data(), long_ctx, nullptr, 0, out.data()));
}

void test_prediction_metadata(const std::string& dir) {
    auto unet = CoreMLUNet::load(dir + "/unet_b2_vpred.mlpackage");
    CHECK(unet != nullptr);
    if (unet) {
        CHECK(unet->prediction_type() == "v_prediction");
        CHECK(unet->batch_size() == 2);
    }
}

// The batch-2 CFG call (cond and uncond stacked) must give the same outputs as separate calls, whether the
// model was exported for batch 2 or batch 1; a batch-2 model also takes a single image.
void test_cfg_batch(const std::string& dir) {
    auto batched = CoreMLUNet::load(dir + "/unet_b2_vpred.mlpackage");
    auto single  = CoreMLUNet::load(dir + "/unet_b1.mlpackage");
    CHECK(batched && single);
    if (!batched || !single) {
        return;
    }
    const int64_t x1_shape[4]   = {1, C, H, W};
    const int64_t x2_shape[4]   = {2, C, H, W};
    const int64_t ctx1_shape[3] = {1, T, D};
    const int64_t ctx2_shape[3] = {2, T, D};
    auto x                      = ramp(C * H * W, 2.0f, -1.0f);
    auto cond                   = ramp(T * D, 1.0f, -0.5f);
    auto uncond                 = ramp(T * D, -0.3f, 0.2f);
    float t                     = 500.0f;
    std::vector<float> xs(x), ctx(cond);
    xs.insert(xs.end(), x.begin(), x.end());
    ctx.insert(ctx.end(), uncond.begin(), uncond.end());
    auto expected = reference_unet(xs, {t, t}, ctx, 2);

    for (auto* unet : {batched.get(), single.get()}) {
        std::vector<float> out(xs.size(), NAN);
        CHECK(unet->run(xs.data(), x2_shape, &t, 1, ctx.data(), ctx2_shape, nullptr, 0, 0, out.data()));
        float err = max_abs_diff(out, expected);
        printf("  batch-%lld model, cond + uncond in one call: max abs error %.2e\n", (long long)unet->batch_size(), err);
        CHECK(err < 1e-4f);
    }

    std::vector<float> out(x.size(), NAN);
    CHECK(batched->run(x.data(), x1_shape, &t, 1, cond.data(), ctx1_shape, nullptr, 0, 0, out.data()));
    CHECK(max_abs_diff(out, reference_unet(x, {t}, cond, 1)) < 1e-4f);

    // Mismatched batches are rejected.
    CHECK(!batched->run(xs.data(), x2_shape, &t, 1, cond.data(), ctx1_shape, nullptr, 0, 0, out.data()));
    const float ts3[3] = {t, t, t};
    CHECK(!batched->run(xs.data(), x2_shape, ts3, 3, ctx.data(), ctx2_shape, nullptr, 0, 0, out.data()));
}

// What the SDXL-style fixtures add to reference_unet for one item: sum_a v[a] * 0.02 * (a + 1) over y or
// text_embeds, plus sum_k ids[k] * 0.001 * (k + 1) over time_ids.
float vector_term(const float* v, int64_t n, const std::vector<float>& ids = {}) {
    double s = 0;
    for (int64_t a = 0; a < n; a++) {
        s += v[a] * 0.02 * (a + 1);
    }
    for (size_t k = 0; k < ids.size(); k++) {
        s += ids[k] * 0.001 * (k + 1);
    }
    return (float)s;
}

void test_unet_vector_conditioning(const std::string& dir) {
    const int64_t x_shape[4]   = {1, C, H, W};
    const int64_t ctx_shape[3] = {1, T, D};
    auto x                     = ramp(C * H * W, 2.0f, -1.0f);
    auto ctx                   = ramp(T * D, 1.0f, -0.5f);
    std::vector<float> t       = {999.0f};

    // A model with a y input takes y as is.
    auto unet = CoreMLUNet::load(dir + "/unet_y_b1.mlpackage");
    CHECK(unet != nullptr);
    if (unet) {
        CHECK(unet->y_dim() == A);
        auto y        = ramp(A, 1.0f, -0.4f);
        auto expected = reference_unet(x, t, ctx, 1);
        for (float& v : expected) {
            v += vector_term(y.data(), A);
        }
        std::vector<float> out(x.size(), NAN);
        CHECK(unet->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, y.data(), A, out.data()));
        float err = max_abs_diff(out, expected);
        printf("  y input: max abs error %.2e\n", err);
        CHECK(err < 1e-4f);
        CHECK(!unet->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, nullptr, 0, out.data()));
        CHECK(!unet->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, y.data(), A - 1, out.data()));
    }

    // Apple's SDXL layout: y is split into text_embeds and the raw time_ids, which the runtime recovers from the
    // embeddings sd.cpp's SDXL conditioner writes (sizes 768x1024 and crop 0, 0 here).
    auto apple = CoreMLUNet::load(dir + "/unet_text_time_b2.mlpackage");
    CHECK(apple != nullptr);
    if (apple) {
        const int64_t y_dim = P + 256 * K;
        CHECK(apple->y_dim() == y_dim);
        CHECK(apple->batch_size() == 2);
        const std::vector<float> ids = {768.0f, 1024.0f};
        auto y                       = ramp(P, 1.0f, -0.2f);
        auto embedded                = timestep_embedding(ids, 256);
        y.insert(y.end(), embedded.begin(), embedded.end());

        // A single item with one y row: the runtime duplicates both for the batch-2 model.
        auto expected = reference_unet(x, t, ctx, 1);
        for (float& v : expected) {
            v += vector_term(y.data(), P, ids);
        }
        std::vector<float> out(x.size(), NAN);
        CHECK(apple->run(x.data(), x_shape, t.data(), 1, ctx.data(), ctx_shape, y.data(), 1, y_dim, out.data()));
        float err = max_abs_diff(out, expected);
        printf("  text_embeds + time_ids (fp16, channels-first context): max abs error %.2e\n", err);
        CHECK(err < 2e-2f);

        // cond + uncond with different y rows.
        const int64_t x2_shape[4]   = {2, C, H, W};
        const int64_t ctx2_shape[3] = {2, T, D};
        std::vector<float> xs(x), ctxs(ctx), ys(y);
        xs.insert(xs.end(), x.begin(), x.end());
        ctxs.insert(ctxs.end(), ctx.begin(), ctx.end());
        ys.insert(ys.end(), y.begin(), y.end());
        std::fill(ys.begin() + y_dim, ys.begin() + y_dim + P, 0.0f);
        auto expected2 = reference_unet(xs, {t[0], t[0]}, ctxs, 2);
        for (int64_t b = 0; b < 2; b++) {
            for (int64_t i = 0; i < C * H * W; i++) {
                expected2[b * C * H * W + i] += vector_term(ys.data() + b * y_dim, P, ids);
            }
        }
        std::vector<float> out2(xs.size(), NAN);
        CHECK(apple->run(xs.data(), x2_shape, t.data(), 1, ctxs.data(), ctx2_shape, ys.data(), 2, y_dim, out2.data()));
        CHECK(max_abs_diff(out2, expected2) < 2e-2f);
    }

    // A model without vector conditioning rejects y.
    auto plain = CoreMLUNet::load(dir + "/unet_b1.mlpackage");
    if (plain) {
        CHECK(plain->y_dim() == 0);
        auto y = ramp(A, 1.0f, 0.0f);
        std::vector<float> out(x.size());
        CHECK(!plain->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, y.data(), A, out.data()));
    }
}

// A folder of chunks runs them in order; the second chunk takes the first's output and encoder_hidden_states.
void test_unet_chunks(const std::string& dir) {
    auto unet = CoreMLUNet::load(dir + "/unet_chunks");
    CHECK(unet != nullptr);
    if (unet) {
        const int64_t x_shape[4]   = {1, C, H, W};
        const int64_t ctx_shape[3] = {1, T, D};
        auto x                     = ramp(C * H * W, 2.0f, -1.0f);
        auto ctx                   = ramp(T * D, 1.0f, -0.5f);
        std::vector<float> t       = {250.0f};
        std::vector<float> out(x.size(), NAN);
        CHECK(unet->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, nullptr, 0, out.data()));
        float err = max_abs_diff(out, reference_unet(x, t, ctx, 1));
        printf("  2 chunks: max abs error %.2e\n", err);
        CHECK(err < 1e-4f);
    }
    // A folder that is neither a model nor holds chunks.
    CHECK(CoreMLUNet::load(dir) == nullptr);
}

void test_tae(const std::string& dir) {
    auto tae = CoreMLTAEDecoder::load(dir + "/tae.mlpackage");
    CHECK(tae != nullptr);
    if (tae) {
        const int64_t latent_shape[4] = {1, C, H, W};
        const int64_t image_shape[4]  = {1, 3, 8 * H, 8 * W};
        auto latent                   = ramp(C * H * W, 1.0f, 0.0f);
        std::vector<float> image(3 * 64 * H * W, NAN);
        CHECK(tae->scale() == 8);
        CHECK(tae->decode(latent.data(), latent_shape, image.data(), image_shape));
        bool exact = true;
        for (int64_t c = 0; c < 3; c++) {
            for (int64_t y = 0; y < 8 * H; y++) {
                for (int64_t x = 0; x < 8 * W; x++) {
                    exact &= image[(c * 8 * H + y) * 8 * W + x] == latent[(c * H + y / 8) * W + x / 8];
                }
            }
        }
        CHECK(exact);

        // A destination of a different size is rejected before anything is written.
        const int64_t small_image[4] = {1, 3, 4 * H, 4 * W};
        std::vector<float> small(3 * 16 * H * W, -7.0f);
        CHECK(!tae->decode(latent.data(), latent_shape, small.data(), small_image));
        CHECK(small[0] == -7.0f);
        const int64_t other_latent[4] = {1, C, 2 * H, 2 * W};
        std::vector<float> big_latent(C * 4 * H * W);
        CHECK(!tae->decode(big_latent.data(), other_latent, image.data(), image_shape));
    }

    // TAEF2 decoders produce 16 image pixels per latent pixel.
    auto tae16 = CoreMLTAEDecoder::load(dir + "/tae_x16.mlpackage");
    CHECK(tae16 != nullptr);
    if (tae16) {
        CHECK(tae16->scale() == 16);
        const int64_t latent_shape[4] = {1, C, H, W};
        const int64_t image_shape[4]  = {1, 3, 16 * H, 16 * W};
        auto latent                   = ramp(C * H * W, 1.0f, 0.0f);
        std::vector<float> image(3 * 256 * H * W, NAN);
        CHECK(tae16->decode(latent.data(), latent_shape, image.data(), image_shape));
        bool exact = true;
        for (int64_t c = 0; c < 3; c++) {
            for (int64_t y = 0; y < 16 * H; y++) {
                for (int64_t x = 0; x < 16 * W; x++) {
                    exact &= image[(c * 16 * H + y) * 16 * W + x] == latent[(c * H + y / 16) * W + x / 16];
                }
            }
        }
        CHECK(exact);
        const int64_t x8_image[4] = {1, 3, 8 * H, 8 * W};
        CHECK(!tae16->decode(latent.data(), latent_shape, image.data(), x8_image));
    }

    // A model whose output is not [N, 3, 8h, 8w] or [N, 3, 16h, 16w] is rejected at load.
    CHECK(CoreMLTAEDecoder::load(dir + "/tae_wrong_scale.mlpackage") == nullptr);
}

// The FLUX.2 sidecar as one model and as two chunks passing an fp16 tensor: both compute
// 0.5 * latent + sum(context * W) + 0.001 * timesteps (see make_fixtures.py).
void test_flux(const std::string& dir) {
    auto latent  = ramp(C * H * W, 1.0f, -0.5f);
    auto context = ramp(T * D, 0.5f, 0.0f);
    const std::vector<float> timestep = {0.75f};
    float bias                         = 0.001f * timestep[0];
    for (int64_t t = 0; t < T; t++) {
        for (int64_t d = 0; d < D; d++) {
            bias += context[t * D + d] * (0.1f * (t + 1) + 0.01f * (d + 1));
        }
    }
    // ggml order (ne): latent [W, H, C, 1], timesteps [1, 1, 1, 1], context [D, T, 1, 1].
    const FluxCoreMLTensor x{latent.data(), {W, H, C, 1}};
    const FluxCoreMLTensor ts{timestep.data(), {1, 1, 1, 1}};
    const FluxCoreMLTensor ctx{context.data(), {D, T, 1, 1}};
    for (const char* name : {"flux.mlpackage", "flux_chunks"}) {
        std::string error;
        FluxCoreMLModel* model = flux_coreml_open((dir + "/" + name).c_str(), &error);
        CHECK(model != nullptr);
        if (!model) {
            fprintf(stderr, "  %s: %s\n", name, error.c_str());
            continue;
        }
        std::vector<float> out(latent.size(), NAN);
        const bool ok = flux_coreml_predict(model, x, ts, ctx, nullptr, nullptr, out.data(), &error);
        CHECK(ok);
        if (!ok) {
            fprintf(stderr, "  %s: %s\n", name, error.c_str());
        }
        float err = 0.0f;
        for (size_t i = 0; i < out.size(); i++) {
            const float e = std::fabs(out[i] - (0.5f * latent[i] + bias));
            err           = std::isnan(e) ? INFINITY : std::max(err, e);
        }
        printf("  %s: max abs error %.2e\n", name, err);
        CHECK(err < 2e-3f);  // the chunks pass the hidden state in fp16
        flux_coreml_close(model);
    }
    // A folder that holds no chunks.
    std::string error;
    CHECK(flux_coreml_open((dir + "/unet_chunks").c_str(), &error) == nullptr);
}

// ---- pipeline tests with the real SDXS artifacts ----

// A one-tensor LoRA for an SD 1.x UNet layer (SDXS keeps the SD 1.x names), in safetensors format.
std::string write_unet_lora(const std::string& path) {
    const std::string name = "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight";
    const size_t n         = 320 * 4;
    std::string header = "{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[320,4,1,1],\"data_offsets\":[0," +
                         std::to_string(n * 4) + "]}}";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream f(path, std::ios::binary);
    uint64_t len = header.size();
    f.write((const char*)&len, 8);
    f.write(header.data(), header.size());
    std::vector<float> data(n, 0.01f);
    f.write((const char*)data.data(), n * 4);
    return path;
}

sd_ctx_params_t sdxs_params(const std::string& dir) {
    static std::string model, unet, tae;
    model = dir + "/sdxs-f16.gguf";
    unet  = dir + "/sdxs_unet_fp16.mlmodelc";
    tae   = dir + "/sdxs_tae_decoder.mlmodelc";
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.model_path              = model.c_str();
    p.coreml_unet_path        = unet.c_str();
    p.coreml_tae_decoder_path = tae.c_str();
    p.n_threads               = 4;
    return p;
}

sd_image_t* generate(sd_ctx_t* sd, const sd_lora_t* loras, uint32_t n_loras, bool tiled, bool circular = false) {
    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt                         = "a red apple on a table";
    g.width                          = 512;
    g.height                         = 512;
    g.seed                           = 42;
    g.sample_params.sample_steps     = 1;
    g.sample_params.guidance.txt_cfg = 1.0f;
    g.loras                          = loras;
    g.lora_count                     = n_loras;
    g.vae_tiling_params.enabled      = tiled;
    g.circular_x                     = circular;
    sd_image_t* images = nullptr;
    int n_images       = 0;
    if (!generate_image(sd, &g, &images, &n_images) || n_images < 1) {
        free_sd_images(images, n_images);
        return nullptr;
    }
    return images;
}

void free_image(sd_image_t* img) {
    free_sd_images(img, img ? 1 : 0);
}

void test_pipeline(const std::string& dir) {
    sd_ctx_params_t p = sdxs_params(dir);
    sd_ctx_t* sd      = new_sd_ctx(&p);
    CHECK(sd != nullptr);
    if (sd) {
        sd_image_t* coreml = generate(sd, nullptr, 0, false);
        sd_image_t* tiled  = generate(sd, nullptr, 0, true);
        CHECK(coreml && tiled);
        if (coreml && tiled) {
            size_t n   = (size_t)coreml->width * coreml->height * coreml->channel;
            double mse = 0;
            for (size_t i = 0; i < n; i++) {
                double d = (double)coreml->data[i] - tiled->data[i];
                mse += d * d;
            }
            double psnr = 10 * std::log10(255.0 * 255.0 / std::fmax(mse / n, 1e-12));
            printf("  CoreML vs tiled ggml tiny decoder: PSNR %.1f dB\n", psnr);
            CHECK(psnr > 25.0);
        }
        free_image(coreml);
        free_image(tiled);

        // Circular padding cannot be applied to the fixed CoreML UNet graph.
        g_log.clear();
        sd_image_t* circular = generate(sd, nullptr, 0, false, true);
        CHECK(circular == nullptr);
        CHECK(log_has("circular padding"));
        free_image(circular);
        free_sd_ctx(sd);
    }

    // A LoRA that changes the UNet is rejected in both apply modes.
    std::string lora_path = write_unet_lora(std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp") + "/sd_coreml_test_lora.safetensors");
    for (lora_apply_mode_t mode : {LORA_APPLY_IMMEDIATELY, LORA_APPLY_AT_RUNTIME}) {
        sd_ctx_params_t p = sdxs_params(dir);
        p.lora_apply_mode = mode;
        sd_ctx_t* sd      = new_sd_ctx(&p);
        CHECK(sd != nullptr);
        if (!sd) {
            continue;
        }
        sd_lora_t lora = {false, 1.0f, lora_path.c_str()};
        g_log.clear();
        sd_image_t* img = generate(sd, &lora, 1, false);
        CHECK(img == nullptr);
        CHECK(log_has("not supported with the CoreML UNet"));
        free_image(img);
        free_sd_ctx(sd);
    }
    remove(lora_path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE_DIR (from tests/coreml/make_fixtures.py)\n", argv[0]);
        return 2;
    }
    sd_set_log_callback(log_cb, nullptr);
    const std::string dir = argv[1];
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"unet variants", [&] { test_unet_variants(dir); }},
        {"unet shape rejection", [&] { test_unet_shape_rejection(dir); }},
        {"prediction metadata", [&] { test_prediction_metadata(dir); }},
        {"batched cfg", [&] { test_cfg_batch(dir); }},
        {"vector conditioning (SDXL)", [&] { test_unet_vector_conditioning(dir); }},
        {"chunks", [&] { test_unet_chunks(dir); }},
        {"tae", [&] { test_tae(dir); }},
        {"flux", [&] { test_flux(dir); }},
    };
    for (auto& [name, fn] : tests) {
        printf("%s\n", name);
        fn();
    }
    if (const char* sdxs = getenv("SD_TEST_SDXS_DIR")) {
        printf("pipeline (SD_TEST_SDXS_DIR=%s)\n", sdxs);
        test_pipeline(sdxs);
    } else {
        printf("pipeline: skipped (set SD_TEST_SDXS_DIR to the SDXS artifacts to run it)\n");
    }
    printf(g_failures ? "FAILED: %d checks\n" : "all tests passed\n", g_failures);
    return g_failures ? 1 : 0;
}
