/**
 * @file ltx2_api.cpp
 * @brief LTX-Video 2.3 C API implementation.
 *
 * Thin facade over the public new_sd_ctx() / generate_video() entry points
 * that exposes a small, LTX-2-specific C surface. Builds the right sampling
 * defaults (Euler sampler, LTX2 flow scheduler, flow_shift 2.37) and routes
 * the Gemma 3 text-encoder path through the sd_ctx_params_t::llm_path slot.
 *
 * T2V leaves init_image.data null; I2V passes the caller's start frame
 * through, and the underlying generate_video() VAE-encodes it, places it at
 * latent temporal index 0, and applies the per-frame denoise mask.
 */

#include "ltx2.h"

#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

/* Internal implementation header – needed for StableDiffusionGGML internals
 * used by the descriptor check.  Included in the same translation unit as
 * stable-diffusion.cpp so the types are visible. */
#include "stable-diffusion.h"

/* -------------------------------------------------------------------------
 * Logging helper (matches the macro pattern in stable-diffusion.cpp)
 * ---------------------------------------------------------------------- */
#ifndef LOG_INFO
#  include <cstdio>
#  define LOG_INFO(fmt, ...)  fprintf(stderr, "[INFO ] ltx2: " fmt "\n", ##__VA_ARGS__)
#  define LOG_WARN(fmt, ...)  fprintf(stderr, "[WARN ] ltx2: " fmt "\n", ##__VA_ARGS__)
#  define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] ltx2: " fmt "\n", ##__VA_ARGS__)
#endif

/* LTX-2.3 sampling defaults: Euler sampler with the LTX2 flow-matching
 * scheduler and the empirical flow_shift used by the LTXAV model family. */
#define LTX2_FLOW_SHIFT 2.37f

/* -------------------------------------------------------------------------
 * Internal context struct
 * ---------------------------------------------------------------------- */
struct ltx2_ctx_t {
    sd_ctx_t* sd_ctx;

    /* Cached parameter copies for diagnostics / future re-init. */
    std::string diffusion_model_path;
    std::string vae_path;
    std::string gemma_path;
    int         n_threads;
    sd_type_t   wtype;

    /* Set to true the first time we emit the pre-sync scheduler warning so
     * we do not spam the log on every generate call. */
    bool scheduler_warning_emitted;
};

/* -------------------------------------------------------------------------
 * ltx2_new_ctx
 * ---------------------------------------------------------------------- */
ltx2_ctx_t* ltx2_new_ctx(const char*    diffusion_model_path,
                          const char*    vae_path,
                          const char*    gemma_path,
                          int            n_threads,
                          sd_type_t      wtype) {
    if (!diffusion_model_path || !vae_path || !gemma_path) {
        LOG_ERROR("ltx2_new_ctx: diffusion_model_path, vae_path, and gemma_path are all required");
        return nullptr;
    }

    /* Resolve thread count. */
    if (n_threads <= 0) {
        n_threads = sd_get_num_physical_cores();
        LOG_INFO("n_threads defaulting to %d physical cores", n_threads);
    }

    sd_ctx_params_t p;
    sd_ctx_params_init(&p);

    p.diffusion_model_path  = diffusion_model_path;
    p.vae_path              = vae_path;
    p.llm_path              = gemma_path;   /* Loaded with prefix "text_encoders.llm." */
    p.n_threads             = n_threads;
    p.wtype                 = wtype;
    p.vae_decode_only       = false;        /* I2V requires VAE encode as well. */
    p.diffusion_flash_attn  = true;         /* Strongly recommended for 22B DiT. */
    p.offload_params_to_cpu = false;        /* Caller can override by wrapping new_sd_ctx directly. */
    p.rng_type              = CUDA_RNG;     /* PhiloxRNG – matches upstream default. */

    sd_ctx_t* sd_ctx = new_sd_ctx(&p);
    if (!sd_ctx) {
        LOG_ERROR("ltx2_new_ctx: new_sd_ctx() failed – check model paths and memory");
        return nullptr;
    }

    ltx2_ctx_t* ctx = static_cast<ltx2_ctx_t*>(malloc(sizeof(ltx2_ctx_t)));
    if (!ctx) {
        LOG_ERROR("ltx2_new_ctx: out of memory allocating ltx2_ctx_t");
        free_sd_ctx(sd_ctx);
        return nullptr;
    }

    /* Placement-new to initialise std::string members properly. */
    new (ctx) ltx2_ctx_t();

    ctx->sd_ctx                      = sd_ctx;
    ctx->diffusion_model_path        = diffusion_model_path;
    ctx->vae_path                    = vae_path;
    ctx->gemma_path                  = gemma_path;
    ctx->n_threads                   = n_threads;
    ctx->wtype                       = wtype;
    ctx->scheduler_warning_emitted   = false;

    LOG_INFO("ltx2 context created (diffusion=%s)", diffusion_model_path);
    return ctx;
}

/* -------------------------------------------------------------------------
 * ltx2_free_ctx
 * ---------------------------------------------------------------------- */
void ltx2_free_ctx(ltx2_ctx_t* ctx) {
    if (!ctx) return;
    if (ctx->sd_ctx) {
        free_sd_ctx(ctx->sd_ctx);
        ctx->sd_ctx = nullptr;
    }
    /* Destruct std::string members before releasing raw memory. */
    ctx->~ltx2_ctx_t();
    free(ctx);
}

/* -------------------------------------------------------------------------
 * Internal helper: fill a sd_vid_gen_params_t with LTX-2.3 defaults.
 * ---------------------------------------------------------------------- */
static void fill_ltx2_vid_params(sd_vid_gen_params_t* vp,
                                  const char*          prompt,
                                  const char*          negative_prompt,
                                  int                  width,
                                  int                  height,
                                  int                  video_frames,
                                  int                  sample_steps,
                                  float                cfg_scale,
                                  int64_t              seed,
                                  bool                 warn_scheduler,
                                  bool*                warning_emitted) {
    sd_vid_gen_params_init(vp);

    vp->prompt          = prompt;
    vp->negative_prompt = (negative_prompt != nullptr) ? negative_prompt : "";
    vp->width           = width;
    vp->height          = height;
    vp->video_frames    = video_frames;
    vp->seed            = seed;
    vp->strength        = 1.0f;   /* Full conditioning strength – adjusted per-frame inside generate_video. */

    /* Sample parameters: Euler sampler, LTX2 scheduler (or discrete fallback),
     * flow_shift=2.37 per upstream LTX-2.3 documentation.               */
    vp->sample_params.sample_method  = EULER_SAMPLE_METHOD;
    vp->sample_params.scheduler      = LTX2_SCHEDULER;
    vp->sample_params.flow_shift     = LTX2_FLOW_SHIFT;
    vp->sample_params.sample_steps   = (sample_steps > 0) ? sample_steps : 30;
    vp->sample_params.guidance.txt_cfg = cfg_scale;

    /* Disable the high-noise two-stage path (not used for LTX-2.3). */
    vp->high_noise_sample_params.sample_steps = -1;

    (void)warn_scheduler;
    (void)warning_emitted;
}

/* -------------------------------------------------------------------------
 * ltx2_generate_t2v
 * ---------------------------------------------------------------------- */
sd_image_t* ltx2_generate_t2v(ltx2_ctx_t*  ctx,
                               const char*  prompt,
                               const char*  negative_prompt,
                               int          width,
                               int          height,
                               int          video_frames,
                               int          fps,
                               int          sample_steps,
                               float        cfg_scale,
                               int64_t      seed,
                               int*         out_num_frames) {
    if (!ctx || !ctx->sd_ctx) {
        LOG_ERROR("ltx2_generate_t2v: null context");
        return nullptr;
    }
    if (!prompt) {
        LOG_ERROR("ltx2_generate_t2v: prompt must not be NULL");
        return nullptr;
    }
    if (!out_num_frames) {
        LOG_ERROR("ltx2_generate_t2v: out_num_frames must not be NULL");
        return nullptr;
    }

    /* fps is informational in this fork; log it for the caller's benefit. */
    LOG_INFO("T2V %dx%d  frames=%d  fps=%d  steps=%d  cfg=%.2f  seed=%" PRId64,
             width, height, video_frames, fps, sample_steps, cfg_scale, seed);

    sd_vid_gen_params_t vp;
    fill_ltx2_vid_params(&vp,
                          prompt, negative_prompt,
                          width, height, video_frames,
                          sample_steps, cfg_scale, seed,
                          /*warn_scheduler=*/true,
                          &ctx->scheduler_warning_emitted);

    /* T2V: leave init_image zeroed (data == nullptr) so generate_video
     * creates a fully-noised latent and skips the I2V conditioning path. */
    vp.init_image = {};   /* .data = nullptr, .width/.height/.channel = 0 */

    sd_image_t* frames     = nullptr;
    sd_audio_t* audio      = nullptr;   /* video-only build; audio is dropped */
    bool        ok         = generate_video(ctx->sd_ctx, &vp, &frames, out_num_frames, &audio);
    if (!ok || !frames) {
        LOG_ERROR("ltx2_generate_t2v: generate_video() failed");
        return nullptr;
    }
    if (audio) {
        free(audio);   /* discard audio output; this fork targets video only */
        audio = nullptr;
    }

    LOG_INFO("T2V done – %d frames decoded", *out_num_frames);
    return frames;
}

/* -------------------------------------------------------------------------
 * ltx2_generate_i2v
 * ---------------------------------------------------------------------- */
sd_image_t* ltx2_generate_i2v(ltx2_ctx_t*  ctx,
                               sd_image_t   init_image,
                               const char*  prompt,
                               const char*  negative_prompt,
                               int          width,
                               int          height,
                               int          video_frames,
                               int          fps,
                               int          sample_steps,
                               float        cfg_scale,
                               int64_t      seed,
                               int*         out_num_frames) {
    if (!ctx || !ctx->sd_ctx) {
        LOG_ERROR("ltx2_generate_i2v: null context");
        return nullptr;
    }
    if (!prompt) {
        LOG_ERROR("ltx2_generate_i2v: prompt must not be NULL");
        return nullptr;
    }
    if (!out_num_frames) {
        LOG_ERROR("ltx2_generate_i2v: out_num_frames must not be NULL");
        return nullptr;
    }
    if (!init_image.data) {
        LOG_ERROR("ltx2_generate_i2v: init_image.data must not be NULL for I2V");
        return nullptr;
    }

    LOG_INFO("I2V %dx%d  frames=%d  fps=%d  steps=%d  cfg=%.2f  seed=%" PRId64,
             width, height, video_frames, fps, sample_steps, cfg_scale, seed);

    sd_vid_gen_params_t vp;
    fill_ltx2_vid_params(&vp,
                          prompt, negative_prompt,
                          width, height, video_frames,
                          sample_steps, cfg_scale, seed,
                          /*warn_scheduler=*/true,
                          &ctx->scheduler_warning_emitted);

    /* I2V: pass the caller's start frame.  generate_video() inspects
     * init_image.data != nullptr to enter the I2V conditioning branch,
     * VAE-encodes the image, places it at temporal position 0 of the
     * latent, and applies the per-frame denoise mask.               */
    vp.init_image  = init_image;
    vp.strength    = 1.0f;   /* Full I2V conditioning – first frame is fully fixed. */

    sd_image_t* frames     = nullptr;
    sd_audio_t* audio      = nullptr;   /* video-only build; audio is dropped */
    bool        ok         = generate_video(ctx->sd_ctx, &vp, &frames, out_num_frames, &audio);
    if (!ok || !frames) {
        LOG_ERROR("ltx2_generate_i2v: generate_video() failed");
        return nullptr;
    }
    if (audio) {
        free(audio);   /* discard audio output; this fork targets video only */
        audio = nullptr;
    }

    LOG_INFO("I2V done – %d frames decoded", *out_num_frames);
    return frames;
}
