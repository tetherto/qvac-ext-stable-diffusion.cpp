/**
 * ltx2.h — Public C API for LTX-2 video generation
 *
 * Thin façade over stable-diffusion.h scoped to LTX-2 T2V and I2V inference.
 * Consumers that only need LTX-2 can include this header instead of the full
 * stable-diffusion.h.  All types and functions are compatible with the broader
 * sd_ctx / generate_video API.
 *
 * Quick-start (text-to-video):
 *
 *   sd_ctx_params_t p;
 *   sd_ctx_params_init(&p);
 *   ltx2_ctx_params_set_defaults(&p);
 *   p.diffusion_model_path          = "ltx-2.3-22b-dev_Q8_0.gguf";
 *   p.vae_path                      = "ltx-2.3-22b-dev_video_vae.safetensors";
 *   p.audio_vae_path                = "ltx-2.3-22b-dev_audio_vae.safetensors";
 *   p.llm_path                      = "gemma-3-12b-it_Q4_0.gguf";
 *   p.embeddings_connector_path     = "ltx-2.3-22b-dev_embeddings_connectors.safetensors";
 *
 *   sd_ctx_t* ctx = new_sd_ctx(&p);
 *
 *   sd_image_t* frames = NULL;
 *   int n_frames = 0;
 *   sd_audio_t* audio = NULL;
 *   ltx2_generate_t2v(ctx, "a serene mountain lake at sunrise", NULL,
 *                     1280, 720, 33, 24, -1,
 *                     &frames, &n_frames, &audio);
 *
 *   // frames[0..n_frames-1] contain raw RGB pixel data
 *   free_sd_ctx(ctx);
 */

#ifndef __LTX2_H__
#define __LTX2_H__

#include "stable-diffusion.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * LTX-2 context helpers
 * ---------------------------------------------------------------------- */

/**
 * Apply LTX-2.3 recommended defaults to an already-initialised sd_ctx_params_t.
 * Call sd_ctx_params_init() first, then this function to overlay LTX-2 settings.
 */
static inline void ltx2_ctx_params_set_defaults(sd_ctx_params_t* p) {
    p->diffusion_flash_attn = true;
}

/**
 * Apply LTX-2.3 recommended sample defaults to an sd_vid_gen_params_t.
 * Call sd_vid_gen_params_init() first, then this function before generation.
 */
static inline void ltx2_vid_params_set_defaults(sd_vid_gen_params_t* vp) {
    vp->sample_params.scheduler     = LTX2_SCHEDULER;
    vp->sample_params.sample_method = EULER_SAMPLE_METHOD;
    vp->sample_params.guidance.txt_cfg = 3.5f;
}

/**
 * Convenience constructor: create an LTX-2 inference context from file paths.
 *
 * @param diffusion_model_path  LTX-2.3 diffusion model (GGUF or safetensors).
 * @param vae_path              Video VAE weights (safetensors).
 * @param audio_vae_path        Audio VAE weights (safetensors), or NULL.
 * @param llm_path              Gemma-3 text encoder (GGUF).
 * @param connectors_path       Embeddings connectors (safetensors).
 * @param n_threads             CPU threads (0 = auto-detect).
 * @param vae_decode_only       true for T2V-only; false to enable I2V VAE encode.
 * @param backend               "cpu", "vulkan", "metal", or NULL for auto.
 * @return Opaque context pointer; free with free_sd_ctx().
 */
static inline sd_ctx_t* ltx2_new_ctx(
    const char* diffusion_model_path,
    const char* vae_path,
    const char* audio_vae_path,
    const char* llm_path,
    const char* connectors_path,
    int         n_threads,
    bool        vae_decode_only,
    const char* backend)
{
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    ltx2_ctx_params_set_defaults(&p);

    p.diffusion_model_path       = diffusion_model_path;
    p.vae_path                   = vae_path;
    p.audio_vae_path             = audio_vae_path;
    p.llm_path                   = llm_path;
    p.embeddings_connectors_path = connectors_path;
    p.n_threads                  = n_threads;
    p.vae_decode_only            = vae_decode_only;
    if (backend != NULL) {
        p.backend = backend;
    }

    return new_sd_ctx(&p);
}

/* -------------------------------------------------------------------------
 * LTX-2 video generation
 * ---------------------------------------------------------------------- */

/**
 * Generate a video from a text prompt (T2V).
 *
 * @param ctx           Context from ltx2_new_ctx() or new_sd_ctx().
 * @param prompt        Text description of the desired video.
 * @param neg_prompt    Negative prompt (NULL uses a sensible default).
 * @param width         Output width in pixels (multiple of 32 recommended).
 * @param height        Output height in pixels (multiple of 32 recommended).
 * @param frames        Number of frames to generate (e.g. 33 ≈ 1.4s at 24fps).
 * @param fps           Container frame-rate written to the output file.
 * @param seed          RNG seed; -1 for random.
 * @param frames_out    Receives pointer to allocated frame array.
 * @param n_frames_out  Receives the number of frames written.
 * @param audio_out     Receives audio track (NULL if audio VAE not loaded).
 * @return true on success.
 */
static inline bool ltx2_generate_t2v(
    sd_ctx_t*    ctx,
    const char*  prompt,
    const char*  neg_prompt,
    int          width,
    int          height,
    int          frames,
    int          fps,
    int64_t      seed,
    sd_image_t** frames_out,
    int*         n_frames_out,
    sd_audio_t** audio_out)
{
    sd_vid_gen_params_t vp;
    sd_vid_gen_params_init(&vp);
    ltx2_vid_params_set_defaults(&vp);
    vp.prompt          = prompt;
    vp.negative_prompt = neg_prompt
        ? neg_prompt
        : "worst quality, low quality, blurry, distorted, artifacts";
    vp.width           = width;
    vp.height          = height;
    vp.video_frames    = frames;
    vp.fps             = fps;
    vp.seed            = seed;
    return generate_video(ctx, &vp, frames_out, n_frames_out, audio_out);
}

/**
 * Generate a video from a text prompt + reference image (I2V).
 *
 * @param init_image  Starting frame (sd_image_t, channel == 3, RGB).
 *                    Width/height fields of init_image need not match output
 *                    dimensions; the engine rescales automatically.
 */
static inline bool ltx2_generate_i2v(
    sd_ctx_t*    ctx,
    const char*  prompt,
    const char*  neg_prompt,
    sd_image_t   init_image,
    int          width,
    int          height,
    int          frames,
    int          fps,
    int64_t      seed,
    sd_image_t** frames_out,
    int*         n_frames_out,
    sd_audio_t** audio_out)
{
    sd_vid_gen_params_t vp;
    sd_vid_gen_params_init(&vp);
    ltx2_vid_params_set_defaults(&vp);
    vp.prompt          = prompt;
    vp.negative_prompt = neg_prompt
        ? neg_prompt
        : "worst quality, low quality, blurry, distorted, artifacts";
    vp.init_image      = init_image;
    vp.width           = width;
    vp.height          = height;
    vp.video_frames    = frames;
    vp.fps             = fps;
    vp.seed            = seed;
    return generate_video(ctx, &vp, frames_out, n_frames_out, audio_out);
}

#ifdef __cplusplus
}
#endif

#endif /* __LTX2_H__ */
