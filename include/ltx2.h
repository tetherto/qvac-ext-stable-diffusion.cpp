/**
 * @file ltx2.h
 * @brief Public C API facade for LTX-Video 2.3 text-to-video and image-to-video generation.
 *
 * This header wraps the existing stable-diffusion.cpp public API (new_sd_ctx /
 * generate_video) with LTX-2-specific defaults and a minimal typed context so
 * callers never need to fill in sd_ctx_params_t or sd_vid_gen_params_t by hand.
 *
 * Design notes
 * ------------
 * - ltx2_ctx_t is an opaque heap struct that owns an sd_ctx_t* plus the paths
 *   and flags needed to reconstruct it.
 * - All returned sd_image_t arrays are heap-allocated (malloc).  The caller is
 *   responsible for freeing each frame's .data pointer and then the array
 *   itself with the standard C free().  This matches the ownership contract of
 *   generate_video() in stable-diffusion.h.
 * - Coordinates / dimensions are in pixels; all must be positive multiples of
 *   32 (the LTX-2.3 spatial compression factor).  The underlying
 *   generate_video() will round up to the next valid multiple automatically,
 *   but callers are encouraged to pass aligned values.
 * - LTX2_SCHEDULER (flow-matching scheduler with token-count-dependent sigma
 *   shift, max_shift=2.05, base_shift=0.95) will be used automatically once
 *   the upstream sync lands and adds the enum value to scheduler_t.  Until
 *   that sync, the implementation falls back to DISCRETE_SCHEDULER with a
 *   corrective flow_shift of 2.37 and logs a one-time warning.
 *
 * Post-sync migration
 * -------------------
 * When src/denoiser.hpp gains LTX2_SCHEDULER (enum value 11, registered as
 * "ltx2") the guard in ltx2_api.cpp switches automatically; no changes to
 * this header are required.
 */

#ifndef LTX2_H
#define LTX2_H

#include "stable-diffusion.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifndef SD_BUILD_SHARED_LIB
#    define LTX2_API
#  else
#    ifdef SD_BUILD_DLL
#      define LTX2_API __declspec(dllexport)
#    else
#      define LTX2_API __declspec(dllimport)
#    endif
#  endif
#else
#  if __GNUC__ >= 4
#    define LTX2_API __attribute__((visibility("default")))
#  else
#    define LTX2_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque context for LTX-Video 2.3 operations.
 *
 * Obtain via ltx2_new_ctx(); release via ltx2_free_ctx().
 * Must not be accessed directly – the layout is an internal implementation
 * detail that may change between versions.
 */
typedef struct ltx2_ctx_t ltx2_ctx_t;

/* -------------------------------------------------------------------------
 * Context lifecycle
 * ---------------------------------------------------------------------- */

/**
 * @brief Allocate and initialise an LTX-2.3 inference context.
 *
 * Loads the three model files into the stable-diffusion.cpp backend and
 * returns a context that can be used for both T2V and I2V generation.  The
 * function blocks until all weights are loaded; for the 22 B model this can
 * take several seconds.
 *
 * @param diffusion_model_path  Path to the LTX-2.3 DiT GGUF or safetensors
 *                              file (required; must not be NULL).
 * @param vae_path              Path to the LTX video VAE safetensors or GGUF
 *                              file (required; must not be NULL).
 * @param gemma_path            Path to the Gemma-3-12B text encoder GGUF
 *                              file (required; must not be NULL).  Loaded
 *                              under prefix "text_encoders.llm.".
 * @param n_threads             Number of CPU threads.  Pass -1 to use the
 *                              value returned by sd_get_num_physical_cores().
 * @param wtype                 Weight type override.  Pass SD_TYPE_F16 for
 *                              full precision, SD_TYPE_COUNT to preserve the
 *                              types stored in the checkpoint files.
 *
 * @return A newly allocated ltx2_ctx_t on success, or NULL if any model file
 *         cannot be loaded or memory allocation fails.  The returned pointer
 *         must be released with ltx2_free_ctx().
 *
 * @note Flash-attention (diffusion_flash_attn) is enabled by default; this
 *       is strongly recommended for the 22 B model.
 */
LTX2_API ltx2_ctx_t* ltx2_new_ctx(const char*      diffusion_model_path,
                                   const char*      vae_path,
                                   const char*      gemma_path,
                                   int              n_threads,
                                   enum sd_type_t   wtype);

/**
 * @brief Release all resources owned by a context.
 *
 * Safe to call with NULL (no-op).  After this call the pointer is dangling
 * and must not be used.
 *
 * @param ctx  Context to release.
 */
LTX2_API void ltx2_free_ctx(ltx2_ctx_t* ctx);

/* -------------------------------------------------------------------------
 * Video generation
 * ---------------------------------------------------------------------- */

/**
 * @brief Generate a video from a text prompt (T2V).
 *
 * Constructs a fully-noised latent, runs the LTX-2.3 DiT denoising loop with
 * LTX2_SCHEDULER + Euler sampler + flow_shift=2.37, decodes the result
 * through the video VAE, and returns the decoded frames.
 *
 * @param ctx              Context created by ltx2_new_ctx().
 * @param prompt           UTF-8 encoded positive text prompt.  Must not be
 *                         NULL; use "" for an empty prompt.
 * @param negative_prompt  Negative prompt.  May be NULL (treated as "").
 * @param width            Output frame width in pixels.  Must be > 0; will
 *                         be rounded up to the next multiple of 32 if needed.
 * @param height           Output frame height in pixels.  Same rounding rule.
 * @param video_frames     Total number of output frames (e.g. 33 for ~1.4 s
 *                         at 24 fps).  The LTX-2 temporal compression factor
 *                         is 8, so valid values satisfy (video_frames - 1)
 *                         divisible by 8; the implementation will round.
 * @param fps              Playback frame rate stored in the output metadata
 *                         (informational; does not affect generation).
 * @param sample_steps     Number of denoising steps (typical: 30–50).
 * @param cfg_scale        Classifier-free guidance scale for the text
 *                         conditioning.  Typical range: 3.0 – 7.0.
 * @param seed             RNG seed.  Pass -1 to sample from time(NULL).
 * @param out_num_frames   On success set to the number of frames in the
 *                         returned array.  Must not be NULL.
 *
 * @return Heap-allocated array of *out_num_frames sd_image_t structs, each
 *         with .data pointing to a malloc'd RGB (channel=3) pixel buffer of
 *         size width * height * 3 bytes.  The caller must free each .data
 *         member and then the array pointer itself with free().  Returns NULL
 *         on error (ctx is NULL, model not loaded, OOM, etc.).
 */
LTX2_API sd_image_t* ltx2_generate_t2v(ltx2_ctx_t*  ctx,
                                        const char*  prompt,
                                        const char*  negative_prompt,
                                        int          width,
                                        int          height,
                                        int          video_frames,
                                        int          fps,
                                        int          sample_steps,
                                        float        cfg_scale,
                                        int64_t      seed,
                                        int*         out_num_frames);

/**
 * @brief Generate a video from an initial image and a text prompt (I2V).
 *
 * VAE-encodes @p init_image, places it at temporal position 0 of the latent,
 * and runs the standard LTX-2.3 denoising loop with the per-frame denoise
 * mask conditioned on @p strength.  Free frames are fully denoised; the
 * conditioning frame is partially denoised according to (1 - strength).
 *
 * @param ctx              Context created by ltx2_new_ctx().
 * @param init_image       Starting frame.  .width / .height must match the
 *                         requested @p width and @p height; .channel must be
 *                         3 (RGB); .data must be non-NULL.  The pixel buffer
 *                         is read but not modified or freed by this function.
 * @param prompt           UTF-8 positive text prompt.  Must not be NULL.
 * @param negative_prompt  Negative prompt.  May be NULL.
 * @param width            Output frame width.  Should match init_image.width.
 * @param height           Output frame height.  Should match init_image.height.
 * @param video_frames     Total number of output frames.
 * @param fps              Playback frame rate (informational).
 * @param sample_steps     Number of denoising steps.
 * @param cfg_scale        Classifier-free guidance scale.
 * @param seed             RNG seed (-1 for time-based).
 * @param out_num_frames   Set to the frame count of the returned array.
 *
 * @return Same ownership and error semantics as ltx2_generate_t2v().
 */
LTX2_API sd_image_t* ltx2_generate_i2v(ltx2_ctx_t*  ctx,
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
                                        int*         out_num_frames);

#ifdef __cplusplus
}
#endif

#endif /* LTX2_H */
