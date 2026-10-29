#include "core/fit_params.h"
#include "abot_world.hpp"
#include "conditioning/conditioner.hpp"
#include "model/vae/wan_vae.hpp"
#include "stable-diffusion.h"
#include "generation_info.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>

#include "core/ggml_extend.h"
#include "core/util.h"
#include "pipeline/diffusion_engine.h"
#include "pipeline/generation.h"
#include "pipeline/request.h"

#define NONE_STR "NONE"

static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
    float reuse_threshold = params.reuse_threshold;
    if (reuse_threshold == INFINITY) {
        if (params.mode == SD_CACHE_EASYCACHE) {
            reuse_threshold = 0.2f;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0f;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

const char* sd_type_name(enum sd_type_t type) {
    if (type == SD_TYPE_F8_E4M3) {
        return "f8_e4m3";
    }
    if (type == SD_TYPE_F8_E5M2) {
        return "f8_e5m2";
    }
    const auto ggml_type = sd_type_to_ggml_type(type);
    if (ggml_type != GGML_TYPE_COUNT) {
        return ggml_type_name(ggml_type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    if (!strcmp(str, "f8_e4m3")) {
        return SD_TYPE_F8_E4M3;
    }
    if (!strcmp(str, "f8_e5m2")) {
        return SD_TYPE_F8_E5M2;
    }
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
    "dpm++2m_sde",
    "dpm++2m_sde_bt",
    "lms",
};

static_assert(SAMPLE_METHOD_COUNT == sizeof(sample_method_to_str) / sizeof(sample_method_to_str[0]),
              "\nnumber of elements in sample_method_to_str[] != SAMPLE_METHOD_COUNT");

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "ltx2",
    "logit_normal",
    "flux2",
    "flux",
    "beta",
    "llada_image",
};

static_assert(SCHEDULER_COUNT == sizeof(scheduler_to_str) / sizeof(scheduler_to_str[0]),
              "\nnumber of elements in scheduler_to_str[] != SCHEDULER_COUNT");

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    if (!strcmp(str, "normal")) {
        return DISCRETE_SCHEDULER;
    }
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "sefi_flow",
    "minit2i_flow",
    "sensenova_u1_flow",
    "flux3_flow",
    "h3_av_flow",
    "ming_image_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction >= EPS_PRED && prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return NONE_STR;
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

const char* hires_upscaler_to_str[] = {
    "None",
    "Latent",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Lanczos",
    "Nearest",
    "Model",
};

const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler) {
    if (upscaler >= SD_HIRES_UPSCALER_NONE && upscaler < SD_HIRES_UPSCALER_COUNT) {
        return hires_upscaler_to_str[upscaler];
    }
    return NONE_STR;
}

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str) {
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; i++) {
        if (!strcmp(str, hires_upscaler_to_str[i])) {
            return (enum sd_hires_upscaler_t)i;
        }
    }
    return SD_HIRES_UPSCALER_COUNT;
}

const char* sd_vae_format_name(enum sd_vae_format_t format) {
    switch (format) {
        case SD_VAE_FORMAT_AUTO:
            return "auto";
        case SD_VAE_FORMAT_FLUX:
            return "flux";
        case SD_VAE_FORMAT_SD3:
            return "sd3";
        case SD_VAE_FORMAT_FLUX2:
            return "flux2";
        case SD_VAE_FORMAT_WAN:
            return "wan";
        default:
            return NONE_STR;
    }
}

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = INFINITY;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
    cache_params->spectrum_w                  = 0.40f;
    cache_params->spectrum_m                  = 3;
    cache_params->spectrum_lam                = 1.0f;
    cache_params->spectrum_window_size        = 2;
    cache_params->spectrum_flex_window        = 0.50f;
    cache_params->spectrum_warmup_steps       = 4;
    cache_params->spectrum_stop_percent       = 0.9f;
}

void sd_hires_params_init(sd_hires_params_t* hires_params) {
    *hires_params                     = {};
    hires_params->enabled             = false;
    hires_params->upscaler            = SD_HIRES_UPSCALER_LATENT;
    hires_params->model_path          = nullptr;
    hires_params->scale               = 2.0f;
    hires_params->target_width        = 0;
    hires_params->target_height       = 0;
    hires_params->steps               = 0;
    hires_params->denoising_strength  = 0.7f;
    hires_params->upscale_tile_size   = 128;
    hires_params->custom_sigmas       = nullptr;
    hires_params->custom_sigmas_count = 0;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                           = {};
    sd_ctx_params->n_threads                 = sd_get_num_physical_cores();
    sd_ctx_params->conditioning_cache_size   = 4;
    sd_ctx_params->wtype                     = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                  = CUDA_RNG;
    sd_ctx_params->sampler_rng_type          = RNG_TYPE_COUNT;
    sd_ctx_params->prediction                = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode           = LORA_APPLY_AUTO;
    sd_ctx_params->max_vram                  = nullptr;
    sd_ctx_params->disable_prefetch          = false;
    sd_ctx_params->disable_segmented_compute = false;
    sd_ctx_params->batched_cfg               = true;
    sd_ctx_params->eager_load                = false;
    sd_ctx_params->enable_mmap               = false;
    sd_ctx_params->flash_attn_mode           = nullptr;
    sd_ctx_params->conv_direct_mode          = nullptr;
    sd_ctx_params->diffusion_flash_attn      = false;
    sd_ctx_params->sage_attn                 = false;
    sd_ctx_params->linear_scale              = 0.f;
    sd_ctx_params->attn_scale                = 0.f;
    sd_ctx_params->vae_format                = SD_VAE_FORMAT_AUTO;
    sd_ctx_params->backend                   = nullptr;
    sd_ctx_params->params_backend            = nullptr;
    sd_ctx_params->split_mode                = nullptr;
    sd_ctx_params->auto_fit                  = true;
    sd_ctx_params->rpc_servers               = nullptr;
    sd_ctx_params->model_args                = nullptr;
    sd_ctx_params->pulid_weights_path        = nullptr;
    sd_ctx_params->preferred_gpu_backend = SD_BACKEND_PREF_GPU;
    sd_ctx_params->vae_auto_cpu_fallback = false;
    sd_ctx_params->vae_auto_cpu_fallback_memory_ratio = 0.9f;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(8192);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 8192 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "clip_vision_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "tokenizer: %s\n"
             "diffusion_model_path: %s\n"
             "high_noise_diffusion_model_path: %s\n"
             "uncond_diffusion_model_path: %s\n"
             "embeddings_connectors_path: %s\n"
             "vae_path: %s\n"
             "audio_vae_path: %s\n"
             "audio_encoder_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "photo_maker_path: %s\n"
             "pulid_weights_path: %s\n"
             "tensor_type_rules: %s\n"
             "n_threads: %d\n"
             "conditioning_cache_size: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "max_vram: %s\n"
             "disable_prefetch: %s\n"
             "disable_segmented_compute: %s\n"
             "eager_load: %s\n"
             "backend: %s\n"
             "params_backend: %s\n"
             "split_mode: %s\n"
             "model_args: %s\n"
             "auto_fit: %s\n"
             "vae_auto_cpu_fallback: %s\n"
             "vae_auto_cpu_fallback_memory_ratio: %.3f\n"
             "flash_attn: %s\n"
             "flash_attn_mode: %s\n"
             "conv_direct_mode: %s\n"
             "diffusion_flash_attn: %s\n"
             "sage_attn: %s\n"
             "linear_scale: %g\n"
             "attn_scale: %g\n"
             "vae_format: %s\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->clip_vision_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->tokenizer),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path),
             SAFE_STR(sd_ctx_params->uncond_diffusion_model_path),
             SAFE_STR(sd_ctx_params->embeddings_connectors_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->audio_vae_path),
             SAFE_STR(sd_ctx_params->audio_encoder_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->photo_maker_path),
             SAFE_STR(sd_ctx_params->pulid_weights_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             sd_ctx_params->n_threads,
             sd_ctx_params->conditioning_cache_size,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             SAFE_STR(sd_ctx_params->max_vram),
             BOOL_STR(sd_ctx_params->disable_prefetch),
             BOOL_STR(sd_ctx_params->disable_segmented_compute),
             BOOL_STR(sd_ctx_params->eager_load),
             SAFE_STR(sd_ctx_params->backend),
             SAFE_STR(sd_ctx_params->params_backend),
             SAFE_STR(sd_ctx_params->split_mode),
             SAFE_STR(sd_ctx_params->model_args),
             BOOL_STR(sd_ctx_params->auto_fit),
             BOOL_STR(sd_ctx_params->vae_auto_cpu_fallback),
             sd_ctx_params->vae_auto_cpu_fallback_memory_ratio,
             BOOL_STR(sd_ctx_params->flash_attn),
             sd_ctx_params->flash_attn_mode && sd_ctx_params->flash_attn_mode[0] ? sd_ctx_params->flash_attn_mode : "auto (legacy flags honored)",
             sd_ctx_params->conv_direct_mode && sd_ctx_params->conv_direct_mode[0] ? sd_ctx_params->conv_direct_mode : "auto (legacy flags honored)",
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->sage_attn),
             sd_ctx_params->linear_scale,
             sd_ctx_params->attn_scale,
             sd_vae_format_name(sd_ctx_params->vae_format));

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
    sample_params->extra_sample_args           = nullptr;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "shifted_timestep: %d, "
             "flow_shift: %.2f, "
             "extra_sample_args: %s)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->shifted_timestep,
             sample_params->flow_shift,
             SAFE_STR(sample_params->extra_sample_args));

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip           = -1;
    sd_img_gen_params->ref_images_count    = 0;
    sd_img_gen_params->ref_image_args      = "";
    sd_img_gen_params->width               = 512;
    sd_img_gen_params->height              = 512;
    sd_img_gen_params->strength            = 0.75f;
    sd_img_gen_params->seed                = -1;
    sd_img_gen_params->batch_count         = 1;
    sd_img_gen_params->control_strength    = 0.9f;
    sd_img_gen_params->ip_adapter_strength = 1.0f;
    sd_img_gen_params->qwen_image_layers   = 3;
    sd_img_gen_params->circular_x          = false;
    sd_img_gen_params->circular_y          = false;
    sd_img_gen_params->pm_params           = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->pulid_params        = {nullptr, 1.0f};
    sd_img_gen_params->vae_tiling_params   = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_cache_params_init(&sd_img_gen_params->cache);
    sd_hires_params_init(&sd_img_gen_params->hires);
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "qwen_image_layers: %d\n"
             "ref_images_count: %d\n"
             "ref_image_args: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s (temporal=%s, extra_tiling_args=%s)\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "hires: {enabled=%s, upscaler=%s, model_path=%s, scale=%.2f, target=%dx%d, steps=%d, denoising_strength=%.2f}\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->qwen_image_layers,
             sd_img_gen_params->ref_images_count,
             SAFE_STR(sd_img_gen_params->ref_image_args),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.temporal_tiling),
             SAFE_STR(sd_img_gen_params->vae_tiling_params.extra_tiling_args),
             BOOL_STR(sd_img_gen_params->circular_x),
             BOOL_STR(sd_img_gen_params->circular_y),
             BOOL_STR(sd_img_gen_params->hires.enabled),
             sd_hires_upscaler_name(sd_img_gen_params->hires.upscaler),
             SAFE_STR(sd_img_gen_params->hires.model_path),
             sd_img_gen_params->hires.scale,
             sd_img_gen_params->hires.target_width,
             sd_img_gen_params->hires.target_height,
             sd_img_gen_params->hires.steps,
             sd_img_gen_params->hires.denoising_strength);
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    } else if (sd_img_gen_params->cache.mode == SD_CACHE_UCACHE) {
        cache_mode_str = "ucache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             get_cache_reuse_threshold(sd_img_gen_params->cache),
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}

void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    *sd_vid_gen_params = {};
    sd_sample_params_init(&sd_vid_gen_params->sample_params);
    sd_sample_params_init(&sd_vid_gen_params->high_noise_sample_params);
    sd_vid_gen_params->high_noise_sample_params.sample_steps = -1;
    sd_vid_gen_params->width                                 = 512;
    sd_vid_gen_params->height                                = 512;
    sd_vid_gen_params->strength                              = 0.75f;
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->fps                                   = 16;
    sd_vid_gen_params->moe_boundary                          = 0.875f;
    sd_vid_gen_params->vace_strength                         = 1.f;
    sd_vid_gen_params->reference_images                      = nullptr;
    sd_vid_gen_params->reference_images_count                = 0;
    sd_vid_gen_params->reference_attention_strength          = 1.f;
    sd_vid_gen_params->reference_downscale_factor            = 1.f;
    sd_vid_gen_params->vae_tiling_params                     = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_vid_gen_params->hires.enabled                         = false;
    sd_vid_gen_params->hires.upscaler                        = SD_HIRES_UPSCALER_LATENT;
    sd_vid_gen_params->hires.scale                           = 2.f;
    sd_vid_gen_params->hires.target_width                    = 0;
    sd_vid_gen_params->hires.target_height                   = 0;
    sd_vid_gen_params->hires.steps                           = 0;
    sd_vid_gen_params->hires.denoising_strength              = 0.7f;
    sd_vid_gen_params->hires.upscale_tile_size               = 128;
    sd_vid_gen_params->hires.custom_sigmas                   = nullptr;
    sd_vid_gen_params->hires.custom_sigmas_count             = 0;
    sd_vid_gen_params->circular_x                            = false;
    sd_vid_gen_params->circular_y                            = false;
    sd_vid_gen_params->ref_image_args                        = "";
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    bool initialized = false;
    try {
        initialized = sd_ctx->sd->init(sd_ctx_params);
    } catch (const std::exception& error) {
        LOG_ERROR("failed to initialize Stable Diffusion context: %s", error.what());
    } catch (...) {
        LOG_ERROR("failed to initialize Stable Diffusion context: unknown exception");
    }
    if (!initialized) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode) {
    if (sd_ctx && sd_ctx->sd) {
        if (mode < SD_CANCEL_ALL || mode > SD_CANCEL_RESET) {
            mode = SD_CANCEL_ALL;
        }
        sd_ctx->sd->set_cancel_flag(mode);
    }
}

void free_sd_audio(sd_audio_t* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}

SD_API bool sd_ctx_uses_reference_images(const sd_ctx_t* sd_ctx) {
    return sd_ctx != nullptr && sd_ctx->sd != nullptr &&
           (sd_version_is_flux2(sd_ctx->sd->version) || sd_ctx->sd->version == VERSION_QWEN_IMAGE_2_1);
}

SD_API int sd_ctx_get_image_channels(const sd_ctx_t* sd_ctx) {
    return sd_ctx != nullptr && sd_ctx->sd != nullptr ? sd_ctx->sd->get_image_channels() : 0;
}

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    if (sd_ctx->sd->config_->animatediff_loaded && sd_version_supports_animatediff(sd_ctx->sd->version)) {
        return true;
    }
    return sd_version_supports_video_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || path == nullptr) {
        return false;
    }
    return sd_ctx->sd->load_control_net_from_file(path);
}

SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_ctx->sd->unload_control_net();
}

SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    if (sd_ctx->sd->control_net != nullptr) {
        return true;
    }
    // z-image keeps control net weights inside the diffusion runner, registered
    // under ModelComponent::ControlNet instead of a separate control_net runner.
    return sd_ctx->sd->model_manager != nullptr &&
           sd_ctx->sd->model_manager->registered_params_size({ModelComponent::ControlNet}) > 0;
}

const char* sd_get_model_version_name(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->version >= VERSION_COUNT) {
        return "Unknown";
    }
    return sd_version_name(sd_ctx->sd->version);
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    return sd::pipeline::default_sample_method(sd_ctx != nullptr ? sd_ctx->sd : nullptr);
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    return sd::pipeline::default_scheduler(sd_ctx != nullptr ? sd_ctx->sd : nullptr, sample_method);
}

SD_API bool generate_image(sd_ctx_t* sd_ctx,
                           const sd_img_gen_params_t* params,
                           sd_image_generation_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    if (!result_out || !sd_ctx || !sd_ctx->sd || !params)
        return false;
    StableDiffusionGGML::ExecutionScope execution(*sd_ctx->sd);
    if (!execution.ready)
        return false;
    sd_image_t* images = nullptr;
    int count          = 0;
    std::vector<nlohmann::json> infos;
    try {
        if (!sd::pipeline::generate_image(sd_ctx->sd, params, &images, &count, &infos)) {
            free_sd_images(images, count);
            return false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("image generation failed: %s", e.what());
        free_sd_images(images, count);
        return false;
    }
    return sd::make_image_generation_result(images, count, infos, result_out);
}

SD_API int sd_get_effective_video_fps(const sd_ctx_t* sd_ctx,
                                      const sd_vid_gen_params_t* sd_vid_gen_params) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_vid_gen_params == nullptr) {
        return 0;
    }

    return sd::pipeline::effective_video_fps(sd_ctx->sd, sd_vid_gen_params->fps);
}

SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* params,
                           sd_video_generation_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    if (!result_out || !sd_ctx || !sd_ctx->sd || !params)
        return false;
    StableDiffusionGGML::ExecutionScope execution(*sd_ctx->sd);
    if (!execution.ready)
        return false;
    auto* result = static_cast<sd_video_generation_result_t*>(calloc(1, sizeof(sd_video_generation_result_t)));
    if (!result)
        return false;
    try {
        nlohmann::json info;
        if (!sd::pipeline::generate_video(sd_ctx->sd, params, &result->frames, &result->frame_count,
                                          &result->audio, &result->fps, &info)) {
            free_sd_video_generation_result(result);
            return false;
        }
        result->generation_info = sd::copy_generation_info(info);
        if (!result->generation_info) {
            free_sd_video_generation_result(result);
            return false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("video generation failed: %s", e.what());
        free_sd_video_generation_result(result);
        return false;
    }
    *result_out = result;
    return true;
}

SD_API void free_sd_images(sd_image_t* result_images, int num_images) {
    if (result_images == nullptr) {
        return;
    }

    for (int i = 0; i < num_images; ++i) {
        if (result_images[i].data != nullptr) {
            free(result_images[i].data);
            result_images[i].data = nullptr;
        }
    }

    free(result_images);
}


struct sd_abot_session_t { SDBackendManager backend_manager; ABOT::AbotWalkSession session; };
void sd_abot_session_params_init(sd_abot_session_params_t* p) {
    if (!p) return; *p = {}; p->n_threads = -1; p->seed = 42;
}
void sd_abot_session_params_v2_init(sd_abot_session_params_v2_t* p) {
    if (!p) return; *p = {}; p->n_threads = -1; p->seed = 42;
}
sd_abot_session_t* sd_abot_session_new(const sd_abot_session_params_t* p) {
    if (!p) return nullptr;
    sd_abot_session_params_v2_t p2;
    sd_abot_session_params_v2_init(&p2);
    p2.dit_model_path            = p->dit_model_path;
    p2.taehv_path                = p->taehv_path;
    p2.scene_path                = p->scene_path;
    p2.backend                   = p->backend;
    p2.n_threads                 = p->n_threads;
    p2.seed                      = p->seed;
    p2.num_frame_per_block       = p->num_frame_per_block;
    p2.local_attn_size           = p->local_attn_size;
    p2.offload_params_to_cpu     = p->offload_params_to_cpu;
    p2.kv_cache                  = p->kv_cache;
    p2.profile                   = p->profile;
    return sd_abot_session_new_v2(&p2);
}
sd_abot_session_t* sd_abot_session_new_v2(const sd_abot_session_params_v2_t* p) {
    try {
        if (!p || !p->dit_model_path || !p->taehv_path || !p->scene_path) return nullptr;
        auto s = std::make_unique<sd_abot_session_t>(); std::string error;
        sd::ggml_graph_cut::MaxVramAssignment max_vram;
        max_vram.reset(0.f);
        if (!max_vram.parse(SAFE_STR(p->max_vram), &error)) {
            LOG_ERROR("sd_abot_session_new: %s", error.c_str());
            return nullptr;
        }
        std::string params_backend = p->offload_params_to_cpu ? "*=cpu" : "";
        if (p->params_backend && p->params_backend[0]) {
            if (!params_backend.empty())
                params_backend += ",";
            params_backend += p->params_backend;
        }
        if (!s->backend_manager.init(SAFE_STR(p->backend), params_backend.c_str(), nullptr, false, &error)) {
            LOG_ERROR("sd_abot_session_new: backend init failed: %s", error.c_str()); return nullptr;
        }
        if (s->backend_manager.params_backend_is_disk(SDBackendModule::VAE)) {
            LOG_ERROR("sd_abot_session_new: vae=disk is unsupported for the ABot decoder");
            return nullptr;
        }
        if (!max_vram.canonicalize_backend_keys(&error)) {
            LOG_ERROR("sd_abot_session_new: %s", error.c_str());
            return nullptr;
        }
        ABOT::AbotWorldConfig cfg;
        if (p->num_frame_per_block > 0) cfg.num_frame_per_block = p->num_frame_per_block;
        if (p->local_attn_size > 0) cfg.local_attn_size = p->local_attn_size;
        cfg.profile = p->profile;
        cfg.kv_cache = p->kv_cache;
        cfg.max_graph_vram_bytes = max_vram.bytes_for_backend(s->backend_manager.runtime_backend(SDBackendModule::DIFFUSION));
        cfg.stream_layers        = p->stream_layers;
        cfg.dit_params_on_disk   = s->backend_manager.params_backend_is_disk(SDBackendModule::DIFFUSION);
        cfg.tae_params_on_disk   = s->backend_manager.params_backend_is_disk(SDBackendModule::VAE);
        if (cfg.stream_layers && !s->backend_manager.params_backend_is_cpu(SDBackendModule::DIFFUSION)) {
            LOG_WARN("stream_layers has no effect unless diffusion params backend is cpu; ignoring");
            cfg.stream_layers = false;
        }
        LOG_INFO("ABot-World DiT: runtime=%s, params=%s, graph budget=%.2f MiB, stream_layers=%s",
                 ggml_backend_name(s->backend_manager.runtime_backend(SDBackendModule::DIFFUSION)),
                 cfg.dit_params_on_disk ? "disk" : ggml_backend_name(s->backend_manager.params_backend(SDBackendModule::DIFFUSION)),
                 cfg.max_graph_vram_bytes / (1024.0 * 1024.0), cfg.stream_layers ? "true" : "false");
        const int threads = p->n_threads > 0 ? p->n_threads : sd_get_num_physical_cores();
        if (!s->session.load(s->backend_manager.runtime_backend(SDBackendModule::DIFFUSION),
                             s->backend_manager.params_backend(SDBackendModule::DIFFUSION),
                             s->backend_manager.runtime_backend(SDBackendModule::VAE),
                             s->backend_manager.params_backend(SDBackendModule::VAE),
                             p->dit_model_path, p->taehv_path, p->scene_path, cfg,
                             static_cast<uint64_t>(p->seed), threads)) return nullptr;
        return s.release();
    } catch (const std::exception& e) { LOG_ERROR("sd_abot_session_new: %s", e.what()); return nullptr; }
}

void sd_abot_fit_workload_init(sd_abot_fit_workload_t* workload) {
    if (workload != nullptr) {
        *workload = {100};
    }
}

enum sd_fit_status_t sd_abot_fit_params(const sd_abot_session_params_v2_t* params,
                                        const sd_abot_fit_workload_t* workload,
                                        sd_fit_result_t* result) {
    if (result == nullptr) {
        return SD_FIT_ERROR;
    }
    *result = {};
    if (params == nullptr || workload == nullptr || workload->walk_steps <= 0 || workload->walk_steps > 1000000 ||
        params->num_frame_per_block < 0 || params->num_frame_per_block > 1024 ||
        params->local_attn_size < 0 || params->local_attn_size > 1024) {
        return SD_FIT_ERROR;
    }
    try {
        SDMetadataOnlyReadScope metadata_only;
        std::unique_ptr<sd_abot_session_t> session(sd_abot_session_new_v2(params));
        if (!session) {
            return SD_FIT_ERROR;
        }
        auto& walk = session->session;
        std::vector<GGMLRunner::graph_memory_measurement> records;
        struct MeasureGuard {
            explicit MeasureGuard(std::vector<GGMLRunner::graph_memory_measurement>& records) {
                GGMLRunner::set_measure_mode(true, &records);
            }
            ~MeasureGuard() { GGMLRunner::set_measure_mode(false); }
        } measure_guard(records);
        size_t host_bytes = 0;
        if (!walk.measure_memory(workload->walk_steps, records, host_bytes)) {
            return SD_FIT_ERROR;
        }
        std::vector<sd::fit_params::BackendMemory> memory;
        std::string report = "ABot fit: " + std::to_string(workload->walk_steps) + " walk steps\n";
        report += "  session host buffers: " + std::to_string(host_bytes) + " bytes\n";
        for (auto module : {SDBackendModule::DIFFUSION, SDBackendModule::VAE}) {
            auto runtime = session->backend_manager.runtime_backend(module);
            auto storage = session->backend_manager.params_backend(module);
            std::map<std::string, ggml_tensor*> tensors;
            if (module == SDBackendModule::DIFFUSION) {
                walk.runner->get_param_tensors(tensors, "model.diffusion_model");
            } else {
                walk.tae->get_param_tensors(tensors);
            }
            const auto buft        = ggml_backend_get_default_buffer_type(storage);
            const size_t alignment = ggml_backend_buft_get_alignment(buft);
            size_t params_bytes    = 0;
            for (const auto& tensor : tensors) {
                params_bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensor.second), alignment);
            }
            size_t compute_bytes   = 0;
            size_t measured_params = 0;
            bool buffer_fits       = true;
            for (const auto& record : records) {
                if (record.module == module) {
                    if (!record.valid) {
                        return SD_FIT_ERROR;
                    }
                    compute_bytes   = std::max(compute_bytes, record.configured_compute_bytes);
                    measured_params = std::max(measured_params, record.configured_params_bytes);
                    buffer_fits     = buffer_fits && record.configured_buffer_bytes <= ggml_backend_get_max_buffer_capacity(runtime);
                }
            }
            const bool on_disk = module == SDBackendModule::DIFFUSION && walk.cfg.dit_params_on_disk;
            if (!on_disk) {
                memory.push_back({storage, params_bytes});
            }
            memory.push_back({runtime, compute_bytes});
            if (runtime != storage || on_disk) {
                const auto runtime_buft        = ggml_backend_get_default_buffer_type(runtime);
                const size_t runtime_alignment = ggml_backend_buft_get_alignment(runtime_buft);
                size_t staged_bytes            = 0;
                for (const auto& tensor : tensors) {
                    staged_bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(runtime_buft, tensor.second), runtime_alignment);
                }
                // Disk/streaming placement releases nonresident segments after use.
                if (module == SDBackendModule::DIFFUSION && walk.cfg.max_graph_vram_bytes > 0 &&
                    (walk.cfg.stream_layers || walk.cfg.dit_params_on_disk) && !sd_backend_is_cpu(runtime)) {
                    staged_bytes = std::min(staged_bytes, measured_params + tensors.size() * runtime_alignment);
                }
                memory.push_back({runtime, staged_bytes});
            }
            report += std::string("  ") + sd_backend_module_name(module) + ": params " +
                      std::to_string(params_bytes / (1024 * 1024)) + " MiB, retained graph/cache " +
                      std::to_string(compute_bytes / (1024 * 1024)) + " MiB\n";
            if (!buffer_fits) {
                report += "  graph exceeds the backend's logical buffer capacity\n";
                result->report = static_cast<char*>(malloc(report.size() + 1));
                if (result->report == nullptr)
                    return SD_FIT_ERROR;
                memcpy(result->report, report.c_str(), report.size() + 1);
                return SD_FIT_FAILURE;
            }
        }
        sd::fit_params::FitPlan plan;
        sd::ggml_graph_cut::MaxVramAssignment budgets;
        budgets.reset(0.f);
        const bool checked = sd::fit_params::check_placement(memory, host_bytes, budgets, &plan);
        report += plan.report;
        result->report = static_cast<char*>(malloc(report.size() + 1));
        if (result->report == nullptr) {
            return SD_FIT_ERROR;
        }
        memcpy(result->report, report.c_str(), report.size() + 1);
        if (!sd::fit_params::write_device_use(plan, result)) {
            sd_fit_result_free(result);
            return SD_FIT_ERROR;
        }
        return checked && plan.valid ? SD_FIT_SUCCESS : SD_FIT_FAILURE;
    } catch (const std::exception& error) {
        LOG_ERROR("ABot fit: %s", error.what());
        return SD_FIT_ERROR;
    }
}

sd_image_t* sd_abot_session_step(sd_abot_session_t* s, uint32_t action, int* n) {
    if (n) *n = 0; if (!s) return nullptr;
    try { std::vector<std::vector<uint8_t>> frames; int64_t w = 0, h = 0;
        if (!s->session.step(static_cast<uint8_t>(action), frames, w, h)) return nullptr;
        auto* out = static_cast<sd_image_t*>(calloc(frames.size(), sizeof(sd_image_t))); if (!out) return nullptr;
        for (size_t i=0;i<frames.size();++i) { out[i] = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 3, static_cast<uint8_t*>(malloc(frames[i].size()))}; if (!out[i].data) { sd_abot_session_frames_free(out, static_cast<int>(i)); return nullptr; } memcpy(out[i].data, frames[i].data(), frames[i].size()); }
        if (n) *n = static_cast<int>(frames.size()); return out;
    } catch (const std::exception& e) { LOG_ERROR("sd_abot_session_step: %s", e.what()); s->session.fail_session("exception during step"); return nullptr; }
}
void sd_abot_session_frames_free(sd_image_t* frames, int n) { if (!frames) return; for (int i=0;i<n;++i) free(frames[i].data); free(frames); }
void sd_abot_session_free(sd_abot_session_t* s) { delete s; }
void sd_abot_scene_params_init(sd_abot_scene_params_t* p) { if (!p) return; *p = {}; p->width = 832; p->height = 480; p->n_threads = -1; }
namespace {
struct AbotT5Runner : T5Runner { using T5Runner::T5Runner; };
struct AbotWanVAE : WAN::WanVAERunner { using WAN::WanVAERunner::WanVAERunner; };
}
bool sd_abot_scene_create(const sd_abot_scene_params_t* p) {
    try {
        const bool has_image = p && p->init_image.data;
        if (!p || !p->t5_path || !p->prompt || !p->output_path || (has_image && !p->vae_path) ||
            p->width <= 0 || p->height <= 0 || p->width % 32 || p->height % 32) {
            LOG_ERROR("sd_abot_scene_create: t5, prompt, output and dimensions divisible by 32 are required (VAE for image scenes)"); return false;
        }
        SDBackendManager backends; std::string error;
        if (!backends.init(SAFE_STR(p->backend), nullptr, p->offload_params_to_cpu, false, false, false, &error)) {
            LOG_ERROR("sd_abot_scene_create: backend init failed: %s", error.c_str()); return false;
        }
        const int threads = p->n_threads > 0 ? p->n_threads : sd_get_num_physical_cores();
        // The runners below own the ggml tensors that ~ModelManager's
        // storage-block teardown writes through (state->tensor->buffer =
        // nullptr), and they hold only weak_ptr refs to the manager. Declare
        // them BEFORE the manager so reverse local destruction runs
        // ~ModelManager first, while every registered tensor is still alive
        // (mirrors the member ordering in StableDiffusionGGML).
        std::shared_ptr<AbotT5Runner> t5;
        std::shared_ptr<AbotWanVAE> vae;
        auto model_manager = std::make_shared<ModelManager>();
        model_manager->set_n_threads(threads);
        ModelLoader& model_loader = model_manager->loader();
        if (!model_loader.init_from_file(p->t5_path, "text_encoders.t5xxl.transformer.")) return false;
        t5 = std::make_shared<AbotT5Runner>(backends.runtime_backend(SDBackendModule::TE),
                                              model_loader.get_tensor_storage_map(),
                                              "text_encoders.t5xxl.transformer",
                                              true,
                                              model_manager);
        if (has_image) {
            if (!model_loader.init_from_file(p->vae_path, "first_stage_model.")) return false;
            vae = std::make_shared<AbotWanVAE>(backends.runtime_backend(SDBackendModule::VAE),
                                                model_loader.get_tensor_storage_map(),
                                                "first_stage_model",
                                                false,
                                                VERSION_ABOT_WORLD,
                                                model_manager);
        }
        if (!model_manager->register_runner_params(ModelComponent::Conditioner,
                                                    *t5,
                                                    "text_encoders.t5xxl.transformer",
                                                    ModelManager::ResidencyMode::ParamBackend,
                                                    backends.runtime_backend(SDBackendModule::TE),
                                                    backends.params_backend(SDBackendModule::TE)) ||
            (has_image && !model_manager->register_runner_params(ModelComponent::VAE,
                                                                   *vae,
                                                                   ModelManager::ResidencyMode::ParamBackend,
                                                                   backends.runtime_backend(SDBackendModule::VAE),
                                                                   backends.params_backend(SDBackendModule::VAE))) ||
            !model_manager->validate_registered_tensors() ||
            !model_manager->load_all_params_eagerly()) {
            LOG_ERROR("sd_abot_scene_create: model parameter registration or loading failed");
            return false;
        }
        T5Embedder tokenizer_helper(backends.runtime_backend(SDBackendModule::TE), {}, "", true);
        auto tokens = tokenizer_helper.tokenize(p->prompt, 512, true);
        sd::Tensor<float> prompt = t5->compute(threads,
            sd::Tensor<int32_t>::from_vector(std::get<0>(tokens)),
            sd::Tensor<float>::from_vector(std::get<2>(tokens)));
        if (prompt.empty()) { LOG_ERROR("sd_abot_scene_create: prompt encoding failed"); return false; }
        // The reference WanTextEncoder zeroes embeddings past the real tokens
        // (`u[v:] = 0`). The encoder's attention mask only affects attention
        // inside the encoder, not its output rows, so without this the pack
        // carries live pad-token embeddings in all 512 context rows and the
        // walk collapses into blur from the first generated block. The
        // tokenizer mask is additive: 0.0 = real token, -inf = padding.
        {
            const std::vector<float>& attn_mask = std::get<2>(tokens);
            const int64_t emb_dim               = prompt.shape()[0];
            float* pd                           = prompt.data();
            for (size_t i = 0; i < attn_mask.size(); i++) {
                if (attn_mask[i] != 0.0f) {
                    memset(pd + static_cast<size_t>(emb_dim) * i, 0, static_cast<size_t>(emb_dim) * sizeof(float));
                }
            }
        }
        if (prompt.shape().size() == 2) prompt = prompt.unsqueeze(2);
        sd::Tensor<float> first;
        if (has_image) {
            auto image = sd_image_to_tensor(p->init_image, p->width, p->height);
            auto encoded = vae->encode(threads, image, {}, true);
            first = vae->vae_to_diffusion_latents(vae->vae_output_to_latents(encoded, nullptr));
            if (first.empty()) return false;
        } else { first.resize({p->width / 16, p->height / 16, 48, 1}); std::fill_n(first.data(), first.numel(), 0.f); }
        const int64_t w = first.shape()[0], h = first.shape()[1], c = first.numel() / (w * h);
        if (c != 48) { LOG_ERROR("sd_abot_scene_create: expected 48-channel Wan2.2 latent"); return false; }
        ABOT::AbotSceneCreateConfig cfg; const int64_t k = cfg.ref_slots, g = cfg.ref_grid;
        std::vector<float> refs(static_cast<size_t>(k * 48 * g * g), 0.f), mask(static_cast<size_t>(k), 0.f);
        float first_mask = has_image ? 1.f : 0.f;
        std::vector<ABOT::AbotSceneWriter::Entry> entries = {
            {"prompt_embeds", {1, 512, 4096}, prompt.data(), prompt.numel()},
            {"first_frame_latents", {1, 1, 48, h, w}, first.data(), first.numel()},
            {"ref_latents", {1, k, 48, 1, g, g}, refs.data(), static_cast<int64_t>(refs.size())},
            {"ref_mask", {1, k}, mask.data(), k}, {"first_frame_mask", {1, 1}, &first_mask, 1}};
        return ABOT::AbotSceneWriter::write(p->output_path, entries);
    } catch (const std::exception& e) { LOG_ERROR("sd_abot_scene_create: %s", e.what()); return false; }
}

static void sd_vid_gen_params_from_image_request(sd_vid_gen_params_t* video,
                                                  const sd_img_gen_params_t& image,
                                                  int video_frames) {
    sd_vid_gen_params_init(video);
    video->loras                 = image.loras;
    video->lora_count            = image.lora_count;
    video->prompt                = image.prompt;
    video->negative_prompt       = image.negative_prompt;
    video->clip_skip             = image.clip_skip;
    video->init_image            = image.init_image;
    video->ref_images            = image.ref_images;
    video->ref_images_count      = image.ref_images_count;
    video->width                 = image.width;
    video->height                = image.height;
    video->sample_params         = image.sample_params;
    video->strength              = image.strength;
    video->seed                  = image.seed;
    video->video_frames          = std::max(video_frames, 1);
    video->vae_tiling_params     = image.vae_tiling_params;
    video->cache                 = image.cache;
    video->hires                 = image.hires;
    video->circular_x            = image.circular_x;
    video->circular_y            = image.circular_y;
    if (image.control_image.data != nullptr) {
        video->control_frames      = const_cast<sd_image_t*>(&image.control_image);
        video->control_frames_size = 1;
    }
}

void sd_fit_workload_init(sd_fit_workload_t* workload) {
    if (workload == nullptr) {
        return;
    }
    *workload                   = {};
    workload->width             = 512;
    workload->height            = 512;
    workload->video_frames      = 1;
    workload->vae_tiling_params = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
}

bool sd_model_supports_video(const char* model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return false;
    }
    try {
        SDMetadataOnlyReadScope metadata_only;
        ModelLoader loader;
        return loader.init_from_file_and_convert_name(model_path) &&
               sd_version_supports_video_generation(loader.get_sd_version());
    } catch (const std::exception&) {
        return false;
    }
}

static enum sd_fit_status_t sd_fit_params_impl(const sd_ctx_params_t* sd_ctx_params,
                                               const sd_fit_workload_t* workload,
                                               const sd_upscaler_fit_params_t* upscaler,
                                               sd_fit_result_t* result) {
    if (result == nullptr) {
        return SD_FIT_ERROR;
    }
    *result = {};
    if (sd_ctx_params == nullptr || workload == nullptr) {
        return SD_FIT_ERROR;
    }
    if (workload->image_gen_params != nullptr && workload->video_gen_params != nullptr) {
        LOG_ERROR("fit-params: set at most one complete generation request");
        return SD_FIT_ERROR;
    }
    const bool offload_params_to_cpu = strcmp(SAFE_STR(sd_ctx_params->params_backend), "*=cpu") == 0;
    if (strlen(SAFE_STR(sd_ctx_params->backend)) > 0 ||
        (strlen(SAFE_STR(sd_ctx_params->params_backend)) > 0 && !offload_params_to_cpu)) {
        LOG_WARN("fit-params: explicit backend placement cannot be validated; clear backend and "
                 "params_backend before fitting (params_backend *=cpu is supported)");
        return SD_FIT_FAILURE;
    }

    ggml_time_init();
    int64_t t0 = ggml_time_ms();

    sd_ctx_params_t dry_params = *sd_ctx_params;
    dry_params.auto_fit        = false;
    dry_params.eager_load      = false;

    sd_ctx_t sd_ctx_storage{};
    sd_ctx_t* sd_ctx        = &sd_ctx_storage;
    sd_ctx->sd              = new StableDiffusionGGML();
    sd_ctx->sd->fit_dry_run = true;
    bool init_ok = false;
    try {
        SDMetadataOnlyReadScope metadata_only;
        init_ok = sd_ctx->sd->init(&dry_params);
    } catch (const std::exception& error) {
        LOG_ERROR("fit-params: dry-run model init failed: %s", error.what());
    }
    if (!init_ok) {
        LOG_ERROR("fit-params: dry-run model init failed");
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        return SD_FIT_ERROR;
    }

    const int workload_width = workload->image_gen_params != nullptr ? workload->image_gen_params->width
                               : workload->video_gen_params != nullptr ? workload->video_gen_params->width
                                                                       : workload->width;
    const int workload_height = workload->image_gen_params != nullptr ? workload->image_gen_params->height
                                : workload->video_gen_params != nullptr ? workload->video_gen_params->height
                                                                        : workload->height;
    const int workload_frames = workload->video_gen_params != nullptr ? workload->video_gen_params->video_frames
                                                                       : workload->video_frames;
    const char* prompt         = workload->prompt != nullptr && workload->prompt[0] != '\0'
                                     ? workload->prompt
                                     : "a photo of an astronaut riding a horse on the moon";
    const bool model_video_only = sd_version_supports_video_generation(sd_ctx->sd->version);
    const bool animatediff_capable = sd_ctx->sd->config_->animatediff_loaded &&
                                      sd_version_supports_animatediff(sd_ctx->sd->version);
    const bool animatediff_video = animatediff_capable && workload_frames > 1;
    if ((workload->video_gen_params != nullptr || workload_frames > 1) &&
        !model_video_only && !animatediff_capable) {
        LOG_ERROR("fit-params: a video workload was supplied for an image-only model");
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        return SD_FIT_ERROR;
    }
    const bool video = workload->video_gen_params != nullptr || model_video_only || animatediff_video;
    if (video) {
        upscaler = nullptr;
    }

    sd_tiling_params_t requested_tiling = workload->vae_tiling_params;
    if (workload->image_gen_params != nullptr) {
        requested_tiling = workload->image_gen_params->vae_tiling_params;
    } else if (workload->video_gen_params != nullptr) {
        requested_tiling = workload->video_gen_params->vae_tiling_params;
    }

    // Silence progress from this dry run without replacing the process-wide
    // callback used by concurrent generation calls.
    struct ProgressSuppressionGuard {
        ProgressSuppressionGuard()
            : previous(sd_get_progress_suppressed()) {
            sd_set_progress_suppressed(true);
        }
        ~ProgressSuppressionGuard() {
            sd_set_progress_suppressed(previous);
        }
        bool previous;
    } progress_suppression_guard;

    // run the real generation pipeline in measure mode: every runner builds its graphs,
    // records memory requirements and returns shaped zero tensors, no weights are read
    // models with a vision tower condition on an input image; feed a dummy one so
    // the clip_vision and VAE encode graphs are built and measured too
    std::vector<uint8_t> dummy_image_data;
    sd_image_t dummy_init_image = {0, 0, 3, nullptr};
    if (sd_ctx->sd->clip_vision != nullptr && workload_width > 0 && workload_height > 0) {
        dummy_image_data.assign((size_t)workload_width * workload_height * 3, 128);
        dummy_init_image = {(uint32_t)workload_width, (uint32_t)workload_height, 3, dummy_image_data.data()};
    }

    auto measure = [&](const sd_tiling_params_t& tiling,
                       std::vector<GGMLRunner::graph_memory_measurement>& records) -> bool {
        SDMetadataOnlyReadScope metadata_only;
        records.clear();
        struct MeasureModeGuard {
            explicit MeasureModeGuard(std::vector<GGMLRunner::graph_memory_measurement>* records) {
                GGMLRunner::set_measure_mode(true, records);
            }
            ~MeasureModeGuard() {
                GGMLRunner::set_measure_mode(false);
            }
        } measure_mode_guard(&records);
        bool ok            = false;
        sd_image_t* images = nullptr;
        int num_images     = 0;
        sd_audio_t* audio  = nullptr;
        try {
            if (video) {
                sd_vid_gen_params_t gen;
                if (workload->video_gen_params != nullptr) {
                    gen = *workload->video_gen_params;
                } else if (workload->image_gen_params != nullptr) {
                    LOG_WARN("fit-params: promoting the image request to the video pipeline required by this model");
                    sd_vid_gen_params_from_image_request(&gen,
                                                         *workload->image_gen_params,
                                                         workload_frames);
                } else {
                    sd_vid_gen_params_init(&gen);
                    gen.prompt       = prompt;
                    gen.width        = workload_width;
                    gen.height       = workload_height;
                    gen.video_frames = std::max(workload_frames, 1);
                }
                gen.sample_params.sample_steps        = 2;
                gen.sample_params.custom_sigmas       = nullptr;
                gen.sample_params.custom_sigmas_count = 0;
                if (gen.high_noise_sample_params.sample_steps > 0) {
                    gen.high_noise_sample_params.sample_steps        = 2;
                    gen.high_noise_sample_params.custom_sigmas       = nullptr;
                    gen.high_noise_sample_params.custom_sigmas_count = 0;
                }
                gen.vae_tiling_params = tiling;
                if (gen.init_image.data == nullptr && dummy_init_image.data != nullptr) {
                    gen.init_image = dummy_init_image;
                }
                ok = sd::pipeline::generate_video(sd_ctx->sd, &gen, &images, &num_images, &audio, nullptr);
            } else {
                sd_img_gen_params_t gen;
                if (workload->image_gen_params != nullptr) {
                    gen = *workload->image_gen_params;
                } else {
                    sd_img_gen_params_init(&gen);
                    gen.prompt = prompt;
                    gen.width  = workload_width;
                    gen.height = workload_height;
                }
                gen.sample_params.sample_steps        = 2;
                gen.sample_params.custom_sigmas       = nullptr;
                gen.sample_params.custom_sigmas_count = 0;
                gen.batch_count                       = 1;
                gen.vae_tiling_params                 = tiling;
                if (gen.ip_adapter_image.data == nullptr && dummy_init_image.data != nullptr) {
                    gen.ip_adapter_image = dummy_init_image;  // image models have clip_vision only for ip-adapter
                }
                ok = sd::pipeline::generate_image(sd_ctx->sd, &gen, &images, &num_images);
            }
        } catch (const std::bad_alloc&) {
            LOG_ERROR("fit-params: host memory exhausted while materializing dry-run pipeline tensors");
            ok = false;
        } catch (const std::exception& error) {
            LOG_ERROR("fit-params: dry-run pipeline failed: %s", error.what());
            ok = false;
        }
        free_sd_audio(audio);
        if (images != nullptr) {
            for (int i = 0; i < num_images; i++) {
                free(images[i].data);
            }
            free(images);
        }
        return ok;
    };

    std::vector<GGMLRunner::graph_memory_measurement> records;
    if (!measure(requested_tiling, records) || records.empty()) {
        LOG_ERROR("fit-params: measurement dry run failed");
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        return SD_FIT_ERROR;
    }

    std::map<SDBackendModule, sd::fit_params::ModuleMemory> module_map;
    for (const auto& kv : sd_ctx->sd->fit_module_params_bytes) {
        auto& m        = module_map[kv.first];
        m.module       = kv.first;
        m.params_bytes = kv.second;
    }
    for (const auto& record : records) {
        SDBackendModule module = record.module;
        if (module == SDBackendModule::UNSET) {
            LOG_ERROR("fit-params: unattributed graph measurement from %s",
                      record.desc.c_str());
            delete sd_ctx->sd;
            sd_ctx->sd = nullptr;
            return SD_FIT_ERROR;
        }
        auto& m                = module_map[module];
        m.module               = module;
        m.params_bytes         = std::max(m.params_bytes, record.params_bytes);
        m.compute_bytes        = std::max(m.compute_bytes, record.compute_bytes);
        if (!record.split_segment_params_bytes.empty()) {
            m.split_graph_segment_params.push_back(record.split_segment_params_bytes);
            m.split_graph_segment_compute.push_back(record.split_segment_compute_bytes);
        }
    }
    for (auto module : {SDBackendModule::DIFFUSION, SDBackendModule::TE}) {
        auto it = module_map.find(module);
        if (it != module_map.end()) {
            it->second.splittable = true;
        }
    }

    // price VAE tiling so the planner can fall back to it when full-resolution decode does not fit
    {
        auto it = module_map.find(SDBackendModule::VAE);
        if (it != module_map.end() && !requested_tiling.enabled) {
            sd_tiling_params_t tiled = requested_tiling;
            tiled.enabled            = true;
            std::vector<GGMLRunner::graph_memory_measurement> tiled_records;
            if (measure(tiled, tiled_records)) {
                for (const auto& record : tiled_records) {
                    if (record.module == SDBackendModule::UNSET) {
                        LOG_ERROR("fit-params: unattributed tiled graph measurement from %s",
                                  record.desc.c_str());
                        delete sd_ctx->sd;
                        sd_ctx->sd = nullptr;
                        return SD_FIT_ERROR;
                    }
                    if (record.module == SDBackendModule::VAE) {
                        it->second.compute_bytes_tiled = std::max(it->second.compute_bytes_tiled, record.compute_bytes);
                    }
                }
            }
        }
    }

    std::vector<sd::fit_params::ModuleMemory> modules;
    for (const auto& kv : module_map) {
        modules.push_back(kv.second);
    }
    if (upscaler != nullptr) {
        sd::fit_params::ModuleMemory memory;
        bool measured = false;
        try {
            measured = sd_measure_upscaler(*upscaler, memory);
        } catch (const std::exception& error) {
            LOG_ERROR("fit-params: upscaler measurement failed: %s", error.what());
        }
        if (!measured) {
            delete sd_ctx->sd;
            return SD_FIT_ERROR;
        }
        modules.push_back(std::move(memory));
    }
    sd::fit_params::FitPlan plan;
    bool planned = sd::fit_params::plan_placement(modules,
                                                  sd_ctx->sd->max_vram_assignment,
                                                  &plan,
                                                  offload_params_to_cpu,
                                                  false,
                                                  upscaler != nullptr);

    delete sd_ctx->sd;
    sd_ctx->sd = nullptr;

    result->report        = strdup(plan.report.c_str());
    result->vae_tiling    = plan.vae_tiling;
    result->stream_layers = plan.stream_layers;
    if (!sd::fit_params::write_device_use(plan, result)) {
        sd_fit_result_free(result);
        return SD_FIT_ERROR;
    }
    if (!planned || !plan.valid) {
        return SD_FIT_FAILURE;
    }

    if (plan.changed) {
        result->changed = true;
        if (!plan.runtime_spec.empty()) {
            result->backend = strdup(plan.runtime_spec.c_str());
        }
        if (!plan.params_spec.empty()) {
            result->params_backend = strdup(plan.params_spec.c_str());
        }
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("fit-params: fitting params to free memory took %.2fs", (t1 - t0) * 1.0f / 1000);
    return SD_FIT_SUCCESS;
}

enum sd_fit_status_t sd_fit_params(const sd_ctx_params_t* ctx_params,
                                   const sd_fit_workload_t* workload,
                                   sd_fit_result_t* result) {
    return sd_fit_params_impl(ctx_params, workload, nullptr, result);
}

enum sd_fit_status_t sd_fit_params_with_upscaler(const sd_ctx_params_t* ctx_params,
                                                 const sd_fit_workload_t* workload,
                                                 const sd_upscaler_fit_params_t* upscaler,
                                                 sd_fit_result_t* result) {
    if (upscaler == nullptr) {
        if (result != nullptr) {
            *result = {};
        }
        return SD_FIT_ERROR;
    }
    return sd_fit_params_impl(ctx_params, workload, upscaler, result);
}

void sd_fit_result_free(sd_fit_result_t* result) {
    if (result == nullptr) {
        return;
    }
    free(result->backend);
    free(result->params_backend);
    free(result->report);
    free(result->devices);
    *result = {};
}
