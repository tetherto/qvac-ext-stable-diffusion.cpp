#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lam_a2e_context lam_a2e_context;

typedef enum lam_a2e_status {
    LAM_A2E_STATUS_OK = 0,
    LAM_A2E_STATUS_INVALID_ARGUMENT = 1,
    LAM_A2E_STATUS_MODEL_LOAD_FAILED = 2,
    LAM_A2E_STATUS_NOT_IMPLEMENTED = 3,
} lam_a2e_status;

typedef struct lam_a2e_params {
    const char * model_path;
    int32_t identity_index;
    int32_t n_threads;
    bool use_gpu;
} lam_a2e_params;

typedef struct lam_a2e_frame {
    int64_t timestamp_us;
    float arkit_52[52];
} lam_a2e_frame;

lam_a2e_context * lam_a2e_create(const lam_a2e_params * params);
void lam_a2e_free(lam_a2e_context * ctx);

lam_a2e_status lam_a2e_process_pcm_f32(
    lam_a2e_context * ctx,
    const float * pcm,
    int64_t pcm_sample_count,
    int32_t sample_rate,
    lam_a2e_frame ** frames,
    int32_t * frame_count);

void lam_a2e_free_frames(lam_a2e_frame * frames);
const char * lam_a2e_get_last_error(const lam_a2e_context * ctx);

#ifdef __cplusplus
}
#endif
