#ifndef __SD_FLUX_COREML_H__
#define __SD_FLUX_COREML_H__

#include <cstdint>
#include <string>

// Core ML sidecars use float32 I/O with shapes in reverse ggml dimension
// order: ggml [W,H,C,N] is Core ML [N,C,H,W]. All arrays are contiguous.
struct FluxCoreMLTensor {
    const float* data;
    int64_t ne[4];
};

struct FluxCoreMLModel;
struct FluxCoreMLVAEModel;

FluxCoreMLModel* flux_coreml_open(const char* path, std::string* error);
void flux_coreml_close(FluxCoreMLModel* model);
bool flux_coreml_predict(FluxCoreMLModel* model,
                         const FluxCoreMLTensor& latent,
                         const FluxCoreMLTensor& timesteps,
                         const FluxCoreMLTensor& context,
                         const FluxCoreMLTensor* pooled,
                         const FluxCoreMLTensor* guidance,
                         float* output,
                         std::string* error);

FluxCoreMLVAEModel* flux_coreml_vae_open(const char* path, std::string* error);
void flux_coreml_vae_close(FluxCoreMLVAEModel* model);
bool flux_coreml_vae_matches_input(FluxCoreMLVAEModel* model, const int64_t ne[4]);
bool flux_coreml_vae_predict(FluxCoreMLVAEModel* model,
                             const FluxCoreMLTensor& latent,
                             float* output,
                             std::string* error);

#endif
