#ifndef __COREML_TAE_H__
#define __COREML_TAE_H__

#include <cstdint>
#include <memory>
#include <string>

// Runs the tiny autoencoder's decoder (TAESD, TAEF2, or the one built into SDXS) exported to CoreML
// (see script/coreml/convert_tae_decoder.py) in place of the ggml TinyDecoder.
// Expected model I/O (float32, fixed shapes):
//   latent [N, C, h, w] -> image [N, 3, s*h, s*w], in [0, 1] before sd.cpp clamps it
// s is 8, or 16 for TAEF2, whose graph includes the 2x2 unpatchify of FLUX.2's 128-channel latent.
// Compute units come from SD_COREML_COMPUTE_UNITS: all | cpu_ne (default) | cpu_gpu | cpu.
class CoreMLTAEDecoder {
public:
    static std::unique_ptr<CoreMLTAEDecoder> load(const std::string& path);
    virtual ~CoreMLTAEDecoder() = default;

    // Image pixels per latent pixel along each axis: 8, or 16 for TAEF2.
    virtual int64_t scale() const = 0;

    // latent is [N, C, h, w] and image is [N, 3, s*h, s*w], both contiguous float32. Both shapes must match
    // the model's, so the decoder never writes more than image_shape holds.
    virtual bool decode(const float* latent, const int64_t latent_shape[4], float* image, const int64_t image_shape[4]) = 0;
};

#endif  // __COREML_TAE_H__
