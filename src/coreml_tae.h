#ifndef __COREML_TAE_H__
#define __COREML_TAE_H__

#include <cstdint>
#include <memory>
#include <string>

// Runs the tiny autoencoder's decoder (TAESD, or the one built into SDXS) exported to CoreML
// (see script/coreml/convert_tae_decoder.py) in place of the ggml TinyDecoder.
// Expected model I/O (float32, fixed shapes):
//   latent [N, 4, h, w] -> image [N, 3, 8h, 8w], in [0, 1] before sd.cpp clamps it
// Compute units come from SD_COREML_COMPUTE_UNITS: all | cpu_ne (default) | cpu_gpu | cpu.
class CoreMLTAEDecoder {
public:
    static std::unique_ptr<CoreMLTAEDecoder> load(const std::string& path);
    virtual ~CoreMLTAEDecoder() = default;

    // latent is [N, C, h, w] and image is [N, 3, 8h, 8w], both contiguous float32. Both shapes must match
    // the model's, so the decoder never writes more than image_shape holds.
    virtual bool decode(const float* latent, const int64_t latent_shape[4], float* image, const int64_t image_shape[4]) = 0;
};

#endif  // __COREML_TAE_H__
