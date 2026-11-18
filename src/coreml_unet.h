#ifndef __COREML_UNET_H__
#define __COREML_UNET_H__

#include <cstdint>
#include <memory>
#include <string>

// Runs a UNet exported to CoreML (see script/coreml/convert_unet.py) in place of the ggml UNet.
// A multifunction model (iOS 18+) holds one fixed-shape function per resolution; each call uses
// the function whose input shapes match, loading it on first use.
// Expected model I/O (fixed shapes; float32 or float16 inputs):
//   sample                [N, C, H, W]
//   timestep              [N]
//   encoder_hidden_states [N, T, D], or [N, D, 1, T] as Apple's ml-stable-diffusion exports it
//   noise_pred            [N, C, H, W]
// Compute units come from SD_COREML_COMPUTE_UNITS: all | cpu_ne (default) | cpu_gpu | cpu.
class CoreMLUNet {
public:
    static std::unique_ptr<CoreMLUNet> load(const std::string& path);
    virtual ~CoreMLUNet() = default;

    // x and out are [N, C, H, W], context is [N, T, D], all contiguous float32.
    virtual bool predict(const float* x,
                         const int64_t x_shape[4],
                         const float* timesteps,
                         const float* context,
                         const int64_t context_shape[3],
                         float* out) = 0;

    // predict() for any batch N: a batch-2 model takes N = 1 by duplicating the input, and a model
    // exported for batch 1 runs N > 1 one item at a time. timesteps has N values, or 1 shared by all.
    bool run(const float* x,
             const int64_t x_shape[4],
             const float* timesteps,
             int64_t n_timesteps,
             const float* context,
             const int64_t context_shape[3],
             float* out);

    virtual std::string get_desc() = 0;

    // "epsilon" or "v_prediction" from the model's metadata (written by convert_unet.py), or "" if absent.
    virtual std::string prediction_type() = 0;

    // The batch the model was exported for: 1, or 2 for models that run CFG's two passes in one call
    // (such as Apple's ml-stable-diffusion exports).
    virtual int64_t batch_size() = 0;
};

#endif  // __COREML_UNET_H__
