#pragma once

// LAM Audio2Expression on ggml — audio (16 kHz f32 PCM) → ARKit-52
// blendshape coefficients at 30 fps.
//
// Ported from packages/lipsync-ggml into qvac-ext-stable-diffusion.cpp as an
// isolated inference target (no sd_ctx_t / diffusion lifecycle).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

struct LamHParams {
    uint32_t sampleRate = 16000;
    uint32_t fps = 30;
    uint32_t nCoeffs = 52;
    uint32_t nIdentity = 12;
    uint32_t identityFeatDim = 64;
    uint32_t hiddenDim = 512;
    uint32_t windowFrames = 64;
    float layerNormEps = 1e-5F;
    uint32_t encLayers = 12;
    uint32_t encHeads = 12;
    uint32_t encHidden = 768;
    uint32_t encFfn = 3072;
    uint32_t posConvKernel = 128;
    uint32_t posConvGroups = 16;
    std::vector<int32_t> feKernels{10, 3, 3, 3, 3, 2, 2};
    std::vector<int32_t> feStrides{5, 2, 2, 2, 2, 2, 2};
    std::vector<std::string> coeffNames;
};

class LamAudio2Expression {
public:
    LamAudio2Expression() = default;
    ~LamAudio2Expression();

    LamAudio2Expression(const LamAudio2Expression&) = delete;
    LamAudio2Expression& operator=(const LamAudio2Expression&) = delete;
    LamAudio2Expression(LamAudio2Expression&&) = delete;
    LamAudio2Expression& operator=(LamAudio2Expression&&) = delete;

    bool load(const std::string& ggufPath, ggml_backend_t backend = nullptr, int n_threads = 0);

    [[nodiscard]] int64_t frameCount(int64_t nSamples) const;
    [[nodiscard]] int64_t convOutLen(int64_t nSamples) const;

    bool run(const std::vector<float>& pcm, uint32_t idIdx,
             std::vector<float>& framesOut,
             std::map<std::string, std::vector<float>>* taps = nullptr);

    [[nodiscard]] const LamHParams& hparams() const { return hparams_; }
    [[nodiscard]] const std::string& lastError() const { return lastError_; }

private:
    struct ggml_tensor* weight(const std::string& name);

    LamHParams hparams_;
    std::string lastError_;

    ggml_backend_t backend_ = nullptr;
    bool ownsBackend_ = false;
    ggml_backend_buffer_t weightBuffer_ = nullptr;
    struct ggml_context* weightCtx_ = nullptr;
    std::map<std::string, struct ggml_tensor*> weights_;
};
