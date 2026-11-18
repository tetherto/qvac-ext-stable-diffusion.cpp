#ifndef __SD_CORE_PROBE_CONV_DIRECT_H__
#define __SD_CORE_PROBE_CONV_DIRECT_H__

#include "ggml-backend.h"

enum class ConvDirectProbeProfile {
    Default,
    Flux2MetalVAE,
};

bool sd_probe_conv_direct(ggml_backend_t backend, bool conv3d, ConvDirectProbeProfile profile = ConvDirectProbeProfile::Default);

#endif  // __SD_CORE_PROBE_CONV_DIRECT_H__
