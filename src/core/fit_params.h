#ifndef __SD_FIT_PARAMS_H__
#define __SD_FIT_PARAMS_H__

#include <string>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/ggml_graph_cut.h"

namespace sd::fit_params {

    // measured memory requirements for one module at the requested workload
    struct ModuleMemory {
        SDBackendModule module;
        size_t params_bytes        = 0;  // weights registered for the module
        size_t compute_bytes       = 0;  // largest measured compute buffer among the module's graphs
        size_t compute_bytes_tiled = 0;  // VAE only: compute buffer with tiling enabled, 0 if not measured
        bool splittable            = false;
    };

    struct FitPlan {
        bool valid      = false;
        bool changed    = false;  // false = current/default placement already fits
        bool time_share = false;
        bool vae_tiling = false;
        bool stream_layers = false;
        std::string runtime_spec;
        std::string params_spec;
        std::string report;  // human readable per-device / per-module table
    };

    // derive placement specs from measured module memory and per-device budgets
    bool plan_placement(const std::vector<ModuleMemory>& modules,
                        sd::ggml_graph_cut::MaxVramAssignment& budgets,
                        FitPlan* plan);

}  // namespace sd::fit_params

#endif  // __SD_FIT_PARAMS_H__
