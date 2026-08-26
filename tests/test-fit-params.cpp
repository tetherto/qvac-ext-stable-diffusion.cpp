#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "core/fit_params.h"

namespace {

constexpr size_t GiB = 1024ull * 1024ull * 1024ull;

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << std::endl;
        return false;
    }
    return true;
}

sd::fit_params::ModuleMemory module(SDBackendModule module,
                                    size_t params_gib,
                                    size_t compute_gib,
                                    bool splittable = false,
                                    size_t tiled_compute_mib = 0) {
    sd::fit_params::ModuleMemory memory;
    memory.module              = module;
    memory.params_bytes        = params_gib * GiB;
    memory.compute_bytes       = compute_gib * GiB;
    memory.splittable          = splittable;
    memory.compute_bytes_tiled = tiled_compute_mib * 1024ull * 1024ull;
    return memory;
}

bool plan_with_devices(const char* devices,
                       float max_vram_gib,
                       const std::vector<sd::fit_params::ModuleMemory>& modules,
                       sd::fit_params::FitPlan* plan) {
    setenv("SD_FIT_DEBUG_DEVICES", devices, 1);
    sd::ggml_graph_cut::MaxVramAssignment budgets;
    budgets.reset(max_vram_gib);
    return sd::fit_params::plan_placement(modules, budgets, plan);
}

bool test_default_fits() {
    sd::fit_params::FitPlan plan;
    bool ok = plan_with_devices("GPU0:8", 8.f,
                                {module(SDBackendModule::DIFFUSION, 1, 1, true),
                                 module(SDBackendModule::TE, 1, 1, true),
                                 module(SDBackendModule::VAE, 1, 1)},
                                &plan);
    return expect(ok && plan.valid, "default fit plan should be valid") &&
           expect(!plan.changed, "default fit should not emit placement changes") &&
           expect(plan.runtime_spec.empty(), "default fit runtime spec should be empty") &&
           expect(plan.params_spec.empty(), "default fit params spec should be empty");
}

bool test_resident_spread() {
    sd::fit_params::FitPlan plan;
    bool ok = plan_with_devices("GPU0:8,GPU1:8", 8.f,
                                {module(SDBackendModule::DIFFUSION, 5, 2, true),
                                 module(SDBackendModule::TE, 3, 1, true),
                                 module(SDBackendModule::VAE, 2, 3)},
                                &plan);
    return expect(ok && plan.valid, "resident spread plan should be valid") &&
           expect(plan.changed, "resident spread should emit placement changes") &&
           expect(!plan.time_share, "resident spread should not time-share params") &&
           expect(plan.runtime_spec == "diffusion=GPU0,te=GPU1,vae=GPU1",
                  ("unexpected resident runtime spec: " + plan.runtime_spec).c_str()) &&
           expect(plan.params_spec.empty(), "resident spread should not place params on disk");
}

bool test_time_share_cpu_fallback() {
    sd::fit_params::FitPlan plan;
    bool ok = plan_with_devices("GPU0:6", 0.f,
                                {module(SDBackendModule::DIFFUSION, 5, 2, true),
                                 module(SDBackendModule::TE, 1, 1, true),
                                 module(SDBackendModule::VAE, 1, 2)},
                                &plan);
    return expect(ok && plan.valid, "time-share plan should be valid") &&
           expect(plan.time_share, "oversized resident plan should time-share") &&
           expect(plan.runtime_spec == "diffusion=cpu,te=GPU0,vae=GPU0",
                  ("unexpected time-share runtime spec: " + plan.runtime_spec).c_str()) &&
           expect(plan.params_spec == "te=disk,vae=disk",
                  ("unexpected time-share params spec: " + plan.params_spec).c_str());
}

bool test_stream_layers_after_split_fails() {
    sd::fit_params::FitPlan plan;
    bool ok = plan_with_devices("GPU0:6,GPU1:6", 6.f,
                                {module(SDBackendModule::DIFFUSION, 20, 2, true)},
                                &plan);
    return expect(ok && plan.valid, "streaming plan should be valid") &&
           expect(plan.time_share, "streaming plan should be a time-share fallback") &&
           expect(plan.stream_layers, "streaming plan should request stream layers") &&
           expect(plan.runtime_spec == "diffusion=GPU0",
                  ("unexpected streaming runtime spec: " + plan.runtime_spec).c_str()) &&
           expect(plan.params_spec == "diffusion=cpu",
                  ("unexpected streaming params spec: " + plan.params_spec).c_str());
}

bool test_split_and_tiling() {
    sd::fit_params::FitPlan split_plan;
    bool split_ok = plan_with_devices("GPU0:6,GPU1:6", 6.f,
                                      {module(SDBackendModule::DIFFUSION, 8, 2, true)},
                                      &split_plan);
    if (!expect(split_ok && split_plan.valid, "split plan should be valid") ||
        !expect(split_plan.runtime_spec == "diffusion=GPU0&GPU1",
                ("unexpected split runtime spec: " + split_plan.runtime_spec).c_str()) ||
        !expect(split_plan.params_spec == "diffusion=disk",
                ("unexpected split params spec: " + split_plan.params_spec).c_str())) {
        return false;
    }

    sd::fit_params::FitPlan tiling_plan;
    bool tiling_ok = plan_with_devices("GPU0:4", 4.f,
                                       {module(SDBackendModule::VAE, 1, 5, false, 512)},
                                       &tiling_plan);
    return expect(tiling_ok && tiling_plan.valid, "tiling plan should be valid") &&
           expect(tiling_plan.vae_tiling, "tiling plan should request VAE tiling") &&
           expect(tiling_plan.runtime_spec == "vae=GPU0",
                  ("unexpected tiling runtime spec: " + tiling_plan.runtime_spec).c_str()) &&
           expect(tiling_plan.params_spec.empty(),
                  ("unexpected tiling params spec: " + tiling_plan.params_spec).c_str());
}

}  // namespace

int main() {
    if (!test_default_fits() ||
        !test_resident_spread() ||
        !test_time_share_cpu_fallback() ||
        !test_stream_layers_after_split_fails() ||
        !test_split_and_tiling()) {
        return 1;
    }
    unsetenv("SD_FIT_DEBUG_DEVICES");
    return 0;
}
