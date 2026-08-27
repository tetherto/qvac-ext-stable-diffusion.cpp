#include "fit_params.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

#include "core/util.h"
#include "ggml-backend.h"

namespace sd::fit_params {
    namespace {

        constexpr int64_t MiB = 1024ll * 1024;
        constexpr int64_t GiB = 1024ll * MiB;
        constexpr int64_t MEMORY_RESERVE = 512 * MiB;

        struct Device {
            ggml_backend_dev_t dev = nullptr;
            std::string name;
            std::string description;
            int64_t free_bytes   = 0;
            int64_t total_bytes  = 0;
            int64_t budget_bytes = 0;
            bool graph_budget_enabled = false;
        };

        struct Decision {
            bool placed      = false;
            bool on_cpu      = false;
            bool disk_params = false;
            bool cpu_params  = false;
            bool tiled       = false;
            bool stream_layers = false;
            std::vector<size_t> device_idxs;
        };

        struct ComputePhases {
            int64_t serial      = 0;
            int64_t diffusion   = 0;
            int64_t control_net = 0;

            void add(SDBackendModule module, int64_t bytes) {
                if (module == SDBackendModule::DIFFUSION) {
                    diffusion = std::max(diffusion, bytes);
                } else if (module == SDBackendModule::CONTROL_NET) {
                    control_net = std::max(control_net, bytes);
                } else {
                    serial = std::max(serial, bytes);
                }
            }

            int64_t peak() const {
                return std::max(serial, diffusion + control_net);
            }
        };

        int64_t available_host_memory() {
            const char* debug_gib = getenv("SD_FIT_DEBUG_HOST_MEMORY_GIB");
            if (debug_gib != nullptr && debug_gib[0] != '\0') {
                return std::max<int64_t>((int64_t)(std::strtod(debug_gib, nullptr) * GiB), 0);
            }
#if defined(_WIN32)
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            return GlobalMemoryStatusEx(&status) ? (int64_t)status.ullAvailPhys : -1;
#elif defined(__APPLE__)
            vm_statistics64_data_t stats{};
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            vm_size_t page_size          = 0;
            if (host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS ||
                host_statistics64(mach_host_self(), HOST_VM_INFO64,
                                  reinterpret_cast<host_info64_t>(&stats), &count) != KERN_SUCCESS) {
                return -1;
            }
            return (int64_t)(stats.free_count + stats.inactive_count + stats.speculative_count) *
                   (int64_t)page_size;
#elif defined(__linux__)
            std::ifstream meminfo("/proc/meminfo");
            std::string key;
            int64_t value = 0;
            std::string unit;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemAvailable:") {
                    return value * 1024;
                }
            }
            struct sysinfo info {};
            if (sysinfo(&info) == 0) {
                return (int64_t)(info.freeram + info.bufferram) * (int64_t)info.mem_unit;
            }
            return -1;
#else
            ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu == nullptr) {
                return -1;
            }
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_dev_memory(cpu, &free_bytes, &total_bytes);
            return free_bytes > 0 ? (int64_t)free_bytes : -1;
#endif
        }

        void apply_device_budget(Device& d, sd::ggml_graph_cut::MaxVramAssignment& budgets) {
            float gib = budgets.default_gib;
            {
                std::string budget_key = d.name;
                std::transform(budget_key.begin(), budget_key.end(), budget_key.begin(), [](unsigned char c) { return (char)std::tolower(c); });
                auto it = budgets.backend_gib.find(budget_key);
                if (it != budgets.backend_gib.end()) {
                    gib = it->second;
                }
            }
            if (gib > 0.f) {
                d.budget_bytes = std::min<int64_t>((int64_t)(gib * GiB), d.free_bytes) - MEMORY_RESERVE;
                d.graph_budget_enabled = true;
            } else if (gib < 0.f) {
                d.budget_bytes = d.free_bytes + (int64_t)(gib * GiB) - MEMORY_RESERVE;
                d.graph_budget_enabled = true;
            } else {
                d.budget_bytes = d.free_bytes - MEMORY_RESERVE;
                d.graph_budget_enabled = false;
            }
            d.budget_bytes = std::max<int64_t>(d.budget_bytes, 0);
        }

        // debug override to exercise multi-device planning on any machine,
        // e.g. SD_FIT_DEBUG_DEVICES="CUDA0:24,CUDA1:16" (name:free_gib)
        std::vector<Device> simulated_devices(const char* spec, sd::ggml_graph_cut::MaxVramAssignment& budgets) {
            std::vector<Device> out;
            std::string s = spec;
            size_t pos    = 0;
            while (pos < s.size()) {
                size_t comma      = s.find(',', pos);
                std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                pos               = comma == std::string::npos ? s.size() : comma + 1;
                size_t colon      = entry.find(':');
                if (colon == std::string::npos) {
                    continue;
                }
                Device d;
                d.name        = entry.substr(0, colon);
                d.description = "simulated device";
                d.free_bytes  = (int64_t)(std::stof(entry.substr(colon + 1)) * 1024.0 * 1024.0 * 1024.0);
                d.total_bytes = d.free_bytes;
                apply_device_budget(d, budgets);
                out.push_back(d);
            }
            return out;
        }

        std::vector<Device> enumerate_gpu_devices(sd::ggml_graph_cut::MaxVramAssignment& budgets) {
            const char* debug_devices = getenv("SD_FIT_DEBUG_DEVICES");
            if (debug_devices != nullptr && debug_devices[0] != '\0') {
                LOG_WARN("fit-params: planning against simulated devices (SD_FIT_DEBUG_DEVICES)");
                return simulated_devices(debug_devices, budgets);
            }

            std::vector<Device> out;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                    continue;
                }
                Device d;
                d.dev             = dev;
                d.name            = ggml_backend_dev_name(dev);
                d.description     = ggml_backend_dev_description(dev);
                size_t free_bytes = 0, total_bytes = 0;
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
                d.free_bytes  = (int64_t)free_bytes;
                d.total_bytes = (int64_t)total_bytes;
                apply_device_budget(d, budgets);
                out.push_back(d);
            }
            return out;
        }

        std::string module_spec_key(SDBackendModule module) {
            // sd_backend_module_name returns tokens sd_parse_backend_assignment accepts
            return sd_backend_module_name(module);
        }

        void append_assignment(std::string& spec, const std::string& key, const std::string& value) {
            if (!spec.empty()) {
                spec += ",";
            }
            spec += key;
            spec += "=";
            spec += value;
        }

        void report_line(std::string& report, const char* fmt, ...) {
            char line[512];
            va_list args;
            va_start(args, fmt);
            vsnprintf(line, sizeof(line), fmt, args);
            va_end(args);
            LOG_INFO("%s", line);
            report += line;
            report += "\n";
        }

        int64_t host_memory_requirement(const std::vector<ModuleMemory>& modules,
                                        const std::vector<Decision>* decisions) {
            int64_t params = 0;
            ComputePhases compute;
            for (size_t i = 0; i < modules.size(); ++i) {
                const bool on_cpu = decisions == nullptr ||
                                    (*decisions)[i].on_cpu ||
                                    (*decisions)[i].cpu_params;
                if (!on_cpu) {
                    continue;
                }
                params += (int64_t)modules[i].params_bytes;
                if (decisions == nullptr || (*decisions)[i].on_cpu) {
                    compute.add(modules[i].module, (int64_t)modules[i].compute_bytes);
                }
            }
            return params + compute.peak();
        }

        bool host_memory_fits(const std::vector<ModuleMemory>& modules,
                              const std::vector<Decision>* decisions,
                              std::string& report) {
            const int64_t required = host_memory_requirement(modules, decisions);
            if (required == 0) {
                return true;
            }
            const int64_t available = available_host_memory();
            const int64_t budget    = available < 0 ? -1 : std::max<int64_t>(available - MEMORY_RESERVE, 0);
            if (budget < 0) {
                report_line(report, "  host memory availability could not be determined; refusing an unverified CPU placement");
                return false;
            }
            report_line(report, "  host memory: available %lld MiB, budget %lld MiB, projected use %lld MiB",
                        (long long)(available / MiB),
                        (long long)(budget / MiB),
                        (long long)(required / MiB));
            return required <= budget;
        }

    }  // namespace

    bool plan_placement(const std::vector<ModuleMemory>& modules,
                        sd::ggml_graph_cut::MaxVramAssignment& budgets,
                        FitPlan* plan) {
        if (plan == nullptr) {
            return false;
        }
        *plan = {};

        {
            std::string error;
            if (!budgets.canonicalize_backend_keys(&error)) {
                LOG_ERROR("%s", error.c_str());
                return false;
            }
        }

        std::vector<Device> devices = enumerate_gpu_devices(budgets);

        report_line(plan->report, "fit-params: measured memory plan");
        report_line(plan->report, "  devices:");
        for (const Device& d : devices) {
            report_line(plan->report, "    %-12s %-32s free %6lld MiB, budget %6lld MiB",
                        d.name.c_str(), d.description.c_str(),
                        (long long)(d.free_bytes / MiB), (long long)(d.budget_bytes / MiB));
        }
        report_line(plan->report, "  modules (measured for this workload):");
        for (const ModuleMemory& m : modules) {
            if (m.params_bytes == 0 && m.compute_bytes == 0) {
                continue;
            }
            if (m.compute_bytes_tiled > 0) {
                report_line(plan->report, "    %-12s params %6lld MiB, compute %6lld MiB (tiled %6lld MiB)",
                            module_spec_key(m.module).c_str(),
                            (long long)(m.params_bytes / MiB),
                            (long long)(m.compute_bytes / MiB),
                            (long long)(m.compute_bytes_tiled / MiB));
            } else {
                report_line(plan->report, "    %-12s params %6lld MiB, compute %6lld MiB",
                            module_spec_key(m.module).c_str(),
                            (long long)(m.params_bytes / MiB),
                            (long long)(m.compute_bytes / MiB));
            }
        }

        if (devices.empty()) {
            report_line(plan->report, "  no usable GPU devices; checking the default CPU backend");
            plan->valid   = host_memory_fits(modules, nullptr, plan->report);
            plan->changed = false;
            return true;
        }

        // check-first: the default placement puts every module on the default (first GPU) device
        {
            int64_t params_sum = 0;
            ComputePhases compute;
            for (const ModuleMemory& m : modules) {
                params_sum += (int64_t)m.params_bytes;
                compute.add(m.module, (int64_t)m.compute_bytes);
            }
            if (params_sum + compute.peak() <= devices[0].budget_bytes) {
                report_line(plan->report, "  projected use %lld MiB <= budget %lld MiB on %s, no changes needed",
                            (long long)((params_sum + compute.peak()) / MiB),
                            (long long)(devices[0].budget_bytes / MiB),
                            devices[0].name.c_str());
                plan->valid   = true;
                plan->changed = false;
                return true;
            }
        }

        std::vector<size_t> order(modules.size());
        for (size_t i = 0; i < order.size(); i++) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return modules[a].params_bytes > modules[b].params_bytes;
        });

        std::vector<Decision> decisions(modules.size());
        bool time_share = false;

        // resident plan: every module keeps its params loaded, compute buffers coexist per device
        {
            std::vector<int64_t> params_sum(devices.size(), 0);
            std::vector<ComputePhases> compute_phases(devices.size());
            bool ok         = true;
            bool vae_tiling = false;
            std::vector<Decision> resident(modules.size());
            auto find_device = [&](const ModuleMemory& m, int64_t compute) -> int {
                int best = -1;
                for (size_t di = 0; di < devices.size(); di++) {
                    ComputePhases candidate = compute_phases[di];
                    candidate.add(m.module, compute);
                    int64_t need = params_sum[di] + (int64_t)m.params_bytes +
                                   candidate.peak();
                    if (need <= devices[di].budget_bytes &&
                        (best < 0 || devices[di].budget_bytes - params_sum[di] > devices[best].budget_bytes - params_sum[best])) {
                        best = (int)di;
                    }
                }
                return best;
            };
            for (size_t mi : order) {
                const ModuleMemory& m = modules[mi];
                if (m.params_bytes == 0 && m.compute_bytes == 0) {
                    resident[mi].placed = true;
                    continue;
                }
                int64_t compute = (int64_t)m.compute_bytes;
                int best        = find_device(m, compute);
                if (best < 0 && m.compute_bytes_tiled > 0 && m.compute_bytes_tiled < m.compute_bytes) {
                    // full-resolution decode does not fit anywhere, tiling may keep the module resident
                    compute = (int64_t)m.compute_bytes_tiled;
                    best    = find_device(m, compute);
                    if (best >= 0) {
                        resident[mi].tiled = true;
                        vae_tiling         = true;
                    }
                }
                if (best < 0) {
                    ok = false;
                    break;
                }
                params_sum[best] += (int64_t)m.params_bytes;
                compute_phases[best].add(m.module, compute);
                resident[mi].placed = true;
                resident[mi].device_idxs.push_back((size_t)best);
            }
            if (ok) {
                decisions        = std::move(resident);
                plan->vae_tiling = vae_tiling;
            } else {
                time_share = true;
            }
        }

        // time-share plan: phases run sequentially, heavy modules load per phase and free after
        if (time_share) {
            std::vector<ComputePhases> compute_phases(devices.size());
            auto compute_with_concurrent_phase = [&](size_t device_idx,
                                                     SDBackendModule module,
                                                     int64_t compute) {
                if (module == SDBackendModule::DIFFUSION) {
                    return compute + compute_phases[device_idx].control_net;
                }
                if (module == SDBackendModule::CONTROL_NET) {
                    return compute + compute_phases[device_idx].diffusion;
                }
                return compute;
            };
            auto concurrent_compute_extra = [&](size_t device_idx,
                                                SDBackendModule module) {
                if (module == SDBackendModule::DIFFUSION) {
                    return compute_phases[device_idx].control_net;
                }
                if (module == SDBackendModule::CONTROL_NET) {
                    return compute_phases[device_idx].diffusion;
                }
                return int64_t{0};
            };
            auto split_graphs_fit = [&](const ModuleMemory& m,
                                        const std::vector<size_t>& device_idxs,
                                        int64_t compute) {
                if (m.split_graph_segment_params.empty()) {
                    return false;
                }
                std::vector<int64_t> capacities;
                capacities.reserve(device_idxs.size());
                for (size_t device_idx : device_idxs) {
                    capacities.push_back(std::max<int64_t>(devices[device_idx].budget_bytes - compute, 0));
                }
                for (const auto& graph_segments : m.split_graph_segment_params) {
                    size_t device_pos = 0;
                    int64_t used      = 0;
                    for (size_t segment_bytes : graph_segments) {
                        while (device_pos + 1 < capacities.size() &&
                               used + (int64_t)segment_bytes > capacities[device_pos]) {
                            ++device_pos;
                            used = 0;
                        }
                        if (used + (int64_t)segment_bytes > capacities[device_pos]) {
                            return false;
                        }
                        used += (int64_t)segment_bytes;
                    }
                }
                return true;
            };
            auto streamed_graphs_fit = [&](const ModuleMemory& m, int64_t budget) {
                if (m.split_graph_segment_params.empty() ||
                    m.split_graph_segment_params.size() != m.split_graph_segment_compute.size()) {
                    return false;
                }
                for (size_t graph_idx = 0; graph_idx < m.split_graph_segment_params.size(); ++graph_idx) {
                    const auto& params  = m.split_graph_segment_params[graph_idx];
                    const auto& compute = m.split_graph_segment_compute[graph_idx];
                    if (params.size() != compute.size()) {
                        return false;
                    }
                    for (size_t segment_idx = 0; segment_idx < params.size(); ++segment_idx) {
                        if ((int64_t)params[segment_idx] + (int64_t)compute[segment_idx] > budget) {
                            return false;
                        }
                    }
                }
                return true;
            };
            for (size_t mi : order) {
                const ModuleMemory& m = modules[mi];
                Decision& decision    = decisions[mi];
                decision              = {};
                if (m.params_bytes == 0 && m.compute_bytes == 0) {
                    decision.placed = true;
                    continue;
                }
                int best = -1;
                for (size_t di = 0; di < devices.size(); di++) {
                    const int64_t compute = compute_with_concurrent_phase(di, m.module, (int64_t)m.compute_bytes);
                    if ((int64_t)m.params_bytes + compute <= devices[di].budget_bytes &&
                        (best < 0 || devices[di].budget_bytes > devices[best].budget_bytes)) {
                        best = (int)di;
                    }
                }
                if (best >= 0) {
                    decision.placed      = true;
                    decision.disk_params = true;
                    decision.device_idxs.push_back((size_t)best);
                    compute_phases[best].add(m.module, (int64_t)m.compute_bytes);
                    continue;
                }
                if (m.compute_bytes_tiled > 0) {
                    for (size_t di = 0; di < devices.size(); di++) {
                        const int64_t compute = compute_with_concurrent_phase(di, m.module, (int64_t)m.compute_bytes_tiled);
                        if ((int64_t)m.params_bytes + compute <= devices[di].budget_bytes &&
                            (best < 0 || devices[di].budget_bytes > devices[best].budget_bytes)) {
                            best = (int)di;
                        }
                    }
                    if (best >= 0) {
                        decision.placed      = true;
                        decision.disk_params = true;
                        decision.tiled       = true;
                        plan->vae_tiling     = true;
                        decision.device_idxs.push_back((size_t)best);
                        compute_phases[best].add(m.module, (int64_t)m.compute_bytes_tiled);
                        continue;
                    }
                }
                if (m.splittable && devices.size() > 1) {
                    int64_t capacity = 0;
                    std::vector<size_t> idxs(devices.size());
                    for (size_t i = 0; i < idxs.size(); i++) {
                        idxs[i] = i;
                    }
                    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
                        return devices[a].budget_bytes > devices[b].budget_bytes;
                    });
                    for (size_t di = 0; di < devices.size(); ++di) {
                        const int64_t compute = compute_with_concurrent_phase(di, m.module, (int64_t)m.compute_bytes);
                        capacity += std::max<int64_t>(devices[di].budget_bytes - compute, 0);
                    }
                    int64_t split_compute = 0;
                    for (size_t di : idxs) {
                        split_compute = std::max(split_compute,
                                                 compute_with_concurrent_phase(di, m.module, (int64_t)m.compute_bytes));
                    }
                    if ((int64_t)m.params_bytes <= capacity && split_graphs_fit(m, idxs, split_compute)) {
                        decision.placed      = true;
                        decision.disk_params = true;
                        decision.device_idxs = std::move(idxs);
                        for (size_t di : decision.device_idxs) {
                            compute_phases[di].add(m.module, (int64_t)m.compute_bytes);
                        }
                        continue;
                    }
                }
                if (m.module == SDBackendModule::DIFFUSION && m.splittable) {
                    for (size_t di = 0; di < devices.size(); di++) {
                        if (devices[di].graph_budget_enabled && devices[di].budget_bytes > 0 &&
                            streamed_graphs_fit(m,
                                                devices[di].budget_bytes -
                                                    concurrent_compute_extra(di, m.module)) &&
                            (best < 0 || devices[di].budget_bytes > devices[best].budget_bytes)) {
                            best = (int)di;
                        }
                    }
                    if (best >= 0) {
                        decision.placed        = true;
                        decision.cpu_params    = true;
                        decision.stream_layers = true;
                        plan->stream_layers    = true;
                        decision.device_idxs.push_back((size_t)best);
                        compute_phases[best].add(m.module, (int64_t)m.compute_bytes);
                        continue;
                    }
                }
                decision.placed = true;
                decision.on_cpu = true;
            }
        }

        if (!host_memory_fits(modules, &decisions, plan->report)) {
            report_line(plan->report, "  no placement fits available host memory");
            plan->valid = false;
            return true;
        }

        report_line(plan->report, "  placement%s:", time_share ? " (time-share: params load per phase and free after)" : "");
        for (size_t mi = 0; mi < modules.size(); mi++) {
            const ModuleMemory& m    = modules[mi];
            const Decision& decision = decisions[mi];
            if (m.params_bytes == 0 && m.compute_bytes == 0) {
                continue;
            }
            std::string target;
            if (decision.on_cpu) {
                target = "cpu";
            } else {
                for (size_t k = 0; k < decision.device_idxs.size(); k++) {
                    if (k > 0) {
                        target += " & ";
                    }
                    target += devices[decision.device_idxs[k]].name;
                }
                if (decision.device_idxs.size() > 1) {
                    target += " (split)";
                }
            }
            report_line(plan->report, "    %-12s -> %s%s%s%s",
                        module_spec_key(m.module).c_str(),
                        target.c_str(),
                        decision.disk_params ? ", params on disk" : "",
                        decision.cpu_params ? ", params on cpu, stream layers" : "",
                        decision.tiled ? ", vae tiling" : "");
        }

        for (size_t mi = 0; mi < modules.size(); mi++) {
            const ModuleMemory& m    = modules[mi];
            const Decision& decision = decisions[mi];
            if (m.params_bytes == 0 && m.compute_bytes == 0) {
                continue;
            }
            const std::string key = module_spec_key(m.module);
            if (decision.on_cpu) {
                append_assignment(plan->runtime_spec, key, "cpu");
                continue;
            }
            if (decision.device_idxs.empty()) {
                continue;
            }
            std::string device_list;
            for (size_t k = 0; k < decision.device_idxs.size(); k++) {
                if (k > 0) {
                    device_list += "&";
                }
                device_list += devices[decision.device_idxs[k]].name;
            }
            append_assignment(plan->runtime_spec, key, device_list);
            if (decision.disk_params) {
                append_assignment(plan->params_spec, key, "disk");
            } else if (decision.cpu_params) {
                append_assignment(plan->params_spec, key, "cpu");
            }
        }

        plan->valid      = true;
        plan->changed    = true;
        plan->time_share = time_share;
        return true;
    }

}  // namespace sd::fit_params
