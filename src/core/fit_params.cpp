#include "fit_params.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>

#include "core/util.h"
#include "ggml-backend.h"

namespace sd::fit_params {
    namespace {

        constexpr int64_t MiB = 1024ll * 1024;

        struct Device {
            ggml_backend_dev_t dev = nullptr;
            std::string name;
            std::string description;
            int64_t free_bytes   = 0;
            int64_t total_bytes  = 0;
            int64_t budget_bytes = 0;
        };

        struct Decision {
            bool placed      = false;
            bool on_cpu      = false;
            bool disk_params = false;
            bool tiled       = false;
            std::vector<size_t> device_idxs;
        };

        std::vector<Device> enumerate_gpu_devices(sd::ggml_graph_cut::MaxVramAssignment& budgets) {
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

                ggml_backend_t backend = nullptr;  // bytes_for_backend needs a backend, resolve via name instead
                (void)backend;
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
                    d.budget_bytes = std::min<int64_t>((int64_t)(gib * 1024.0 * 1024.0 * 1024.0), d.free_bytes);
                } else if (gib < 0.f) {
                    d.budget_bytes = d.free_bytes + (int64_t)(gib * 1024.0 * 1024.0 * 1024.0);
                } else {
                    d.budget_bytes = d.free_bytes - 512 * MiB;
                }
                d.budget_bytes = std::max<int64_t>(d.budget_bytes, 0);
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
            report_line(plan->report, "  no usable GPU devices; keeping the default backend");
            plan->valid   = true;
            plan->changed = false;
            return true;
        }

        // check-first: the default placement puts every module on the default (first GPU) device
        {
            int64_t params_sum  = 0;
            int64_t compute_max = 0;
            for (const ModuleMemory& m : modules) {
                params_sum += (int64_t)m.params_bytes;
                compute_max = std::max<int64_t>(compute_max, (int64_t)m.compute_bytes);
            }
            if (params_sum + compute_max <= devices[0].budget_bytes) {
                report_line(plan->report, "  projected use %lld MiB <= budget %lld MiB on %s, no changes needed",
                            (long long)((params_sum + compute_max) / MiB),
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
            std::vector<int64_t> max_compute(devices.size(), 0);
            bool ok = true;
            std::vector<Decision> resident(modules.size());
            for (size_t mi : order) {
                const ModuleMemory& m = modules[mi];
                if (m.params_bytes == 0 && m.compute_bytes == 0) {
                    resident[mi].placed = true;
                    continue;
                }
                int best = -1;
                for (size_t di = 0; di < devices.size(); di++) {
                    int64_t need = params_sum[di] + (int64_t)m.params_bytes +
                                   std::max<int64_t>(max_compute[di], (int64_t)m.compute_bytes);
                    if (need <= devices[di].budget_bytes &&
                        (best < 0 || devices[di].budget_bytes - params_sum[di] > devices[best].budget_bytes - params_sum[best])) {
                        best = (int)di;
                    }
                }
                if (best < 0) {
                    ok = false;
                    break;
                }
                params_sum[best] += (int64_t)m.params_bytes;
                max_compute[best] = std::max<int64_t>(max_compute[best], (int64_t)m.compute_bytes);
                resident[mi].placed = true;
                resident[mi].device_idxs.push_back((size_t)best);
            }
            if (ok) {
                decisions = std::move(resident);
            } else {
                time_share = true;
            }
        }

        // time-share plan: phases run sequentially, heavy modules load per phase and free after
        if (time_share) {
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
                    if ((int64_t)m.params_bytes + (int64_t)m.compute_bytes <= devices[di].budget_bytes &&
                        (best < 0 || devices[di].budget_bytes > devices[best].budget_bytes)) {
                        best = (int)di;
                    }
                }
                if (best >= 0) {
                    decision.placed      = true;
                    decision.disk_params = true;
                    decision.device_idxs.push_back((size_t)best);
                    continue;
                }
                if (m.compute_bytes_tiled > 0) {
                    for (size_t di = 0; di < devices.size(); di++) {
                        if ((int64_t)m.params_bytes + (int64_t)m.compute_bytes_tiled <= devices[di].budget_bytes &&
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
                        continue;
                    }
                }
                if (m.splittable && devices.size() > 1) {
                    int64_t capacity = 0;
                    for (const Device& d : devices) {
                        capacity += std::max<int64_t>(d.budget_bytes - (int64_t)m.compute_bytes, 0);
                    }
                    if ((int64_t)m.params_bytes <= capacity) {
                        decision.placed      = true;
                        decision.disk_params = true;
                        std::vector<size_t> idxs(devices.size());
                        for (size_t i = 0; i < idxs.size(); i++) {
                            idxs[i] = i;
                        }
                        std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
                            return devices[a].budget_bytes > devices[b].budget_bytes;
                        });
                        decision.device_idxs = std::move(idxs);
                        continue;
                    }
                }
                decision.placed = true;
                decision.on_cpu = true;
            }
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
            report_line(plan->report, "    %-12s -> %s%s%s",
                        module_spec_key(m.module).c_str(),
                        target.c_str(),
                        decision.disk_params ? ", params on disk" : "",
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
            }
        }

        plan->valid      = true;
        plan->changed    = true;
        plan->time_share = time_share;
        return true;
    }

}  // namespace sd::fit_params
