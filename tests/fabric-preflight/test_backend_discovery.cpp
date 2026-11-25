#include <cstdio>
#include <cstring>
#include "ggml-backend.h"

static bool has_backend(const char* name) {
    auto reg = ggml_backend_reg_by_name(name);
    return reg && ggml_backend_reg_dev_count(reg) > 0;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: test_backend_discovery MODULE_DIRECTORY CPU|CUDA|MTL\n");
        return 1;
    }
    ggml_backend_load_all_from_path(argv[1]);
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::fprintf(stderr, "CPU initialization failed\n");
        return 1;
    }
    ggml_backend_free(cpu);
    if (!has_backend(argv[2])) {
        std::fprintf(stderr, "Required backend %s is unavailable\n", argv[2]);
        return 1;
    }
    std::printf("CPU initialization and %s discovery passed\n", argv[2]);
    return 0;
}
