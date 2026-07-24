#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "lam_wav2vec_frontend.hpp"
#include "model.h"

namespace {

bool read_f32_file(const char* path, std::vector<float>* values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        return false;
    }
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        return false;
    }
    values->resize(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values->data()), bytes);
    return static_cast<bool>(input);
}

bool write_f32_file(const char* path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(output);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr,
                     "usage: lam-a2e-frontend-smoke <model.gguf> <pcm.f32> <output.f32> [frontend|interpolated|norm|projection|position]\n");
        return 2;
    }

    std::vector<float> pcm;
    if (!read_f32_file(argv[2], &pcm)) {
        std::fprintf(stderr, "failed to read PCM fixture\n");
        return 1;
    }

    ModelLoader loader;
    if (!loader.init_from_file(argv[1])) {
        std::fprintf(stderr, "failed to read GGUF metadata\n");
        return 1;
    }
    loader.process_model_files(false);

    ggml_backend_t backend = ggml_backend_cpu_init();
    LamWav2VecFrontend frontend(backend, backend, loader.get_tensor_storage_map());
    if (!frontend.load(loader, 4)) {
        std::fprintf(stderr, "failed to load frontend parameters\n");
        ggml_backend_free(backend);
        return 1;
    }

    const int64_t sample_count = static_cast<int64_t>(pcm.size());
    sd::Tensor<float> input({sample_count, 1, 1}, std::move(pcm));
    const std::string stage = argc == 5 ? argv[4] : "frontend";
    const auto output = stage == "projection"
                            ? frontend.compute_projection(input, 4)
                            : stage == "position"
                                  ? frontend.compute_position_cpu_reference(input, 4)
                            : stage == "norm"
                                  ? frontend.compute_normalized(input, 4)
                                  : stage == "interpolated"
                                        ? frontend.compute_interpolated(input, 4)
                                        : frontend.compute_frontend(input, 4);
    if (!write_f32_file(argv[3], output.values())) {
        std::fprintf(stderr, "failed to write frontend output\n");
        ggml_backend_free(backend);
        return 1;
    }

    const auto& shape = output.shape();
    std::printf("frontend output shape:");
    for (const auto dim : shape) {
        std::printf(" %lld", static_cast<long long>(dim));
    }
    std::printf("\n");
    ggml_backend_free(backend);
    return 0;
}
