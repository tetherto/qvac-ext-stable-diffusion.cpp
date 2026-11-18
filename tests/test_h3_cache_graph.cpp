#include "ggml-cpu.h"
#include "model/diffusion/h3_cache_graph.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cstdio>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        sd::Tensor<float> latent({2, 3}, {1, 2, 3, 4, 5, 6});
        sd::Tensor<float> context({2, 2}, {1, 2, 3, 4});
        sd::Tensor<float> timestep({1}, {900});
        MiniMaxH3DiffusionExtra extra;
        extra.audio_length = 2;
        DiffusionParams params;
        params.x            = &latent;
        params.context      = &context;
        params.timesteps    = &timestep;
        params.extra        = extra;
        const auto original = MiniMaxH3::cache_signature(params);
        latent.data()[0] += 1.f;
        timestep.data()[0] = 700.f;
        require(MiniMaxH3::cache_signature(params) == original, "Expected per-step latent/timestep change invalidated conditioning");
        latent = sd::Tensor<float>({3, 2}, {1, 2, 3, 4, 5, 6});
        require(MiniMaxH3::cache_signature(params) != original, "Same-numel shape change not detected");
        latent            = sd::Tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
        context.data()[0] = 9.f;
        require(MiniMaxH3::cache_signature(params) != original, "In-place condition change not detected");
        context.data()[0]       = 1.f;
        extra.audio_sigma_shift = 4.f;
        params.extra            = extra;
        require(MiniMaxH3::cache_signature(params) != original, "Audio schedule change not detected");
        extra.audio_sigma_shift = 3.f;
        params.extra            = extra;
        std::vector<sd::Tensor<float>> refs{sd::Tensor<float>({1}, {2.f})};
        params.ref_latents       = &refs;
        const auto ref_signature = MiniMaxH3::cache_signature(params);
        refs[0].data()[0]        = 3.f;
        require(MiniMaxH3::cache_signature(params) != ref_signature, "Reference data change not detected");

        const bool cuda        = argc == 2 && std::strcmp(argv[1], "--cuda") == 0;
        ggml_backend_t backend = nullptr;
        if (cuda) {
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
#else
            return 77;
#endif
        } else {
            backend = ggml_backend_cpu_init();
        }
        require(backend, "Requested backend unavailable");
        require(!MiniMaxH3::direct_backend_supported(nullptr), "Null backend enabled approximations");
        if (!cuda)
            require(!MiniMaxH3::direct_backend_supported(backend), "CPU/older backend must retain dense execution");
        auto ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        require(ctx, "Context allocation failed");
        auto source   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
        auto reshaped = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
        auto buffer   = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer, "Source allocation failed");
        {
            MiniMaxH3::H3GpuCacheStore store;
            store.backend = backend;
            for (int run : {0, 1, 0}) {
                const float values[6] = {float(run), 2, 3, 4, 5, 6};
                ggml_backend_tensor_set(source, values, 0, sizeof(values));
                require(store.capture("probe", source) && store.commit(), "Cache copy failed");
                float actual[6];
                ggml_backend_tensor_get(store.get("probe"), actual, 0, sizeof(actual));
                require(std::memcmp(values, actual, sizeof(values)) == 0, "A/B/A cache contents stale");
            }
            const float values[6] = {6, 5, 4, 3, 2, 1};
            ggml_backend_tensor_set(reshaped, values, 0, sizeof(values));
            require(store.capture("probe", reshaped) && store.commit(), "Cache reshape failed");
            require(store.get("probe")->ne[0] == 3 && store.get("probe")->ne[1] == 2, "Equal-size shape change reused old allocation");
        }
        std::weak_ptr<MiniMaxH3::H3GpuCacheStore> lifetime;
        {
            sd_sample::H3TeaCacheState request;
            auto store           = std::make_shared<MiniMaxH3::H3GpuCacheStore>();
            store->backend       = backend;
            lifetime             = store;
            request.device_store = std::move(store);
        }
        require(lifetime.expired(), "Request cache ownership leaked");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        std::puts("PASS conditioning/schedule/reference signatures, equal-size shapes, A/B/A backend cache copies, request ownership");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
