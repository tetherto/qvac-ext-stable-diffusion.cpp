#include "coreml_unet.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>

#include "coreml_utils.h"

using namespace sd_coreml;

namespace {

// One fixed-shape UNet. A multifunction model has one per resolution, all sharing the same weights.
struct UNetFunction {
    std::string name;  // empty for a single-function model
    std::vector<int64_t> sample_shape;
    // [N, T, D], or [N, D, 1, T] for models exported with Apple's ml-stable-diffusion (channels first for the ANE)
    std::vector<int64_t> context_shape;
    MLModel* model = nil;  // loaded on first use

    bool context_channels_first() const {
        return context_shape.size() == 4;
    }

    bool matches(const int64_t x_shape[4], const int64_t ctx_shape[3]) const {
        if (!std::equal(sample_shape.begin(), sample_shape.end(), x_shape)) {
            return false;
        }
        if (context_channels_first()) {
            return context_shape[0] == ctx_shape[0] && context_shape[1] == ctx_shape[2] && context_shape[2] == 1 &&
                   context_shape[3] == ctx_shape[1];
        }
        return std::equal(context_shape.begin(), context_shape.end(), ctx_shape);
    }
};

class CoreMLUNetImpl : public CoreMLUNet {
public:
    NSURL* url = nil;
    MLComputeUnits compute_units;
    std::vector<UNetFunction> functions;
    std::string desc;
    std::string prediction;

    UNetFunction* find_function(const int64_t x_shape[4], const int64_t ctx_shape[3]) {
        for (auto& f : functions) {
            if (f.matches(x_shape, ctx_shape)) {
                return &f;
            }
        }
        return nullptr;
    }

    bool load_function(UNetFunction& f) {
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits          = compute_units;
        if (!f.name.empty()) {
            if (@available(macOS 15.0, iOS 18.0, *)) {
                config.functionName = [NSString stringWithUTF8String:f.name.c_str()];
            } else {
                LOG_ERROR("coreml unet: multifunction models need iOS 18 / macOS 15");
                return false;
            }
        }
        NSError* error = nil;
        auto t0        = std::chrono::steady_clock::now();
        f.model        = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (f.model == nil) {
            LOG_ERROR("coreml unet: failed to load %s: %s",
                      f.name.empty() ? "model" : f.name.c_str(), error.localizedDescription.UTF8String);
            return false;
        }
        if (f.sample_shape.empty()) {  // single-function model: shapes come from the loaded model
            MLModelDescription* description = f.model.modelDescription;
            f.sample_shape                  = declared_shape(description.inputDescriptionsByName[@"sample"]);
            f.context_shape                 = declared_shape(description.inputDescriptionsByName[@"encoder_hidden_states"]);
        }
        LOG_INFO("coreml unet: loaded %s in %.2f s, sample %s",
                 f.name.empty() ? "model" : f.name.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 shape_str(f.sample_shape.data(), f.sample_shape.size()).c_str());
        return true;
    }

    bool predict(const float* x,
                 const int64_t x_shape[4],
                 const float* timesteps,
                 const float* context,
                 const int64_t ctx_shape[3],
                 float* out) override {
        @autoreleasepool {
            UNetFunction* f = find_function(x_shape, ctx_shape);
            if (f == nullptr) {
                std::string available;
                for (auto& fn : functions) {
                    available += " " + shape_str(fn.sample_shape.data(), 4) + "/" + shape_str(fn.context_shape.data(), fn.context_shape.size());
                }
                LOG_ERROR("coreml unet: no function for sample %s, context %s; the model has:%s",
                          shape_str(x_shape, 4).c_str(), shape_str(ctx_shape, 3).c_str(), available.c_str());
                return false;
            }
            if (f->model == nil && !load_function(*f)) {
                return false;
            }
            MLModel* model                            = f->model;
            const std::vector<int64_t>& sample_shape  = f->sample_shape;
            const std::vector<int64_t>& context_shape = f->context_shape;

            // [N, T, D] -> [N, D, 1, T] for channels-first models: one T x D matrix transpose per batch item
            // (vDSP_mtrans, 1.7x faster than plain loops; about 0.1 ms per step for SD 2.x).
            std::vector<float> transposed;
            if (f->context_channels_first()) {
                const int64_t N = ctx_shape[0], T = ctx_shape[1], D = ctx_shape[2];
                transposed.resize(N * T * D);
                for (int64_t b = 0; b < N; b++) {
                    vDSP_mtrans(context + b * T * D, 1, transposed.data() + b * T * D, 1, D, T);
                }
                context = transposed.data();
            }

            NSDictionary* in       = model.modelDescription.inputDescriptionsByName;
            NSError* error         = nil;
            MLMultiArray* sample   = make_input(x, sample_shape, declared_type(in[@"sample"]), &error);
            MLMultiArray* timestep = sample ? make_input(timesteps, {sample_shape[0]}, declared_type(in[@"timestep"]), &error) : nil;
            MLMultiArray* ehs      = timestep ? make_input(context, context_shape, declared_type(in[@"encoder_hidden_states"]), &error) : nil;
            if (ehs == nil) {
                LOG_ERROR("coreml unet: failed to wrap inputs: %s", error.localizedDescription.UTF8String);
                return false;
            }

            NSDictionary* inputs = @{
                @"sample" : [MLFeatureValue featureValueWithMultiArray:sample],
                @"timestep" : [MLFeatureValue featureValueWithMultiArray:timestep],
                @"encoder_hidden_states" : [MLFeatureValue featureValueWithMultiArray:ehs],
            };
            MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&error];
            if (provider == nil) {
                LOG_ERROR("coreml unet: %s", error.localizedDescription.UTF8String);
                return false;
            }

            auto t0                           = std::chrono::steady_clock::now();
            id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
            auto t1                           = std::chrono::steady_clock::now();
            if (prediction == nil) {
                LOG_ERROR("coreml unet: prediction failed: %s", error.localizedDescription.UTF8String);
                return false;
            }
            LOG_DEBUG("coreml unet: prediction took %.2f ms",
                      std::chrono::duration<double, std::milli>(t1 - t0).count());

            MLMultiArray* result = [prediction featureValueForName:@"noise_pred"].multiArrayValue;
            if (result == nil) {
                LOG_ERROR("coreml unet: model has no 'noise_pred' output");
                return false;
            }

            int64_t n = 1;
            for (int64_t d : sample_shape) {
                n *= d;
            }
            return copy_output(result, n, out, "coreml unet");
        }
    }

    std::string get_desc() override {
        return desc;
    }

    std::string prediction_type() override {
        return prediction;
    }

    int64_t batch_size() override {
        return functions[0].sample_shape[0];
    }
};

}  // namespace

bool CoreMLUNet::run(const float* x,
                     const int64_t x_shape[4],
                     const float* timesteps,
                     int64_t n_timesteps,
                     const float* context,
                     const int64_t context_shape[3],
                     float* out) {
    const int64_t n = x_shape[0], model_n = batch_size();
    if (context_shape[0] != n || (n_timesteps != 1 && n_timesteps != n)) {
        LOG_ERROR("coreml unet: batch mismatch: latent %lld, context %lld, timesteps %lld",
                  (long long)n, (long long)context_shape[0], (long long)n_timesteps);
        return false;
    }
    const int64_t nx = x_shape[1] * x_shape[2] * x_shape[3];
    const int64_t nc = context_shape[1] * context_shape[2];
    if (n == model_n) {
        std::vector<float> ts(n, timesteps[0]);
        if (n_timesteps == n) {
            ts.assign(timesteps, timesteps + n);
        }
        return predict(x, x_shape, ts.data(), context, context_shape, out);
    }
    if (n == 1 && model_n == 2) {
        std::vector<float> xs(2 * nx), ts(2, timesteps[0]), ctx(2 * nc), outs(2 * nx);
        std::copy(x, x + nx, xs.begin());
        std::copy(x, x + nx, xs.begin() + nx);
        std::copy(context, context + nc, ctx.begin());
        std::copy(context, context + nc, ctx.begin() + nc);
        const int64_t xs_shape[4]  = {2, x_shape[1], x_shape[2], x_shape[3]};
        const int64_t ctx_shape[3] = {2, context_shape[1], context_shape[2]};
        if (!predict(xs.data(), xs_shape, ts.data(), ctx.data(), ctx_shape, outs.data())) {
            return false;
        }
        std::copy(outs.begin(), outs.begin() + nx, out);
        return true;
    }
    if (model_n == 1) {
        const int64_t x1_shape[4]   = {1, x_shape[1], x_shape[2], x_shape[3]};
        const int64_t ctx1_shape[3] = {1, context_shape[1], context_shape[2]};
        for (int64_t b = 0; b < n; b++) {
            const float t = timesteps[n_timesteps == n ? b : 0];
            if (!predict(x + b * nx, x1_shape, &t, context + b * nc, ctx1_shape, out + b * nx)) {
                return false;
            }
        }
        return true;
    }
    LOG_ERROR("coreml unet: cannot run batch %lld on a model exported for batch %lld", (long long)n, (long long)model_n);
    return false;
}

std::unique_ptr<CoreMLUNet> CoreMLUNet::load(const std::string& path) {
    @autoreleasepool {
        NSError* error = nil;
        auto t0        = std::chrono::steady_clock::now();
        NSURL* url     = compiled_model_url(path, "coreml unet");
        if (url == nil) {
            return nullptr;
        }

        auto impl           = std::make_unique<CoreMLUNetImpl>();
        impl->url           = url;
        impl->compute_units = compute_units_from_env();

        // A multifunction model (one fixed shape per resolution) lists its functions and their
        // descriptions up front, so only the functions actually used get loaded and compiled.
        if (@available(macOS 15.0, iOS 18.0, *)) {
            MLModelAsset* asset = [MLModelAsset modelAssetWithURL:url error:&error];
            if (asset != nil) {
                __block NSArray<NSString*>* names = nil;
                dispatch_semaphore_t done         = dispatch_semaphore_create(0);
                [asset functionNamesWithCompletionHandler:^(NSArray<NSString*>* result, NSError* err) {
                    names = result;
                    dispatch_semaphore_signal(done);
                }];
                dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
                for (NSString* name in names) {
                    __block MLModelDescription* description = nil;
                    [asset modelDescriptionOfFunctionNamed:name
                                         completionHandler:^(MLModelDescription* result, NSError* err) {
                                             description = result;
                                             dispatch_semaphore_signal(done);
                                         }];
                    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
                    UNetFunction f;
                    f.name          = name.UTF8String;
                    f.sample_shape  = declared_shape(description.inputDescriptionsByName[@"sample"]);
                    f.context_shape = declared_shape(description.inputDescriptionsByName[@"encoder_hidden_states"]);
                    impl->functions.push_back(f);
                }
            }
        }

        if (impl->functions.empty()) {
            UNetFunction f;
            if (!impl->load_function(f)) {
                return nullptr;
            }
            impl->functions.push_back(f);
        }

        for (auto& f : impl->functions) {
            if (f.sample_shape.size() != 4 || (f.context_shape.size() != 3 && f.context_shape.size() != 4)) {
                LOG_ERROR("coreml unet: %s%s does not have fixed-shape sample/encoder_hidden_states inputs",
                          path.c_str(), f.name.empty() ? "" : (" (" + f.name + ")").c_str());
                return nullptr;
            }
        }

        // Load the first function now, so a bad model fails here rather than mid-generation.
        if (impl->functions[0].model == nil && !impl->load_function(impl->functions[0])) {
            return nullptr;
        }

        NSDictionary* metadata = impl->functions[0].model.modelDescription.metadata[MLModelCreatorDefinedKey];
        NSString* prediction   = metadata[@"prediction_type"];
        if (prediction != nil) {
            impl->prediction = prediction.UTF8String;
        }

        std::string shapes;
        for (auto& f : impl->functions) {
            shapes += " " + std::to_string(f.sample_shape[3] * 8) + "x" + std::to_string(f.sample_shape[2] * 8);
        }
        impl->desc = "CoreML UNet (" + std::string(compute_units_name(impl->compute_units)) + ")";
        LOG_INFO("coreml unet: %s ready in %.2f s, compute units %s, %zu function(s), WxH:%s, prediction %s",
                 path.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 compute_units_name(impl->compute_units),
                 impl->functions.size(),
                 shapes.c_str(),
                 impl->prediction.empty() ? "not recorded" : impl->prediction.c_str());
        return impl;
    }
}
