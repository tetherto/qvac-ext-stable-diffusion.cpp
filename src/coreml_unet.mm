#include "coreml_unet.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <regex>

#include "coreml_utils.h"

using namespace sd_coreml;

namespace {

// SDXL's addition_time_embed_dim: every time id (sizes and crop) is embedded into this many values.
constexpr int64_t kTimeIdEmbedDim = 256;

// The inputs the UNet itself provides. Any other input of a chunk is an output of an earlier chunk.
const char* const kUNetInputs[] = {"sample", "timestep", "encoder_hidden_states", "y", "text_embeds", "time_ids"};

bool is_unet_input(NSString* name) {
    for (const char* input : kUNetInputs) {
        if ([name isEqualToString:@(input)]) {
            return true;
        }
    }
    return false;
}

int64_t element_count(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return n;
}

// sd.cpp passes SDXL's size and crop conditioning already embedded in y (see the SDXL conditioner): each id t
// becomes cos(t * f_j) for 128 frequencies f_j, then sin(t * f_j). Apple's exports take the raw ids and embed
// them in the model, so recover each id from its lowest frequency, where t * f stays below pi for t < 29,000.
void decode_time_ids(const float* embedded, int64_t n_ids, float* ids) {
    const int64_t half = kTimeIdEmbedDim / 2;
    const double freq  = std::exp(-std::log(10000.0) * (double)(half - 1) / (double)half);
    for (int64_t i = 0; i < n_ids; i++) {
        const float* e = embedded + i * kTimeIdEmbedDim;
        ids[i]         = (float)(std::atan2((double)e[2 * half - 1], (double)e[half - 1]) / freq);
    }
}

// One fixed-shape UNet. A multifunction model has one per resolution, all sharing the same weights.
struct UNetFunction {
    std::string name;  // empty for a single-function model
    std::vector<int64_t> sample_shape;
    // [N, T, D], or [N, D, 1, T] for models exported with Apple's ml-stable-diffusion (channels first for the ANE)
    std::vector<int64_t> context_shape;
    std::vector<int64_t> y_shape;            // [N, A], convert_unet.py's SDXL exports
    std::vector<int64_t> text_embeds_shape;  // [N, P] and
    std::vector<int64_t> time_ids_shape;     // [N, K], Apple's SDXL exports
    std::vector<MLModel*> stages;            // one per chunk, loaded on first use

    bool loaded() const {
        return !stages.empty();
    }

    bool context_channels_first() const {
        return context_shape.size() == 4;
    }

    int64_t y_dim() const {
        if (y_shape.size() == 2) {
            return y_shape[1];
        }
        if (text_embeds_shape.size() == 2 && time_ids_shape.size() == 2) {
            return text_embeds_shape[1] + kTimeIdEmbedDim * time_ids_shape[1];
        }
        return 0;
    }

    void read_shapes(MLModelDescription* description) {
        NSDictionary<NSString*, MLFeatureDescription*>* in = description.inputDescriptionsByName;
        for (auto [input, shape] : {std::pair<NSString*, std::vector<int64_t>*>{@"sample", &sample_shape},
                                    {@"encoder_hidden_states", &context_shape},
                                    {@"y", &y_shape},
                                    {@"text_embeds", &text_embeds_shape},
                                    {@"time_ids", &time_ids_shape}}) {
            if (shape->empty()) {
                *shape = declared_shape(in[input]);
            }
        }
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

    std::string label() const {
        return name.empty() ? "model" : name;
    }
};

class CoreMLUNetImpl : public CoreMLUNet {
public:
    std::vector<NSURL*> urls;  // one model, or the chunks in order
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
        auto t0 = std::chrono::steady_clock::now();
        std::vector<MLModel*> stages;
        for (size_t i = 0; i < urls.size(); i++) {
            NSError* error = nil;
            MLModel* model = [MLModel modelWithContentsOfURL:urls[i] configuration:config error:&error];
            if (model == nil) {
                LOG_ERROR("coreml unet: failed to load %s%s: %s", f.label().c_str(),
                          urls.size() > 1 ? (", chunk " + std::to_string(i + 1)).c_str() : "",
                          error.localizedDescription.UTF8String);
                return false;
            }
            stages.push_back(model);
            f.read_shapes(model.modelDescription);
        }
        if (!check_stages(f, stages)) {
            return false;
        }
        f.stages = std::move(stages);
        LOG_INFO("coreml unet: loaded %s in %.2f s, sample %s%s",
                 f.label().c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 shape_str(f.sample_shape.data(), f.sample_shape.size()).c_str(),
                 urls.size() > 1 ? (", " + std::to_string(urls.size()) + " chunks").c_str() : "");
        return true;
    }

    // Every chunk input must be a UNet input or an earlier chunk's output, and the last chunk returns noise_pred.
    bool check_stages(const UNetFunction& f, const std::vector<MLModel*>& stages) {
        NSMutableSet<NSString*>* available = [NSMutableSet set];
        for (size_t i = 0; i < stages.size(); i++) {
            MLModelDescription* description = stages[i].modelDescription;
            for (NSString* input in description.inputDescriptionsByName) {
                if (!is_unet_input(input) && ![available containsObject:input]) {
                    LOG_ERROR("coreml unet: %s chunk %zu takes '%s', which is neither a UNet input nor an earlier chunk's output",
                              f.label().c_str(), i + 1, input.UTF8String);
                    return false;
                }
            }
            [available addObjectsFromArray:description.outputDescriptionsByName.allKeys];
        }
        if (stages.back().modelDescription.outputDescriptionsByName[@"noise_pred"] == nil) {
            LOG_ERROR("coreml unet: model has no 'noise_pred' output");
            return false;
        }
        if ((f.text_embeds_shape.empty()) != (f.time_ids_shape.empty())) {
            LOG_ERROR("coreml unet: a model with text_embeds also needs time_ids, and the other way round");
            return false;
        }
        return true;
    }

    bool predict(const float* x,
                 const int64_t x_shape[4],
                 const float* timesteps,
                 const float* context,
                 const int64_t ctx_shape[3],
                 const float* y,
                 int64_t y_dim,
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
            if (!f->loaded() && !load_function(*f)) {
                return false;
            }
            const int64_t n = x_shape[0];
            if ((y != nullptr ? y_dim : 0) != f->y_dim()) {
                LOG_ERROR("coreml unet: the model takes %lld values of vector conditioning (y), got %lld",
                          (long long)f->y_dim(), (long long)(y != nullptr ? y_dim : 0));
                return false;
            }

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

            std::map<std::string, std::pair<const float*, int64_t>> unet_inputs = {
                {"sample", {x, element_count(f->sample_shape)}},
                {"timestep", {timesteps, n}},
                {"encoder_hidden_states", {context, ctx_shape[0] * ctx_shape[1] * ctx_shape[2]}},
            };
            std::vector<float> text_embeds, time_ids;
            if (!f->y_shape.empty()) {
                unet_inputs["y"] = {y, n * y_dim};
            } else if (!f->text_embeds_shape.empty()) {
                const int64_t p = f->text_embeds_shape[1], k = f->time_ids_shape[1];
                text_embeds.resize(n * p);
                time_ids.resize(n * k);
                for (int64_t b = 0; b < n; b++) {
                    std::copy(y + b * y_dim, y + b * y_dim + p, text_embeds.begin() + b * p);
                    decode_time_ids(y + b * y_dim + p, k, time_ids.data() + b * k);
                }
                unet_inputs["text_embeds"] = {text_embeds.data(), n * p};
                unet_inputs["time_ids"]    = {time_ids.data(), n * k};
            }

            // Outputs of the chunks run so far, by name; a single model is one chunk.
            NSMutableDictionary<NSString*, MLFeatureValue*>* carried = [NSMutableDictionary dictionary];
            id<MLFeatureProvider> prediction                         = nil;
            auto t0                                                  = std::chrono::steady_clock::now();
            for (size_t i = 0; i < f->stages.size(); i++) {
                MLModel* model                                      = f->stages[i];
                NSMutableDictionary<NSString*, MLFeatureValue*>* in = [NSMutableDictionary dictionary];
                NSError* error                                      = nil;
                for (NSString* name in model.modelDescription.inputDescriptionsByName) {
                    MLFeatureDescription* input = model.modelDescription.inputDescriptionsByName[name];
                    auto it                     = unet_inputs.find(name.UTF8String);
                    if (it == unet_inputs.end()) {
                        if (carried[name] == nil) {
                            LOG_ERROR("coreml unet: chunk %zu needs '%s', which neither the UNet nor an earlier chunk provides",
                                      i + 1, name.UTF8String);
                            return false;
                        }
                        in[name] = carried[name];
                        continue;
                    }
                    std::vector<int64_t> shape = declared_shape(input);
                    if (it->second.first == nullptr || element_count(shape) != it->second.second) {
                        LOG_ERROR("coreml unet: input '%s' %s does not match the %lld values sd.cpp has",
                                  name.UTF8String, shape_str(shape.data(), shape.size()).c_str(), (long long)it->second.second);
                        return false;
                    }
                    MLMultiArray* array = make_input(it->second.first, shape, declared_type(input), &error);
                    if (array == nil) {
                        LOG_ERROR("coreml unet: failed to wrap input '%s': %s", name.UTF8String, error.localizedDescription.UTF8String);
                        return false;
                    }
                    in[name] = [MLFeatureValue featureValueWithMultiArray:array];
                }
                MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:in error:&error];
                if (provider == nil) {
                    LOG_ERROR("coreml unet: %s", error.localizedDescription.UTF8String);
                    return false;
                }
                prediction = [model predictionFromFeatures:provider error:&error];
                if (prediction == nil) {
                    LOG_ERROR("coreml unet: prediction failed%s: %s",
                              f->stages.size() > 1 ? (" in chunk " + std::to_string(i + 1)).c_str() : "",
                              error.localizedDescription.UTF8String);
                    return false;
                }
                if (i + 1 < f->stages.size()) {
                    for (NSString* name in prediction.featureNames) {
                        carried[name] = [prediction featureValueForName:name];
                    }
                }
            }
            LOG_DEBUG("coreml unet: prediction took %.2f ms",
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

            MLMultiArray* result = [prediction featureValueForName:@"noise_pred"].multiArrayValue;
            if (result == nil) {
                LOG_ERROR("coreml unet: model has no 'noise_pred' output");
                return false;
            }
            return copy_output(result, element_count(f->sample_shape), out, "coreml unet");
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

    int64_t y_dim() override {
        return functions[0].y_dim();
    }
};

}  // namespace

bool CoreMLUNet::run(const float* x,
                     const int64_t x_shape[4],
                     const float* timesteps,
                     int64_t n_timesteps,
                     const float* context,
                     const int64_t context_shape[3],
                     const float* y,
                     int64_t n_y,
                     int64_t y_dim,
                     float* out) {
    const int64_t n = x_shape[0], model_n = batch_size();
    if (context_shape[0] != n || (n_timesteps != 1 && n_timesteps != n) || (y != nullptr && n_y != 1 && n_y != n)) {
        LOG_ERROR("coreml unet: batch mismatch: latent %lld, context %lld, timesteps %lld, y %lld",
                  (long long)n, (long long)context_shape[0], (long long)n_timesteps, (long long)(y ? n_y : 0));
        return false;
    }
    const int64_t nx = x_shape[1] * x_shape[2] * x_shape[3];
    const int64_t nc = context_shape[1] * context_shape[2];
    auto y_row       = [&](int64_t b) { return y == nullptr ? nullptr : y + (n_y == n ? b : 0) * y_dim; };
    // y for `count` items starting at item 0 (or repeated from item 0): rows copied out only when y is shared.
    std::vector<float> ys;
    auto y_rows = [&](int64_t count) -> const float* {
        if (y == nullptr || n_y == count) {
            return y;
        }
        ys.resize(count * y_dim);
        for (int64_t b = 0; b < count; b++) {
            std::copy(y_row(std::min(b, n - 1)), y_row(std::min(b, n - 1)) + y_dim, ys.begin() + b * y_dim);
        }
        return ys.data();
    };
    if (n == model_n) {
        std::vector<float> ts(n, timesteps[0]);
        if (n_timesteps == n) {
            ts.assign(timesteps, timesteps + n);
        }
        return predict(x, x_shape, ts.data(), context, context_shape, y_rows(n), y_dim, out);
    }
    if (n == 1 && model_n == 2) {
        std::vector<float> xs(2 * nx), ts(2, timesteps[0]), ctx(2 * nc), outs(2 * nx);
        std::copy(x, x + nx, xs.begin());
        std::copy(x, x + nx, xs.begin() + nx);
        std::copy(context, context + nc, ctx.begin());
        std::copy(context, context + nc, ctx.begin() + nc);
        const int64_t xs_shape[4]  = {2, x_shape[1], x_shape[2], x_shape[3]};
        const int64_t ctx_shape[3] = {2, context_shape[1], context_shape[2]};
        if (!predict(xs.data(), xs_shape, ts.data(), ctx.data(), ctx_shape, y_rows(2), y_dim, outs.data())) {
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
            if (!predict(x + b * nx, x1_shape, &t, context + b * nc, ctx1_shape, y_row(b), y_dim, out + b * nx)) {
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

        std::vector<std::string> paths = {path};
        BOOL is_dir                    = NO;
        NSString* ns_path              = @(path.c_str());
        if ([[NSFileManager defaultManager] fileExistsAtPath:ns_path isDirectory:&is_dir] && is_dir &&
            ![ns_path.pathExtension isEqualToString:@"mlmodelc"] && ![ns_path.pathExtension isEqualToString:@"mlpackage"]) {
            paths = find_chunks(path, "coreml unet");
            if (paths.empty()) {
                LOG_ERROR("coreml unet: %s is a folder without UNet chunks (*chunk1.mlmodelc, *chunk2.mlmodelc, ...)", path.c_str());
                return nullptr;
            }
        }

        auto impl           = std::make_unique<CoreMLUNetImpl>();
        impl->compute_units = compute_units_from_env();
        for (const std::string& p : paths) {
            NSURL* url = compiled_model_url(p, "coreml unet");
            if (url == nil) {
                return nullptr;
            }
            impl->urls.push_back(url);
        }

        // A multifunction model (one fixed shape per resolution) lists its functions and their
        // descriptions up front, so only the functions actually used get loaded and compiled.
        // Chunks of a multifunction model all have the same function names; the first chunk lists them.
        if (@available(macOS 15.0, iOS 18.0, *)) {
            MLModelAsset* asset = [MLModelAsset modelAssetWithURL:impl->urls[0] error:&error];
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
                    f.name = name.UTF8String;
                    f.read_shapes(description);
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
        if (!impl->functions[0].loaded() && !impl->load_function(impl->functions[0])) {
            return nullptr;
        }

        NSDictionary* metadata = impl->functions[0].stages[0].modelDescription.metadata[MLModelCreatorDefinedKey];
        NSString* prediction   = metadata[@"prediction_type"];
        if (prediction != nil) {
            impl->prediction = prediction.UTF8String;
        }

        std::string shapes;
        for (auto& f : impl->functions) {
            shapes += " " + std::to_string(f.sample_shape[3] * 8) + "x" + std::to_string(f.sample_shape[2] * 8);
        }
        impl->desc = "CoreML UNet (" + std::string(compute_units_name(impl->compute_units)) + ")";
        LOG_INFO("coreml unet: %s ready in %.2f s, compute units %s, %zu chunk(s), %zu function(s), WxH:%s, y %lld, prediction %s",
                 path.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 compute_units_name(impl->compute_units),
                 impl->urls.size(),
                 impl->functions.size(),
                 shapes.c_str(),
                 (long long)impl->functions[0].y_dim(),
                 impl->prediction.empty() ? "not recorded" : impl->prediction.c_str());
        return impl;
    }
}
