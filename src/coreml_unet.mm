#include "coreml_unet.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cstring>

#include "util.h"

namespace {

MLComputeUnits compute_units_from_env() {
    const char* env = getenv("SD_COREML_COMPUTE_UNITS");
    std::string v   = env ? env : "cpu_ne";
    if (v == "all") {
        return MLComputeUnitsAll;
    }
    if (v == "cpu_gpu") {
        return MLComputeUnitsCPUAndGPU;
    }
    if (v == "cpu") {
        return MLComputeUnitsCPUOnly;
    }
    return MLComputeUnitsCPUAndNeuralEngine;
}

const char* compute_units_name(MLComputeUnits units) {
    switch (units) {
        case MLComputeUnitsAll:
            return "all";
        case MLComputeUnitsCPUAndGPU:
            return "cpu_gpu";
        case MLComputeUnitsCPUOnly:
            return "cpu";
        default:
            return "cpu_ne";
    }
}

// Shape of a fixed-size multiarray input or output, or empty if the model does not declare one.
std::vector<int64_t> declared_shape(MLFeatureDescription* desc) {
    std::vector<int64_t> shape;
    if (desc == nil || desc.multiArrayConstraint == nil) {
        return shape;
    }
    for (NSNumber* d in desc.multiArrayConstraint.shape) {
        shape.push_back(d.longLongValue);
    }
    return shape;
}

MLMultiArray* wrap(const float* data, const std::vector<int64_t>& shape, NSError** error) {
    NSMutableArray<NSNumber*>* dims    = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];
    int64_t stride                     = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        [dims addObject:@(shape[i])];
    }
    for (size_t i = shape.size(); i-- > 0;) {
        [strides insertObject:@(stride) atIndex:0];
        stride *= shape[i];
    }
    // CoreML only reads inputs, so the const_cast does not let it write to caller memory.
    return [[MLMultiArray alloc] initWithDataPointer:const_cast<float*>(data)
                                               shape:dims
                                            dataType:MLMultiArrayDataTypeFloat32
                                             strides:strides
                                         deallocator:nil
                                               error:error];
}

// One fixed-shape UNet. A multifunction model has one per resolution, all sharing the same weights.
struct UNetFunction {
    std::string name;  // empty for a single-function model
    std::vector<int64_t> sample_shape;
    std::vector<int64_t> context_shape;
    MLModel* model = nil;  // loaded on first use
};

std::string shape_str(const int64_t* shape, size_t n) {
    std::string s = "[";
    for (size_t i = 0; i < n; i++) {
        s += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return s + "]";
}

class CoreMLUNetImpl : public CoreMLUNet {
public:
    NSURL* url = nil;
    MLComputeUnits compute_units;
    std::vector<UNetFunction> functions;
    std::string desc;

    UNetFunction* find_function(const int64_t x_shape[4], const int64_t ctx_shape[3]) {
        for (auto& f : functions) {
            if (std::equal(f.sample_shape.begin(), f.sample_shape.end(), x_shape) &&
                std::equal(f.context_shape.begin(), f.context_shape.end(), ctx_shape)) {
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
                    available += " " + shape_str(fn.sample_shape.data(), 4) + "/" + shape_str(fn.context_shape.data(), 3);
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

            NSError* error         = nil;
            MLMultiArray* sample   = wrap(x, sample_shape, &error);
            MLMultiArray* timestep = sample ? wrap(timesteps, {sample_shape[0]}, &error) : nil;
            MLMultiArray* ehs      = timestep ? wrap(context, context_shape, &error) : nil;
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
            if (result.count != n) {
                LOG_ERROR("coreml unet: output has %lld elements, expected %lld", (long long)result.count, (long long)n);
                return false;
            }

            // The output may be float16 and may have padded strides, so copy element by element in logical order.
            __block bool ok = true;
            [result getBytesWithHandler:^(const void* bytes, NSInteger size) {
                (void)size;
                const int rank = (int)result.shape.count;
                std::vector<int64_t> shape(rank), strides(rank), idx(rank, 0);
                for (int i = 0; i < rank; i++) {
                    shape[i]   = result.shape[i].longLongValue;
                    strides[i] = result.strides[i].longLongValue;
                }
                for (int64_t i = 0; i < n; i++) {
                    int64_t offset = 0;
                    for (int d = 0; d < rank; d++) {
                        offset += idx[d] * strides[d];
                    }
                    switch (result.dataType) {
                        case MLMultiArrayDataTypeFloat32:
                            out[i] = ((const float*)bytes)[offset];
                            break;
                        case MLMultiArrayDataTypeFloat16:
                            out[i] = (float)((const _Float16*)bytes)[offset];
                            break;
                        default:
                            ok = false;
                            return;
                    }
                    for (int d = rank - 1; d >= 0; d--) {
                        if (++idx[d] < shape[d]) {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
            }];
            if (!ok) {
                LOG_ERROR("coreml unet: unsupported output data type");
            }
            return ok;
        }
    }

    std::string get_desc() override {
        return desc;
    }
};

}  // namespace

std::unique_ptr<CoreMLUNet> CoreMLUNet::load(const std::string& path) {
    @autoreleasepool {
        NSError* error = nil;
        NSURL* url     = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];

        auto t0 = std::chrono::steady_clock::now();
        if (![url.pathExtension isEqualToString:@"mlmodelc"]) {
            LOG_INFO("coreml unet: compiling %s (ship a .mlmodelc to skip this step)", path.c_str());
            url = [MLModel compileModelAtURL:url error:&error];
            if (url == nil) {
                LOG_ERROR("coreml unet: failed to compile %s: %s", path.c_str(), error.localizedDescription.UTF8String);
                return nullptr;
            }
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
            MLModelDescription* description = f.model.modelDescription;
            f.sample_shape                  = declared_shape(description.inputDescriptionsByName[@"sample"]);
            f.context_shape                 = declared_shape(description.inputDescriptionsByName[@"encoder_hidden_states"]);
            impl->functions.push_back(f);
        }

        for (auto& f : impl->functions) {
            if (f.sample_shape.size() != 4 || f.context_shape.size() != 3) {
                LOG_ERROR("coreml unet: %s%s does not have fixed-shape sample/encoder_hidden_states inputs",
                          path.c_str(), f.name.empty() ? "" : (" (" + f.name + ")").c_str());
                return nullptr;
            }
        }

        // Load the first function now, so a bad model fails here rather than mid-generation.
        if (impl->functions[0].model == nil && !impl->load_function(impl->functions[0])) {
            return nullptr;
        }

        std::string shapes;
        for (auto& f : impl->functions) {
            shapes += " " + std::to_string(f.sample_shape[3] * 8) + "x" + std::to_string(f.sample_shape[2] * 8);
        }
        impl->desc = "CoreML UNet (" + std::string(compute_units_name(impl->compute_units)) + ")";
        LOG_INFO("coreml unet: %s ready in %.2f s, compute units %s, %zu function(s), WxH:%s",
                 path.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 compute_units_name(impl->compute_units),
                 impl->functions.size(),
                 shapes.c_str());
        return impl;
    }
}
