#include "coreml_unet.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdlib>
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

class CoreMLUNetImpl : public CoreMLUNet {
public:
    MLModel* model = nil;
    std::vector<int64_t> sample_shape;
    std::vector<int64_t> context_shape;
    std::string desc;

    bool predict(const float* x,
                 const int64_t x_shape[4],
                 const float* timesteps,
                 const float* context,
                 const int64_t ctx_shape[3],
                 float* out) override {
        @autoreleasepool {
            for (int i = 0; i < 4; i++) {
                if (x_shape[i] != sample_shape[i]) {
                    LOG_ERROR("coreml unet: sample shape [%lld, %lld, %lld, %lld] does not match the model's [%lld, %lld, %lld, %lld]",
                              x_shape[0], x_shape[1], x_shape[2], x_shape[3],
                              sample_shape[0], sample_shape[1], sample_shape[2], sample_shape[3]);
                    return false;
                }
            }
            for (int i = 0; i < 3; i++) {
                if (ctx_shape[i] != context_shape[i]) {
                    LOG_ERROR("coreml unet: context shape [%lld, %lld, %lld] does not match the model's [%lld, %lld, %lld]",
                              ctx_shape[0], ctx_shape[1], ctx_shape[2],
                              context_shape[0], context_shape[1], context_shape[2]);
                    return false;
                }
            }

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

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits          = compute_units_from_env();

        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (model == nil) {
            LOG_ERROR("coreml unet: failed to load %s: %s", path.c_str(), error.localizedDescription.UTF8String);
            return nullptr;
        }
        auto t1 = std::chrono::steady_clock::now();

        auto impl           = std::make_unique<CoreMLUNetImpl>();
        impl->model         = model;
        impl->sample_shape  = declared_shape(model.modelDescription.inputDescriptionsByName[@"sample"]);
        impl->context_shape = declared_shape(model.modelDescription.inputDescriptionsByName[@"encoder_hidden_states"]);
        if (impl->sample_shape.size() != 4 || impl->context_shape.size() != 3 ||
            model.modelDescription.inputDescriptionsByName[@"timestep"] == nil) {
            LOG_ERROR("coreml unet: %s does not have the expected sample/timestep/encoder_hidden_states inputs", path.c_str());
            return nullptr;
        }

        impl->desc = "CoreML UNet (" + std::string(compute_units_name(config.computeUnits)) + ")";
        LOG_INFO("coreml unet: loaded %s in %.2f s, compute units %s, sample [%lld, %lld, %lld, %lld], context [%lld, %lld, %lld]",
                 path.c_str(),
                 std::chrono::duration<double>(t1 - t0).count(),
                 compute_units_name(config.computeUnits),
                 impl->sample_shape[0], impl->sample_shape[1], impl->sample_shape[2], impl->sample_shape[3],
                 impl->context_shape[0], impl->context_shape[1], impl->context_shape[2]);
        return impl;
    }
}
