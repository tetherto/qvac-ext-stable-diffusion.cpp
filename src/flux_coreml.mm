#if !__has_feature(objc_arc)
#error "flux_coreml.mm requires Objective-C ARC"
#endif

#include "flux_coreml.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <mutex>

struct FluxCoreMLModel {
    MLModel* __strong model = nil;
    std::mutex prediction_mutex;
};

namespace flux_coreml_detail {

    bool matches_shape(NSArray<NSNumber*>* shape, const int64_t ne[4]) {
        if (shape.count != 4)
            return false;
        for (int i = 0; i < 4; ++i) {
            if (shape[i].longLongValue != ne[3 - i])
                return false;
        }
        return true;
    }

    bool is_contiguous(MLMultiArray* array) {
        int64_t stride = 1;
        for (NSInteger i = array.shape.count - 1; i >= 0; --i) {
            if (array.strides[i].longLongValue != stride)
                return false;
            stride *= array.shape[i].longLongValue;
        }
        return true;
    }

    MLMultiArray* make_input(MLFeatureDescription* description,
                             const FluxCoreMLTensor& tensor,
                             std::string* error) {
        MLMultiArrayConstraint* constraint = description.multiArrayConstraint;
        if (description.type != MLFeatureTypeMultiArray ||
            constraint.dataType != MLMultiArrayDataTypeFloat32 ||
            !matches_shape(constraint.shape, tensor.ne)) {
            *error = "Core ML input must be fixed-shape float32 with dimensions reversed from ggml";
            return nil;
        }
        NSError* ns_error   = nil;
        MLMultiArray* array = [[MLMultiArray alloc] initWithShape:constraint.shape
                                                         dataType:MLMultiArrayDataTypeFloat32
                                                            error:&ns_error];
        if (!array || !is_contiguous(array)) {
            *error = "Core ML could not allocate a contiguous input array";
            return nil;
        }
        std::memcpy(array.dataPointer, tensor.data,
                    static_cast<size_t>(array.count) * sizeof(float));
        return array;
    }

}  // namespace flux_coreml_detail

FluxCoreMLModel* flux_coreml_open(const char* path, std::string* error) {
    @autoreleasepool {
        if (!path || !path[0]) {
            *error = "Core ML model path is empty";
            return nullptr;
        }
        NSString* string_path = [NSString stringWithUTF8String:path];
        if (!string_path) {
            *error = "Core ML model path is invalid UTF-8";
            return nullptr;
        }
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits          = MLComputeUnitsAll;
        NSError* ns_error            = nil;
        MLModel* model               = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:string_path]
                                           configuration:config
                                                   error:&ns_error];
        if (!model) {
            *error = ns_error ? ns_error.localizedDescription.UTF8String : "Core ML model load failed";
            return nullptr;
        }
        NSDictionary<NSString*, MLFeatureDescription*>* inputs =
            model.modelDescription.inputDescriptionsByName;
        for (NSString* required in @[ @"latent", @"timesteps", @"context" ]) {
            if (!inputs[required] || inputs[required].type != MLFeatureTypeMultiArray) {
                *error = "Core ML FLUX.2 model is missing input: ";
                *error += required.UTF8String;
                return nullptr;
            }
        }
        MLFeatureDescription* output = model.modelDescription.outputDescriptionsByName[@"output"];
        if (!output || output.type != MLFeatureTypeMultiArray) {
            *error = "Core ML FLUX.2 model is missing output: output";
            return nullptr;
        }
        auto* result  = new FluxCoreMLModel();
        result->model = model;
        return result;
    }
}

void flux_coreml_close(FluxCoreMLModel* model) {
    delete model;
}

bool flux_coreml_predict(FluxCoreMLModel* model,
                         const FluxCoreMLTensor& latent,
                         const FluxCoreMLTensor& timesteps,
                         const FluxCoreMLTensor& context,
                         const FluxCoreMLTensor* pooled,
                         const FluxCoreMLTensor* guidance,
                         float* output,
                         std::string* error) {
    @autoreleasepool {
        if (!model || !output) {
            *error = "Core ML model or output is missing";
            return false;
        }
        // One compiled model instance is shared by calls from the same engine context.
        std::lock_guard<std::mutex> lock(model->prediction_mutex);
        NSDictionary<NSString*, MLFeatureDescription*>* descriptions =
            model->model.modelDescription.inputDescriptionsByName;
        NSMutableDictionary<NSString*, MLFeatureValue*>* values =
            [NSMutableDictionary dictionaryWithCapacity:descriptions.count];
        for (NSString* name in descriptions) {
            const FluxCoreMLTensor* tensor = nullptr;
            if ([name isEqualToString:@"latent"])
                tensor = &latent;
            else if ([name isEqualToString:@"timesteps"])
                tensor = &timesteps;
            else if ([name isEqualToString:@"context"])
                tensor = &context;
            else if ([name isEqualToString:@"pooled"])
                tensor = pooled;
            else if ([name isEqualToString:@"guidance"])
                tensor = guidance;
            if (!tensor || !tensor->data) {
                *error = "Core ML model requires an unavailable or unknown input: ";
                *error += name.UTF8String;
                return false;
            }
            MLMultiArray* array = flux_coreml_detail::make_input(descriptions[name], *tensor, error);
            if (!array) {
                *error += " (";
                *error += name.UTF8String;
                *error += ")";
                return false;
            }
            values[name] = [MLFeatureValue featureValueWithMultiArray:array];
        }
        NSError* ns_error = nil;
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:values
                                                              error:&ns_error];
        if (!provider) {
            *error = ns_error ? ns_error.localizedDescription.UTF8String : "Core ML input provider failed";
            return false;
        }
        id<MLFeatureProvider> result = [model->model predictionFromFeatures:provider error:&ns_error];
        if (!result) {
            *error = ns_error ? ns_error.localizedDescription.UTF8String : "Core ML prediction failed";
            return false;
        }
        MLMultiArray* array = [result featureValueForName:@"output"].multiArrayValue;
        if (!array || array.dataType != MLMultiArrayDataTypeFloat32 ||
            !flux_coreml_detail::matches_shape(array.shape, latent.ne) || !flux_coreml_detail::is_contiguous(array)) {
            *error = "Core ML output must be contiguous float32 and match latent shape";
            return false;
        }
        std::memcpy(output, array.dataPointer,
                    static_cast<size_t>(array.count) * sizeof(float));
        return true;
    }
}
