#if !__has_feature(objc_arc)
#error "flux_coreml.mm requires Objective-C ARC"
#endif

#include "flux_coreml.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <mutex>

#include "coreml_utils.h"

struct FluxCoreMLModel {
    // One model, or the chunks of one in order. Chunk inputs other than the engine's are outputs of
    // earlier chunks, which are passed on without conversion.
    NSArray<MLModel*>* __strong stages = nil;
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

    bool is_engine_input(NSString* name) {
        for (NSString* engine in @[ @"latent", @"timesteps", @"context", @"pooled", @"guidance" ]) {
            if ([name isEqualToString:engine])
                return true;
        }
        return false;
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
        // A folder that is not itself a model holds chunk1.mlmodelc, chunk2.mlmodelc, ...
        std::vector<std::string> paths = {path};
        BOOL is_dir                    = NO;
        NSString* extension            = string_path.pathExtension.lowercaseString;
        if ([[NSFileManager defaultManager] fileExistsAtPath:string_path isDirectory:&is_dir] && is_dir &&
            ![extension isEqualToString:@"mlmodelc"] && ![extension isEqualToString:@"mlpackage"]) {
            paths = sd_coreml::find_chunks(path, "coreml flux");
            if (paths.empty()) {
                *error = "folder holds no Core ML chunks (chunk1.mlmodelc, chunk2.mlmodelc, ...)";
                return nullptr;
            }
        }
        // CPU and Neural Engine unless SD_COREML_COMPUTE_UNITS says otherwise, as for the other sidecars.
        // A float32 export (export_flux2_klein_coreml.py convert) needs SD_COREML_COMPUTE_UNITS=all.
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits          = sd_coreml::compute_units_from_env();
        NSMutableArray<MLModel*>* stages = [NSMutableArray arrayWithCapacity:paths.size()];
        NSMutableSet<NSString*>* available =
            [NSMutableSet setWithArray:@[ @"latent", @"timesteps", @"context", @"pooled", @"guidance" ]];
        for (size_t i = 0; i < paths.size(); ++i) {
            NSURL* url = sd_coreml::compiled_model_url(paths[i], "coreml flux");
            if (!url) {
                *error = "could not compile " + paths[i];
                return nullptr;
            }
            NSError* ns_error = nil;
            MLModel* model    = [MLModel modelWithContentsOfURL:url configuration:config error:&ns_error];
            if (!model) {
                *error = paths[i] + ": " + (ns_error ? ns_error.localizedDescription.UTF8String : "Core ML model load failed");
                return nullptr;
            }
            for (NSString* name in model.modelDescription.inputDescriptionsByName) {
                if (![available containsObject:name]) {
                    *error = paths[i] + " takes '" + name.UTF8String +
                             "', which is neither an engine input nor an earlier chunk's output";
                    return nullptr;
                }
            }
            [available addObjectsFromArray:model.modelDescription.outputDescriptionsByName.allKeys];
            [stages addObject:model];
        }
        for (NSString* required in @[ @"latent", @"timesteps", @"context" ]) {
            bool taken = false;
            for (MLModel* stage in stages) {
                MLFeatureDescription* input = stage.modelDescription.inputDescriptionsByName[required];
                taken |= input != nil && input.type == MLFeatureTypeMultiArray;
            }
            if (!taken) {
                *error = "Core ML FLUX.2 model is missing input: ";
                *error += required.UTF8String;
                return nullptr;
            }
        }
        MLFeatureDescription* output = stages.lastObject.modelDescription.outputDescriptionsByName[@"output"];
        if (!output || output.type != MLFeatureTypeMultiArray) {
            *error = "Core ML FLUX.2 model is missing output: output";
            return nullptr;
        }
        LOG_INFO("coreml flux: %zu stage(s), compute units %s", paths.size(),
                 sd_coreml::compute_units_name(config.computeUnits));
        auto* result   = new FluxCoreMLModel();
        result->stages = stages;
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
        NSMutableDictionary<NSString*, MLFeatureValue*>* values = [NSMutableDictionary dictionary];
        id<MLFeatureProvider> result                            = nil;
        for (MLModel* stage in model->stages) {
            NSDictionary<NSString*, MLFeatureDescription*>* descriptions =
                stage.modelDescription.inputDescriptionsByName;
            NSMutableDictionary<NSString*, MLFeatureValue*>* feed =
                [NSMutableDictionary dictionaryWithCapacity:descriptions.count];
            for (NSString* name in descriptions) {
                if (!flux_coreml_detail::is_engine_input(name)) {
                    feed[name] = values[name];
                    continue;
                }
                if (!values[name]) {
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
                        *error = "Core ML model requires an unavailable input: ";
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
                feed[name] = values[name];
            }
            NSError* ns_error = nil;
            MLDictionaryFeatureProvider* provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&ns_error];
            if (!provider) {
                *error = ns_error ? ns_error.localizedDescription.UTF8String : "Core ML input provider failed";
                return false;
            }
            result = [stage predictionFromFeatures:provider error:&ns_error];
            if (!result) {
                *error = ns_error ? ns_error.localizedDescription.UTF8String : "Core ML prediction failed";
                return false;
            }
            for (NSString* name in result.featureNames) {
                values[name] = [result featureValueForName:name];
            }
        }
        MLMultiArray* array = [result featureValueForName:@"output"].multiArrayValue;
        if (!array || !flux_coreml_detail::matches_shape(array.shape, latent.ne)) {
            *error = "Core ML output must match the latent shape";
            return false;
        }
        // Outputs can be fp16 or have padded rows, so copy element by element.
        if (!sd_coreml::copy_output(array, array.count, output, "coreml flux")) {
            *error = "Core ML output must be float32 or float16";
            return false;
        }
        return true;
    }
}
