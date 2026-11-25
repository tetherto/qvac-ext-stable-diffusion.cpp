#if !__has_feature(objc_arc)
#error "coreml_te.mm requires Objective-C ARC"
#endif

#include "coreml_te.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cstring>
#include <mutex>

#include "coreml_utils.h"

struct CoreMLTextEncoder {
    NSArray<MLModel*>* __strong stages = nil;
    int64_t length                     = 0;
    std::mutex mutex;
};

CoreMLTextEncoder* coreml_te_open(const std::string& path, std::string* error) {
    @autoreleasepool {
        const auto t0                  = std::chrono::steady_clock::now();
        std::vector<std::string> paths = sd_coreml::find_chunks(path, "coreml te");
        if (paths.empty()) {
            *error = "no chunk1.mlmodelc, chunk2.mlmodelc, ... in " + path;
            return nullptr;
        }
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits          = sd_coreml::compute_units_from_env();
        NSMutableArray<MLModel*>* stages = [NSMutableArray arrayWithCapacity:paths.size()];
        NSMutableSet<NSString*>* available = [NSMutableSet setWithArray:@[ @"input_ids", @"key_mask" ]];
        for (const auto& p : paths) {
            NSURL* url = sd_coreml::compiled_model_url(p, "coreml te");
            if (!url) {
                *error = "could not compile " + p;
                return nullptr;
            }
            NSError* ns_error = nil;
            MLModel* model    = [MLModel modelWithContentsOfURL:url configuration:config error:&ns_error];
            if (!model) {
                *error = p + ": " + (ns_error ? ns_error.localizedDescription.UTF8String : "load failed");
                return nullptr;
            }
            for (NSString* name in model.modelDescription.inputDescriptionsByName) {
                if (![available containsObject:name]) {
                    *error = p + " takes '" + name.UTF8String + "', which no earlier chunk provides";
                    return nullptr;
                }
            }
            [available addObjectsFromArray:model.modelDescription.outputDescriptionsByName.allKeys];
            [stages addObject:model];
        }
        std::vector<int64_t> ids_shape =
            sd_coreml::declared_shape(stages.firstObject.modelDescription.inputDescriptionsByName[@"input_ids"]);
        if (ids_shape.size() != 2 || ids_shape[0] != 1) {
            *error = "the first chunk must take input_ids [1, L]";
            return nullptr;
        }
        auto* encoder   = new CoreMLTextEncoder();
        encoder->stages = stages;
        encoder->length = ids_shape[1];
        LOG_INFO("coreml te: %s ready in %.2f s, %zu chunk(s), %lld tokens, compute units %s", path.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), paths.size(),
                 (long long)encoder->length, sd_coreml::compute_units_name(config.computeUnits));
        return encoder;
    }
}

void coreml_te_close(CoreMLTextEncoder* encoder) {
    delete encoder;
}

int64_t coreml_te_length(const CoreMLTextEncoder* encoder) {
    return encoder ? encoder->length : 0;
}

bool coreml_te_encode(CoreMLTextEncoder* encoder,
                      const std::vector<int32_t>& tokens,
                      const std::vector<float>& mask,
                      const std::vector<std::string>& taps,
                      std::vector<float>& out,
                      int64_t* width,
                      std::string* error) {
    @autoreleasepool {
        if (!encoder || (int64_t)tokens.size() != encoder->length || mask.size() != tokens.size()) {
            *error = "coreml te: expected " + std::to_string(encoder ? encoder->length : 0) + " tokens and mask values";
            return false;
        }
        std::lock_guard<std::mutex> lock(encoder->mutex);
        const int64_t L   = encoder->length;
        NSError* ns_error = nil;
        MLMultiArray* ids = [[MLMultiArray alloc] initWithShape:@[ @1, @(L) ] dataType:MLMultiArrayDataTypeInt32 error:&ns_error];
        MLMultiArray* km  = [[MLMultiArray alloc] initWithShape:@[ @1, @(L) ] dataType:MLMultiArrayDataTypeFloat16 error:&ns_error];
        if (!ids || !km) {
            *error = "coreml te: could not allocate inputs";
            return false;
        }
        std::memcpy(ids.dataPointer, tokens.data(), L * sizeof(int32_t));
        auto* km_data = static_cast<_Float16*>(km.dataPointer);
        for (int64_t i = 0; i < L; ++i)
            km_data[i] = static_cast<_Float16>(mask[i]);

        NSMutableDictionary<NSString*, MLFeatureValue*>* values = [NSMutableDictionary dictionary];
        values[@"input_ids"] = [MLFeatureValue featureValueWithMultiArray:ids];
        values[@"key_mask"]  = [MLFeatureValue featureValueWithMultiArray:km];
        for (MLModel* stage in encoder->stages) {
            NSMutableDictionary<NSString*, MLFeatureValue*>* feed = [NSMutableDictionary dictionary];
            for (NSString* name in stage.modelDescription.inputDescriptionsByName)
                feed[name] = values[name];
            MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&ns_error];
            id<MLFeatureProvider> result          = provider ? [stage predictionFromFeatures:provider error:&ns_error] : nil;
            if (!result) {
                *error = std::string("coreml te: ") + (ns_error ? ns_error.localizedDescription.UTF8String : "prediction failed");
                return false;
            }
            for (NSString* name in result.featureNames)
                values[name] = [result featureValueForName:name];
        }

        std::vector<std::vector<float>> outputs;
        std::vector<int64_t> widths;
        int64_t total = 0;
        for (const auto& tap : taps) {
            MLMultiArray* array = values[@(tap.c_str())].multiArrayValue;
            if (!array || array.shape.count != 3 || array.shape[1].longLongValue != L) {
                *error = "coreml te: no [1, L, W] output named " + tap;
                return false;
            }
            const int64_t w = array.shape[2].longLongValue;
            outputs.emplace_back(L * w);
            if (!sd_coreml::copy_output(array, L * w, outputs.back().data(), "coreml te")) {
                *error = "coreml te: unsupported output type for " + tap;
                return false;
            }
            widths.push_back(w);
            total += w;
        }
        out.assign(L * total, 0.0f);
        for (int64_t t = 0; t < L; ++t) {
            int64_t offset = 0;
            for (size_t k = 0; k < outputs.size(); ++k) {
                std::memcpy(&out[t * total + offset], &outputs[k][t * widths[k]], widths[k] * sizeof(float));
                offset += widths[k];
            }
        }
        *width = total;
        return true;
    }
}
