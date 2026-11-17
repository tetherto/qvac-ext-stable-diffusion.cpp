#ifndef __COREML_UTILS_H__
#define __COREML_UTILS_H__

// Helpers shared by the CoreML sidecars (coreml_unet.mm, coreml_tae.mm). Objective-C++ only.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "util.h"

namespace sd_coreml {

// SD_COREML_COMPUTE_UNITS: all | cpu_ne (default) | cpu_gpu | cpu.
inline MLComputeUnits compute_units_from_env() {
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

inline const char* compute_units_name(MLComputeUnits units) {
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

inline std::string shape_str(const int64_t* shape, size_t n) {
    std::string s = "[";
    for (size_t i = 0; i < n; i++) {
        s += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return s + "]";
}

// Shape of a fixed-size multiarray input or output, or empty if the model does not declare one.
inline std::vector<int64_t> declared_shape(MLFeatureDescription* desc) {
    std::vector<int64_t> shape;
    if (desc == nil || desc.multiArrayConstraint == nil) {
        return shape;
    }
    for (NSNumber* d in desc.multiArrayConstraint.shape) {
        shape.push_back(d.longLongValue);
    }
    return shape;
}

// The compiled model to load: an .mlmodelc as is, anything else compiled first (slow, done on every load).
inline NSURL* compiled_model_url(const std::string& path, const char* tag) {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if ([url.pathExtension isEqualToString:@"mlmodelc"]) {
        return url;
    }
    LOG_INFO("%s: compiling %s (ship a .mlmodelc to skip this step)", tag, path.c_str());
    NSError* error = nil;
    NSURL* out     = [MLModel compileModelAtURL:url error:&error];
    if (out == nil) {
        LOG_ERROR("%s: failed to compile %s: %s", tag, path.c_str(), error.localizedDescription.UTF8String);
    }
    return out;
}

// Wraps contiguous float32 memory as a multiarray without copying.
inline MLMultiArray* wrap(const float* data, const std::vector<int64_t>& shape, NSError** error) {
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

// The data type a model declares for a multiarray input (float32 when it declares none).
inline MLMultiArrayDataType declared_type(MLFeatureDescription* desc) {
    return desc && desc.multiArrayConstraint ? desc.multiArrayConstraint.dataType : MLMultiArrayDataTypeFloat32;
}

// An input in the type the model declares: float32 memory is wrapped without copying, float16 is converted.
inline MLMultiArray* make_input(const float* data, const std::vector<int64_t>& shape, MLMultiArrayDataType type, NSError** error) {
    if (type != MLMultiArrayDataTypeFloat16) {
        return wrap(data, shape, error);
    }
    NSMutableArray<NSNumber*>* dims = [NSMutableArray array];
    int64_t n                       = 1;
    for (int64_t d : shape) {
        [dims addObject:@(d)];
        n *= d;
    }
    MLMultiArray* array = [[MLMultiArray alloc] initWithShape:dims dataType:MLMultiArrayDataTypeFloat16 error:error];
    if (array != nil) {
        [array getMutableBytesWithHandler:^(void* bytes, NSInteger size, NSArray<NSNumber*>* strides) {
            (void)size;
            (void)strides;  // a freshly allocated array is contiguous
            _Float16* dst = (_Float16*)bytes;
            for (int64_t i = 0; i < n; i++) {
                dst[i] = (_Float16)data[i];
            }
        }];
    }
    return array;
}

// Copies a model output into contiguous float32 memory. The output may be float16 and may have
// padded strides, so it is read element by element in logical order.
inline bool copy_output(MLMultiArray* result, int64_t n, float* out, const char* tag) {
    if (result.count != n) {
        LOG_ERROR("%s: output has %lld elements, expected %lld", tag, (long long)result.count, (long long)n);
        return false;
    }
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
        LOG_ERROR("%s: unsupported output data type", tag);
    }
    return ok;
}

}  // namespace sd_coreml

#endif  // __COREML_UTILS_H__
