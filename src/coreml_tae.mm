#include "coreml_tae.h"

#include <algorithm>
#include <chrono>

#include "coreml_utils.h"

using namespace sd_coreml;

namespace {

class CoreMLTAEDecoderImpl : public CoreMLTAEDecoder {
public:
    MLModel* model = nil;
    std::vector<int64_t> latent_shape;
    std::vector<int64_t> image_shape;

    int64_t scale() const override { return image_shape[2] / latent_shape[2]; }

    bool decode(const float* latent, const int64_t shape[4], float* image, const int64_t out_shape[4]) override {
        @autoreleasepool {
            if (!std::equal(latent_shape.begin(), latent_shape.end(), shape)) {
                LOG_ERROR("coreml tae: latent %s does not match the model's %s",
                          shape_str(shape, 4).c_str(), shape_str(latent_shape.data(), latent_shape.size()).c_str());
                return false;
            }
            if (!std::equal(image_shape.begin(), image_shape.end(), out_shape)) {
                LOG_ERROR("coreml tae: image buffer %s does not match the model's output %s",
                          shape_str(out_shape, 4).c_str(), shape_str(image_shape.data(), image_shape.size()).c_str());
                return false;
            }

            NSError* error  = nil;
            MLMultiArray* z = wrap(latent, latent_shape, &error);
            if (z == nil) {
                LOG_ERROR("coreml tae: failed to wrap the latent: %s", error.localizedDescription.UTF8String);
                return false;
            }
            MLDictionaryFeatureProvider* provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"latent" : [MLFeatureValue featureValueWithMultiArray:z]}
                                                                  error:&error];
            if (provider == nil) {
                LOG_ERROR("coreml tae: %s", error.localizedDescription.UTF8String);
                return false;
            }

            auto t0                          = std::chrono::steady_clock::now();
            id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
            auto t1                          = std::chrono::steady_clock::now();
            if (prediction == nil) {
                LOG_ERROR("coreml tae: prediction failed: %s", error.localizedDescription.UTF8String);
                return false;
            }
            LOG_DEBUG("coreml tae: prediction took %.2f ms", std::chrono::duration<double, std::milli>(t1 - t0).count());

            MLMultiArray* result = [prediction featureValueForName:@"image"].multiArrayValue;
            if (result == nil) {
                LOG_ERROR("coreml tae: model has no 'image' output");
                return false;
            }
            int64_t n = 1;
            for (int64_t d : image_shape) {
                n *= d;
            }
            return copy_output(result, n, image, "coreml tae");
        }
    }
};

}  // namespace

std::unique_ptr<CoreMLTAEDecoder> CoreMLTAEDecoder::load(const std::string& path) {
    @autoreleasepool {
        auto t0    = std::chrono::steady_clock::now();
        NSURL* url = compiled_model_url(path, "coreml tae");
        if (url == nil) {
            return nullptr;
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits          = compute_units_from_env();
        NSError* error               = nil;
        MLModel* model               = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (model == nil) {
            LOG_ERROR("coreml tae: failed to load %s: %s", path.c_str(), error.localizedDescription.UTF8String);
            return nullptr;
        }

        auto impl          = std::make_unique<CoreMLTAEDecoderImpl>();
        impl->model        = model;
        impl->latent_shape = declared_shape(model.modelDescription.inputDescriptionsByName[@"latent"]);
        impl->image_shape  = declared_shape(model.modelDescription.outputDescriptionsByName[@"image"]);
        if (impl->latent_shape.size() != 4 || impl->image_shape.size() != 4) {
            LOG_ERROR("coreml tae: %s does not have a fixed-shape 'latent' input and 'image' output", path.c_str());
            return nullptr;
        }
        const std::vector<int64_t>& z = impl->latent_shape;
        const int64_t s               = z[2] > 0 ? impl->image_shape[2] / z[2] : 0;
        const std::vector<int64_t> expected = {z[0], 3, s * z[2], s * z[3]};
        if ((s != 8 && s != 16) || impl->image_shape != expected) {
            LOG_ERROR("coreml tae: %s maps latent %s to image %s, expected 8x or 16x the latent size with 3 channels",
                      path.c_str(), shape_str(z.data(), 4).c_str(), shape_str(impl->image_shape.data(), 4).c_str());
            return nullptr;
        }
        LOG_INFO("coreml tae: %s ready in %.2f s, compute units %s, latent %s, image %s",
                 path.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 compute_units_name(config.computeUnits),
                 shape_str(impl->latent_shape.data(), 4).c_str(),
                 shape_str(impl->image_shape.data(), 4).c_str());
        return impl;
    }
}
