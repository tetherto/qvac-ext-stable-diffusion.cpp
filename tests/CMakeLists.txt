include_directories("${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/thirdparty")
add_executable(test-ltx-rope test-ltx-rope.cpp)
target_link_libraries(test-ltx-rope PRIVATE stable-diffusion)
add_test(NAME test-ltx-rope COMMAND test-ltx-rope)

add_executable(test-vae-routing test-vae-routing.cpp)
target_link_libraries(test-vae-routing PRIVATE stable-diffusion)
add_test(NAME test-vae-routing COMMAND test-vae-routing)

add_executable(test-fit-params test-fit-params.cpp)
target_include_directories(test-fit-params PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-fit-params PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME test-fit-params COMMAND test-fit-params)

add_executable(test-upscaler-fit test_upscaler_fit.cpp)
target_link_libraries(test-upscaler-fit PRIVATE stable-diffusion)
add_test(NAME test-upscaler-fit COMMAND test-upscaler-fit)

add_executable(test-abot-fit test_abot_fit.cpp)
target_link_libraries(test-abot-fit PRIVATE stable-diffusion)
add_test(NAME test-abot-fit COMMAND test-abot-fit)

add_executable(test-safetensors-metadata test_safetensors_metadata.cpp)
target_include_directories(test-safetensors-metadata PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-safetensors-metadata PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME test-safetensors-metadata COMMAND test-safetensors-metadata)

add_executable(test-ltx-vae-temporal test-ltx-vae-temporal.cpp)
target_link_libraries(test-ltx-vae-temporal PRIVATE stable-diffusion)
add_test(NAME test-ltx-vae-temporal COMMAND test-ltx-vae-temporal)

add_executable(test-ltx-vae-decoder-streaming
    test-ltx-vae-decoder-streaming.cpp)
target_link_libraries(test-ltx-vae-decoder-streaming
    PRIVATE stable-diffusion)
add_test(NAME test-ltx-vae-decoder-streaming
    COMMAND test-ltx-vae-decoder-streaming)

add_executable(test-ltx-conditioning test-ltx-conditioning.cpp)
target_link_libraries(test-ltx-conditioning PRIVATE stable-diffusion)
add_test(NAME test-ltx-conditioning COMMAND test-ltx-conditioning)

add_executable(test-lora-validation test-lora-validation.cpp)
target_link_libraries(test-lora-validation PRIVATE stable-diffusion)
add_test(NAME test-lora-validation COMMAND test-lora-validation)

add_executable(test-common-json
    test-common-json.cpp
    ../examples/common/common.cpp
    ../examples/common/log.cpp
    ../examples/common/media_io.cpp
    ../examples/common/media_metadata.cpp
    ../examples/common/generation_metadata.cpp)
target_include_directories(test-common-json PRIVATE
    "${PROJECT_SOURCE_DIR}/examples"
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-common-json
    PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME test-common-json COMMAND test-common-json)

add_executable(test-gguf-comfy-shape test-gguf-comfy-shape.cpp)
target_include_directories(test-gguf-comfy-shape PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-gguf-comfy-shape PRIVATE stable-diffusion)
add_test(NAME test-gguf-comfy-shape COMMAND test-gguf-comfy-shape)

add_executable(test-abot-streaming test-abot-streaming.cpp)
target_link_libraries(test-abot-streaming PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME test-abot-streaming COMMAND test-abot-streaming)

add_executable(test-vae-name-conversion test-vae-name-conversion.cpp)
target_include_directories(test-vae-name-conversion PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-vae-name-conversion PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME test-vae-name-conversion COMMAND test-vae-name-conversion)

add_executable(test-safetensors-convrot test-safetensors-convrot.cpp)
target_include_directories(test-safetensors-convrot PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-safetensors-convrot PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME test-safetensors-convrot COMMAND test-safetensors-convrot)

add_executable(test-qwen-image-21 test_qwen_image_21.cpp)
target_include_directories(test-qwen-image-21 PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-qwen-image-21 PRIVATE stable-diffusion)
add_test(NAME test-qwen-image-21 COMMAND test-qwen-image-21)

add_executable(test-h3-localops test-h3-localops.cpp)
target_include_directories(test-h3-localops PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-h3-localops PRIVATE stable-diffusion)
add_test(NAME test-h3-localops COMMAND test-h3-localops)

add_executable(test-group-norm-row-norm test-group-norm-row-norm.cpp)
target_link_libraries(test-group-norm-row-norm PRIVATE stable-diffusion)
add_test(NAME test-group-norm-row-norm COMMAND test-group-norm-row-norm)

add_executable(test-conv-subsample test-conv-subsample.cpp)
target_link_libraries(test-conv-subsample PRIVATE stable-diffusion)
add_test(NAME test-conv-subsample COMMAND test-conv-subsample)

# CoreML sidecars. Needs fixtures from tests/coreml/make_fixtures.py (coremltools), so it is not an add_test:
#   python tests/coreml/make_fixtures.py build/coreml-fixtures && build/bin/test-coreml build/coreml-fixtures
if(SD_COREML AND NOT SD_BUILD_SHARED_LIBS)
    add_executable(test-coreml coreml/test_coreml.cpp)
    target_include_directories(test-coreml PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/core")
    target_link_libraries(test-coreml PRIVATE stable-diffusion)
    set_target_properties(test-coreml PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
