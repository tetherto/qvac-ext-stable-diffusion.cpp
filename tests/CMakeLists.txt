add_executable(test-ltx-rope test-ltx-rope.cpp)
target_link_libraries(test-ltx-rope PRIVATE stable-diffusion)
add_test(NAME test-ltx-rope COMMAND test-ltx-rope)

add_executable(test-vae-routing test-vae-routing.cpp)
target_link_libraries(test-vae-routing PRIVATE stable-diffusion)
add_test(NAME test-vae-routing COMMAND test-vae-routing)

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
    ../examples/common/media_io.cpp)
target_include_directories(test-common-json PRIVATE
    "${PROJECT_SOURCE_DIR}/examples"
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test-common-json
    PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME test-common-json COMMAND test-common-json)
