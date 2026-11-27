# Standalone benchmark for the Metal backend and the CoreML UNet sidecar (docs/coreml.md).
# macOS: builds the sd-bench command-line tool.
# iOS:   builds sdbench.app, which runs either sd-cli or sd-bench with the launch arguments.
cmake_minimum_required(VERSION 3.21)
project(sdbench C CXX OBJC OBJCXX)

set(SD_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(SDBENCH_TEAM "" CACHE STRING "Apple development team ID used to sign the iOS app")
set(SDBENCH_BUNDLE_ID "io.tether.sdbench" CACHE STRING "bundle identifier of the iOS app")

set(SD_METAL ON CACHE BOOL "" FORCE)
set(SD_COREML ON CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${SD_ROOT} sd)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_executable(sd-bench bench.cpp)
    target_include_directories(sd-bench PRIVATE ${SD_ROOT}/thirdparty)
    target_link_libraries(sd-bench PRIVATE stable-diffusion)
    target_compile_features(sd-bench PRIVATE cxx_std_17)
    return()
endif()

if(NOT SDBENCH_TEAM)
    message(FATAL_ERROR "set -DSDBENCH_TEAM=<team id> to sign the iOS app")
endif()

# sd-cli's main is renamed so the app can dispatch to it.
set_source_files_properties(${SD_ROOT}/examples/cli/main.cpp PROPERTIES COMPILE_DEFINITIONS "main=sd_cli_main")
set_source_files_properties(bench.cpp PROPERTIES COMPILE_DEFINITIONS "SDBENCH_NO_MAIN")
# The same sources as examples/cli/CMakeLists.txt, without WebP and WebM.
add_executable(sdbench MACOSX_BUNDLE ios_main.mm bench.cpp
    ${SD_ROOT}/examples/cli/main.cpp
    ${SD_ROOT}/examples/cli/image_metadata.cpp
    ${SD_ROOT}/examples/common/common.cpp
    ${SD_ROOT}/examples/common/generation_metadata.cpp
    ${SD_ROOT}/examples/common/log.cpp
    ${SD_ROOT}/examples/common/media_io.cpp
    ${SD_ROOT}/examples/common/media_metadata.cpp)
target_include_directories(sdbench PRIVATE ${SD_ROOT}/examples ${SD_ROOT}/src ${SD_ROOT}/thirdparty)
target_link_libraries(sdbench PRIVATE stable-diffusion zip "-framework Foundation" "-framework Metal" "-framework UIKit")
target_compile_features(sdbench PRIVATE cxx_std_17)
set_target_properties(sdbench PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER ${SDBENCH_BUNDLE_ID}
    MACOSX_BUNDLE_BUNDLE_NAME sdbench
    MACOSX_BUNDLE_BUNDLE_VERSION 1
    MACOSX_BUNDLE_SHORT_VERSION_STRING 1.0
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${SDBENCH_BUNDLE_ID}
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
    XCODE_ATTRIBUTE_DEVELOPMENT_TEAM ${SDBENCH_TEAM}
    XCODE_ATTRIBUTE_CODE_SIGN_ENTITLEMENTS ${CMAKE_CURRENT_SOURCE_DIR}/sdbench.entitlements
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET 17.0
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")
