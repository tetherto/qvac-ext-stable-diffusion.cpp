cmake_minimum_required(VERSION 3.25)

# Resolve the Bare addon CMake helpers (installed by `npm i` / `bare-make`).
find_package(cmake-bare REQUIRED PATHS node_modules/cmake-bare)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(ltx2_bare C)

# Path to the qvac-ext-stable-diffusion.cpp checkout. Defaults to the parent of
# this addon dir (bare/ lives inside the repo). Override with -DSD_ROOT=... if the
# addon is vendored elsewhere.
set(SD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "qvac-ext-stable-diffusion.cpp root")

# Build the stable-diffusion library (with LTX-2 support) but skip its examples.
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${SD_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/sd_build" EXCLUDE_FROM_ALL)

add_bare_module(ltx2_bare)

target_sources(
  ${ltx2_bare}
  PRIVATE
    binding.c
)

target_include_directories(
  ${ltx2_bare}
  PRIVATE
    "${SD_ROOT}/include"
)

# Link the stable-diffusion library target. NOTE: confirm the exact target name
# exported by the repo's top-level CMakeLists (commonly `stable-diffusion`).
target_link_libraries(
  ${ltx2_bare}
  PUBLIC
    stable-diffusion
)
