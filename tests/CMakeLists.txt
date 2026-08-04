add_executable(test-ltx-rope test-ltx-rope.cpp)
target_link_libraries(test-ltx-rope PRIVATE stable-diffusion)
add_test(NAME test-ltx-rope COMMAND test-ltx-rope)
