add_executable(test-ltx-rope test-ltx-rope.cpp)
target_link_libraries(test-ltx-rope PRIVATE stable-diffusion)
add_test(NAME test-ltx-rope COMMAND test-ltx-rope)

add_executable(test-vae-routing test-vae-routing.cpp)
target_link_libraries(test-vae-routing PRIVATE stable-diffusion)
add_test(NAME test-vae-routing COMMAND test-vae-routing)
