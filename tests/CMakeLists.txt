add_executable(test-ltx-rope test-ltx-rope.cpp)
target_link_libraries(test-ltx-rope PRIVATE stable-diffusion)
add_test(NAME test-ltx-rope COMMAND test-ltx-rope)

add_executable(test-vae-routing test-vae-routing.cpp)
target_link_libraries(test-vae-routing PRIVATE stable-diffusion)
add_test(NAME test-vae-routing COMMAND test-vae-routing)

add_executable(test-ltx-vae-temporal test-ltx-vae-temporal.cpp)
target_link_libraries(test-ltx-vae-temporal PRIVATE stable-diffusion)
add_test(NAME test-ltx-vae-temporal COMMAND test-ltx-vae-temporal)
