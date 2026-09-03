include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)
add_subdirectory(fit-params)
add_subdirectory(server)
if(NOT SD_BUILD_SHARED_LIBS)
    # ABot validation harnesses link engine internals not exported by the shared library.
    add_subdirectory(abot-parity)
    add_subdirectory(abot-session)
    add_subdirectory(abot-walk)
endif()
