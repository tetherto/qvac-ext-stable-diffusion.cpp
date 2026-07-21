include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)
add_subdirectory(server)
if(NOT SD_BUILD_SHARED_LIBS)
    # ABot walk validation harnesses link against engine internals
    # (ModelLoader, SDBackendManager, the runner classes), which are not
    # exported from the shared library — static builds only.
    add_subdirectory(abot-parity)
    add_subdirectory(abot-session)
    add_subdirectory(abot-walk)
endif()