# Install script for directory: /Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/_build_cli/thirdparty/libwebm/libwebm.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libwebm.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libwebm.a")
    execute_process(COMMAND "/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libwebm.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/webm" TYPE FILE FILES
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/buffer_reader.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/callback.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/dom_types.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/element.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/file_reader.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/id.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/istream_reader.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/reader.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/status.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/webm_parser/include/webm/webm_parser.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/webm/common" TYPE FILE FILES "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/common/webmids.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/webm/mkvmuxer" TYPE FILE FILES
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/mkvmuxer/mkvmuxer.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/mkvmuxer/mkvmuxertypes.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/mkvmuxer/mkvmuxerutil.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/mkvmuxer/mkvwriter.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/webm/mkvparser" TYPE FILE FILES
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/mkvparser/mkvparser.h"
    "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/thirdparty/libwebm/mkvparser/mkvreader.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/user030/Documents/qvac-ext-stable-diffusion.cpp/_build_cli/thirdparty/libwebm/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
