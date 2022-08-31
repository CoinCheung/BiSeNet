# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tutorialrecommendedbackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because recommended backend
# doesn't use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

find_package (OpenCV REQUIRED)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_recommended.ldscript libtriton_recommended.ldscript COPYONLY)

add_library(
  triton-self_backend-backend SHARED
  src/recommended.cc
)

add_library(
  TutorialRecommendedBackend::triton-self_backend-backend ALIAS triton-self_backend-backend
)

target_include_directories(
  triton-self_backend-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(triton-self_backend-backend PRIVATE cxx_std_14)
target_compile_options(
  triton-self_backend-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

target_link_libraries(
  triton-self_backend-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ${OpenCV_LIBS}
)

if(WIN32)
  set_target_properties(
    triton-self_backend-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_self_backend
  )
else()
  set_target_properties(
    triton-self_backend-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_self_backend
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_recommended.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_recommended.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TutorialRecommendedBackend)

install(
  TARGETS
    triton-self_backend-backend
  EXPORT
    triton-self_backend-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/self_backend
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/self_backend
)

install(
  EXPORT
    triton-self_backend-backend-targets
  FILE
    TutorialRecommendedBackendTargets.cmake
  NAMESPACE
    TutorialRecommendedBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TutorialRecommendedBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TutorialRecommendedBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TutorialRecommendedBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-self_backend-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TutorialRecommendedBackendTargets.cmake
  NAMESPACE TutorialRecommendedBackend::
)

export(PACKAGE TutorialRecommendedBackend)
