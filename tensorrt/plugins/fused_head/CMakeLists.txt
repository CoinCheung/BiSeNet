cmake_minimum_required(VERSION 3.22)
# Default all archs; -DCMAKE_CUDA_ARCHITECTURES or a parent project overrides.
# Before project(), which would otherwise cache nvcc's default.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70-real 75-real 80-real 86-real 89-real 90-real 90-virtual)
endif()
project(fused_head LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

add_library(fused_head_plugin SHARED fused_up_argmax_plugin.cu)
target_compile_features(fused_head_plugin PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(fused_head_plugin PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/include/x86_64-linux-gnu)
target_link_libraries(fused_head_plugin PRIVATE nvinfer)
set_target_properties(fused_head_plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON)
