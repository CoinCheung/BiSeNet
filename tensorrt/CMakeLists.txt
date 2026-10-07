CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

# Before project(): enabling CUDA caches nvcc's default arch.
# Real SASS for sm_70-90, plus sm_90 PTX for newer GPUs.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set (CMAKE_CUDA_ARCHITECTURES
         70-real 75-real 80-real 86-real 89-real 90-real 90-virtual)
endif ()

PROJECT(segment LANGUAGES CUDA CXX)

# set(CMAKE_CXX_FLAGS "-std=c++17 -O2")

message (STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set (CMAKE_BUILD_TYPE Release)
set (CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")

# link_directories(${PROJECT_SOURCE_DIR}/build ${PROJECT_SOURCE_DIR}/build/plugins)
# include_directories(/root/build/TensorRT-8.2.5.1/include)
# link_directories(/root/build/TensorRT-8.2.5.1/lib)


enable_language (CUDA)
find_package (OpenCV REQUIRED)

include (CheckLanguage)
check_language (CUDA)
check_language (OpenCV)

# CustomArgMax plugin, superseded by fused_head (sources kept).
# add_subdirectory(./plugins/)
add_subdirectory(./plugins/fused_head/)  # libfused_head_plugin.so, for trtexec --staticPlugins


# Compiled in, not linked: the plugin self-registers with no referenced symbol,
# so --as-needed would drop a shared library.
add_executable(segment segment.cu read_img.cpp trt_dep.cu
    plugins/fused_head/fused_up_argmax_plugin.cu
)
target_compile_features(segment PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(segment PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(segment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plugins/fused_head)
target_link_libraries(segment ${OpenCV_LIBRARIES}
    nvinfer nvinfer_plugin nvonnxparser
)

