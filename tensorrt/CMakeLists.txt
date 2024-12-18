CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

PROJECT(segment LANGUAGES CUDA CXX)

# set(CMAKE_CXX_FLAGS "-std=c++17 -O2")

set (CMAKE_BUILD_TYPE Release)
set (CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set (CMAKE_CUDA_ARCHITECTURES 90)
endif ()

# link_directories(${PROJECT_SOURCE_DIR}/build ${PROJECT_SOURCE_DIR}/build/plugins)
# include_directories(/root/build/TensorRT-8.2.5.1/include)
# link_directories(/root/build/TensorRT-8.2.5.1/lib)


enable_language (CUDA)
find_package (OpenCV REQUIRED)

include (CheckLanguage)
check_language (CUDA)
check_language (OpenCV)

add_subdirectory(./plugins/) # custom_plugin


add_executable(segment segment.cu read_img.cpp trt_dep.cu)
target_compile_features(segment PRIVATE cxx_std_17 cuda_std_14)
target_include_directories(segment PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(segment ${OpenCV_LIBRARIES}
    nvinfer nvinfer_plugin nvonnxparser custom_plugin
)



## deprecated
# link_directories(/usr/local/cuda/lib64)
# find_package(CUDA REQUIRED)
# cuda_add_library(kernels STATIC kernels.cu)
# target_include_directories(
#     segment PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(
#     segment -lnvinfer -lnvinfer_plugin -lnvonnxparser
#     ${CUDA_LIBRARIES} ${CUDART_LIB} ${NVINFER_LIB}
#     ${OpenCV_LIBRARIES})


