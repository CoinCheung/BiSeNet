CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(segment)

set(CMAKE_CXX_FLAGS "-std=c++14 -O1")


link_directories(/usr/local/cuda/lib64)


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(segment segment.cpp trt_dep.cpp)
target_include_directories(
    segment PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
    segment -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    )
