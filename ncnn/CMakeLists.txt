CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(segment)

set(CMAKE_CXX_FLAGS "-std=c++14 -O2")


set (ncnn_DIR ${NCNN_ROOT}/lib/cmake/ncnn)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED)


add_executable(segment segment.cpp)
target_include_directories(segment PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(segment ${OpenCV_LIBRARIES} ncnn OpenMP::OpenMP_CXX)
