cmake_minimum_required (VERSION 3.18)

project(Samples)

set(CMAKE_CXX_FLAGS "-std=c++14 -O2")
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_PREFIX_PATH
    /opt/triton_client/
    /opt/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /opt/triton_client/include
)
link_directories(
    /opt/triton_client/lib
    )


add_executable(client main.cpp)
target_link_libraries(client PRIVATE
    grpcclient
    ${OpenCV_LIBS}
    -lpthread
    )
