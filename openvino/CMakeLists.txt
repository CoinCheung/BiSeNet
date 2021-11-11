
cmake_minimum_required (VERSION 3.10)

cmake_policy(SET CMP0025 NEW)

project(Samples)


set (CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized -Winit-self")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")


find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(segment main.cpp)
target_link_libraries(
    segment
    ${InferenceEngine_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${OpenCV_LIBS}
)



