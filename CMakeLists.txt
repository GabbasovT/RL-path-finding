cmake_minimum_required(VERSION 3.10)
project(RLPathFinding)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/environment
        ${CMAKE_SOURCE_DIR}/src/common
)

add_subdirectory(src/environment)
add_subdirectory(src/ml)

add_executable(main src/main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES} environment ml)
