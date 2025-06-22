cmake_minimum_required(VERSION 3.15)
project(RLPathFinding)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)
find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/environment
        ${CMAKE_SOURCE_DIR}/include/ml
        ${CMAKE_SOURCE_DIR}/src/common
)

add_subdirectory(src/environment)
add_subdirectory(src/ml)

add_executable(${PROJECT_NAME}
        src/main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        environment
        ml
        ${TORCH_LIBRARIES}
        sfml-graphics
        sfml-window
        sfml-system
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET environment PROPERTY CXX_STANDARD 20)
set_property(TARGET ml PROPERTY CXX_STANDARD 20)