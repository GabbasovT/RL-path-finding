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

add_executable(main
        src/main.cpp
        src/ml/RL.cpp
        src/environment/Env.cpp
        src/environment/Renderer.cpp
)

target_link_libraries(main PRIVATE
        ${TORCH_LIBRARIES}
        environment
        ml
        sfml-graphics
        sfml-window
        sfml-system
)
