cmake_minimum_required(VERSION 3.15)
project(RLPathFinding)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

# Find dependencies
find_package(Torch REQUIRED)
find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)

# Include directories
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/environment
        ${CMAKE_SOURCE_DIR}/include/ml
        ${CMAKE_SOURCE_DIR}/src/common
)

# Add subdirectories
add_subdirectory(src/environment)
add_subdirectory(src/ml)

# Main executable
add_executable(${PROJECT_NAME}
        src/main.cpp
)

# Link libraries to main executable
target_link_libraries(${PROJECT_NAME} PRIVATE
        environment
        ml
        ${TORCH_LIBRARIES}
        sfml-graphics
        sfml-window
        sfml-system
)

# Enable C++20 for all targets
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET environment PROPERTY CXX_STANDARD 20)
set_property(TARGET ml PROPERTY CXX_STANDARD 20)