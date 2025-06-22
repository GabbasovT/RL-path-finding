add_library(environment
        Env.cpp
        Renderer.cpp
)

target_include_directories(environment PRIVATE
        ${CMAKE_SOURCE_DIR}/include/environment
        ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(environment PRIVATE
        ${TORCH_LIBRARIES}
        sfml-graphics
        sfml-window
        sfml-system
)