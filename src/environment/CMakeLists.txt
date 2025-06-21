add_library(environment Env.cpp Renderer.cpp)
target_link_libraries(environment ${TORCH_LIBRARIES})
