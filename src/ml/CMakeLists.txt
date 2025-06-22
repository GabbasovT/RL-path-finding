add_library(ml
        RL.cpp
)

target_include_directories(ml PRIVATE
        ${CMAKE_SOURCE_DIR}/include/ml
        ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(ml PRIVATE
        ${TORCH_LIBRARIES}
)