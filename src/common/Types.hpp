#pragma once
#include <utility>
#include <array>
#include "Enums.hpp"
#include "Consts.hpp"

namespace project::common {

struct State {
    std::array<float, SIZE_OF_ARRAY_OF_OBSERVATIONS> obs;
    std::array<std::pair<float, float>, SIZE_OF_ARRAY_OF_OBSERVATIONS> obs_intersect;
    std::pair<float, float> direction_to_goal; 
    float distance_to_goal;
    EnvState env_type;
};

struct Action {
    std::pair<float, float> dir;
    float len;                  
};

}
