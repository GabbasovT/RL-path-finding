#ifndef ENV_H
#define ENV_H

#include <utility>
#include <array>
#include <vector>
#include <cmath>
#include "Consts.hpp"
#include "Types.hpp"
#include "Enums.hpp"

namespace project::env{

class Agent {
    float x, y;
    std::array<std::pair<float, float>, project::common::SIZE_OF_ARRAY_OF_OBSERVATIONS> rdrs;
    
    float size;
public:
    Agent(float x, float y);
    void shift(float u, float v);
    std::pair<float, float> get_coords();
    std::array<float, project::common::SIZE_OF_ARRAY_OF_OBSERVATIONS> launch_rays(std::vector<Object> &objects_);
};

class Object {
private:
    float x, y;
public:
    void set_coords(float n_x, float n_y);
    std::pair<float, float> get_coords();
    bool check_colision(float o_x, float o_y);
    virtual float get_intersect(float o_x, float o_y, std::pair<float, float> n_ray);
};

class Box : public Object {
    float w, h;
public:
    Box(float x, float y, float w, float h);
    bool check_colision(float o_x, float o_y);
    float get_intersect(float o_x, float o_y, std::pair<float, float> n_ray) override;
};

class Goal : public Box {
public:
    std::pair<float, float> get_dir(float o_x, float o_y);
    float Goal::get_dist(float o_x, float o_y);
};

class Environment {
    struct Data{
        std::vector<Object> objects_;
        Goal goal;
        Agent agent;
        float bord_x0, bord_y0;
        float bord_x1, bord_y1;
        float dtime = 0;
    };
    Data cur;
    Data backup;
public:
    Environment(std::vector<Object> objects_, Goal goal, Agent agent, 
        float bord_x0, float bord_y0, float bord_x1, float bord_y1);

    project::common::State do_action(project::common::Action action);
    void reset();
};

}
#endif
