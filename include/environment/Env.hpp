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

    class Object;

    class Box;

    class Agent {
        float x, y;
        std::array<std::pair<float, float>, common::SIZE_OF_ARRAY_OF_OBSERVATIONS> rdrs;

        float size;
    public:
        Agent(float x, float y);
        void shift(float u, float v);
        std::pair<float, float> get_coords();
        std::array<float, common::SIZE_OF_ARRAY_OF_OBSERVATIONS> launch_rays(std::vector<Box>& objects_);
    };

    class Object {
    private:
        float x, y;
    public:
        void set_coords(float n_x, float n_y);
        std::pair<float, float> get_coords();
        virtual bool check_colision(float o_x, float o_y);              // Сделали виртуальным
        virtual float get_intersect(float o_x, float o_y, std::pair<float, float> n_ray);
        virtual ~Object() = default;                                     // Виртуальный деструктор
    };

    class Box : public Object {
        float w, h;
    public:
        Box(float x, float y, float w, float h);
        std::pair<float, float> get_left_top();
        std::pair<float, float> get_w_h();
        bool check_colision(float o_x, float o_y) override;
        float get_intersect(float o_x, float o_y, std::pair<float, float> n_ray) override;
    };

    class Goal : public Box {
    public:
        std::pair<float, float> get_dir(float o_x, float o_y);
        Goal(float x, float y, float w, float h)
            : Box(x, y, w, h) {}
        float get_dist(float o_x, float o_y);
    };

    class Environment {
        struct Data{
            std::vector<Box> objects_;
            Goal goal;
            Agent agent;
            float bord_x0, bord_y0;
            float bord_x1, bord_y1;
            float dtime = 0;
        };
        Data cur;
        Data backup;
    public:
        Environment(std::vector<Box> objects_, Goal goal, Agent agent,
            float bord_x0, float bord_y0, float bord_x1, float bord_y1);

        Goal* get_goal();
        Agent* get_agent();
        std::vector<Box>* get_objects();
        std::pair<float, float> get_w_h();

        common::State do_action(common::Action action);
        common::State reset();
    };

}

#endif
