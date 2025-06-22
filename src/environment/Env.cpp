#include "Env.hpp"
#include <utility>
#include <array>
#include <vector>
#include <cmath>
#include "Consts.hpp"

namespace project::env{

    float Object::get_intersect(float o_x, float o_y, std::pair<float, float> n_ray) {
        return -1.0f;
    }


float euclid(float a, float b) {
    return std::pow(std::pow(a, 2) + std::pow(b, 2), 0.5);
}

Agent::Agent(float x, float y) {
    this->x = x;
    this->y = y;

    float phi = 2 * acos(-1) / rdrs.size();
    float theta = 0.0;
    for (int i = 0; i < rdrs.size(); i++) {
        rdrs[i] = {cos(theta), sin(theta)};
        theta += phi;
    }
}
void Agent::shift(float u, float v) {
    this->x += u;
    this->y += v;
}
std::pair<float, float> Agent::get_coords() {
    return {this->x, this->y};
}
std::array<float, common::SIZE_OF_ARRAY_OF_OBSERVATIONS> Agent::launch_rays(
    std::vector<Box> &objects_, std::array<std::pair<float, float>, common::SIZE_OF_ARRAY_OF_OBSERVATIONS> &inters
) {
    std::array<float, common::SIZE_OF_ARRAY_OF_OBSERVATIONS> res;
    for (int i = 0; i < res.size(); i++) {
        res[i] = 0;
    }
    for (int i = 0; i < rdrs.size(); i++) {
        float d = INFINITY;
        for (Box &o : objects_) {
            float t = o.get_intersect(x, y, rdrs[i]);
            if (t < 0) {
                continue;
            }
            d = std::min(d, t);
        }
        res[i] = d;
        inters[i] = {x + rdrs[i].first * d, y + rdrs[i].second * d};
    }
    return res;
}

void Object::set_coords(float n_x, float n_y) {
    this->x = n_x;
    this->y = n_y;
}
std::pair<float, float> Object::get_coords() {
    return {this->x, this->y};
}

bool Object::check_colision(float o_x, float o_y) {
    // Базовая реализация (можно считать, что объект - точка, коллизия не происходит)
    return false;
}

Box::Box(float x, float y, float w, float h) {
    set_coords(x, y);
    this->w = w;
    this->h = h;
}
std::pair<float, float> Box::get_right_bottom() {
    float x = get_coords().first;
    float y = get_coords().second;
    return {x + w/2, y - h/2};
}
std::pair<float, float> Box::get_w_h() {
    return {w, h};
}
bool Box::check_colision(float o_x, float o_y) {
    float x = get_coords().first;
    float y = get_coords().second;
    if ((o_x > x - w/2) && (o_x < x + w/2) && (o_y > y - h/2) && (o_y < y + h/2)) {
        return true;
    }
    return false;
}
float Box::get_intersect(float o_x, float o_y, std::pair<float, float> n_ray) {
    float x = get_coords().first;
    float y = get_coords().second;
    if (std::abs(n_ray.second) <= 0.000001) {
        if (std::abs(o_y - y) < h/2) {
            return std::min(std::abs(o_x - (x - w/2)), std::abs(o_x - (x + w/2)));
        } else {
            return -1;
        }
    } else {
        float k_y;
        k_y = n_ray.first / n_ray.second;
        std::pair<float, float> c1 = {x - w/2, (1 / k_y) * (x - w/2 - o_x) + o_y};
        std::pair<float, float> c2 = {x + w/2, (1 / k_y) * (x + w/2 - o_x) + o_y};
        std::pair<float, float> c3 = {k_y * (y + h/2 - o_y) + o_x, y + h/2};
        std::pair<float, float> c4 = {k_y * (y - h/2 - o_y) + o_x, y - h/2};
        float res = INFINITY;
        float t;
        if (std::abs(c1.second - y) <= h/2 && std::signbit(c1.second - o_y) == std::signbit(n_ray.second)) {
            t = euclid(c1.first - o_x, c1.second - o_y);
            if (res > t) {
                res = t;
            }
        }
        if (std::abs(c2.second - y) <= h/2 && std::signbit(c2.second - o_y) == std::signbit(n_ray.second)) {
            t = euclid(c2.first - o_x, c2.second - o_y);
            if (res > t) {
                res = t;
            }
        }
        if (std::abs(c3.first - x) <= w/2 && std::signbit(c3.second - o_y) == std::signbit(n_ray.second)) {
            t = euclid(c3.first - o_x, c3.second - o_y);
            if (res > t) {
                res = t;
            }
        }
        if (std::abs(c4.first - x) <= w/2 && std::signbit(c4.second - o_y) == std::signbit(n_ray.second)) {
            t = euclid(c4.first - o_x, c4.second - o_y);
            if (res > t) {
                res = t;
            }
        }
        if (res >= INFINITY - 1) {
            return -1;
        }
        return res;
    }
}

std::pair<float, float> Goal::get_dir(float o_x, float o_y) {
    float x = get_coords().first;
    float y = get_coords().second;
    float norm = euclid(x - o_x, y - o_y);
    return {(o_x - x) / norm, (o_y - y) / norm};
}
float Goal::get_dist(float o_x, float o_y) {
    float x = get_coords().first;
    float y = get_coords().second;
    float norm = euclid(x - o_x, y - o_y);
    return norm;
}

Environment::Environment(std::vector<Box> objects_, Goal goal, Agent agent,
        float bord_x0, float bord_y0, float bord_x1, float bord_y1) :
        cur{
            std::move(objects_),
            std::move(goal),
            std::move(agent),
            bord_x0, bord_y0, bord_x1, bord_y1
        },
        backup{ cur }
{
    // Добавление граничных Box-ов (стен)
    cur.objects_.push_back(Box(
                            (bord_x0 + bord_x1) / 2,
                            bord_y1,
                            std::abs(bord_x0 - bord_x1) * 1.1,
                            std::abs(bord_y0 - bord_y1) / 10
                        ));
    cur.objects_.push_back(Box(
                            (bord_x0 + bord_x1) / 2,
                            bord_y0,
                            std::abs(bord_x0 - bord_x1) * 1.1,
                            std::abs(bord_y0 - bord_y1) / 10
                        ));
    cur.objects_.push_back(Box(
                            bord_x0,
                            (bord_y0 + bord_y1) / 2,
                            std::abs(bord_x0 - bord_x1) / 10,
                            std::abs(bord_y0 - bord_y1) * 1.1
                        ));
    cur.objects_.push_back(Box(
                            bord_x1,
                            (bord_y0 + bord_y1) / 2,
                            std::abs(bord_x0 - bord_x1) / 10,
                            std::abs(bord_y0 - bord_y1) * 1.1
                        ));
    backup = cur;
}   

Goal* Environment::get_goal() {
    return &cur.goal;
}
Agent* Environment::get_agent() {
    return &cur.agent;
}
std::vector<Box>* Environment::get_objects() {
    return &cur.objects_;
}
std::pair<float, float> Environment::get_w_h() {
    return {cur.bord_x1 - cur.bord_x1, cur.bord_y1 - cur.bord_y0};
}

common::State Environment::reset() {
    this->cur = this->backup;
    common::Action start;
    start.dir = {0.0, 0.0};
    start.len = 0.0;
    return do_action(start);
}

common::State Environment::do_action(common::Action action) const {
    common::State st;
    cur.agent.shift(action.dir.first * action.len, action.dir.second * action.len);
    st.obs = cur.agent.launch_rays(cur.objects_, st.obs_intersect);
    std::pair<float, float> a_xy = cur.agent.get_coords();
    st.direction_to_goal = cur.goal.get_dir(a_xy.first, a_xy.second);
    st.distance_to_goal = cur.goal.get_dist(a_xy.first, a_xy.second);
    for (Object &o : cur.objects_) {
        if (o.check_colision(a_xy.first, a_xy.second)) {
            st.env_type = common::EnvState::COLLISION;
            return st;
        }
    }
    if (cur.goal.check_colision(a_xy.first, a_xy.second)) {
        st.env_type = common::EnvState::TERMINAL;
        return st;
    }
    st.env_type = common::EnvState::NONE;
    return st;
}

}
