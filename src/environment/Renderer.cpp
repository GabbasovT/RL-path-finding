#include <SFML/Graphics.hpp>
#include <vector>
#include "Types.hpp"
#include "Consts.hpp"
#include "Env.hpp"
#include "Renderer.hpp"

#include <ATen/core/interned_strings.h>

namespace project::ren{

void DynamicRectangles::addRectangle(const sf::FloatRect& rect, const sf::Color& color, std::vector<sf::VertexArray>& target) {
    sf::VertexArray va(sf::Quads, 4);
    
    va[0].position = sf::Vector2f(rect.left, rect.top);
    va[1].position = sf::Vector2f(rect.left + rect.width, rect.top);
    va[2].position = sf::Vector2f(rect.left + rect.width, rect.top + rect.height);
    va[3].position = sf::Vector2f(rect.left, rect.top + rect.height);
    
    for (int i = 0; i < 4; ++i) {
        va[i].color = color;
    }
    
    target.push_back(va);
}

void DynamicRectangles::addStaticRect(const sf::FloatRect& rect, const sf::Color& color) {
    addRectangle(rect, color, staticRects_);
}

void DynamicRectangles::addDynamicRect(const sf::FloatRect& rect, const sf::Color& color) {
    addRectangle(rect, color, dynamicRects_);
}

void DynamicRectangles::addAgentRect(const sf::FloatRect& rect, const sf::Color& color) {
    addRectangle(rect, color, agentRects_);
}

DynamicRectangles::DynamicRectangles(project::env::Environment &env, bool addInters) {
    withInters = addInters;
    width_ = env.get_w_h().first;
    height_ = env.get_w_h().second;

    background_.setFillColor(sf::Color::White);
    background_.setSize({width_, height_});

    for (env::Box o : *env.get_objects()) {
        std::pair<float, float> corn = o.get_right_bottom();
        std::pair<float, float> w_h = o.get_w_h();
        addStaticRect({corn.first - w_h.first, corn.second, w_h.first, w_h.second}, sf::Color::Blue);
    }

    std::pair<float, float> g_corn = env.get_goal()->get_right_bottom();
    std::pair<float, float> g_w_h = env.get_goal()->get_w_h();
    addStaticRect({g_corn.first - g_w_h.first, g_corn.second, g_w_h.first, g_w_h.second}, sf::Color::Yellow);
    addAgentRect({env.get_agent()->get_coords().first - 1, env.get_agent()->get_coords().second + 1, 2, 2}, sf::Color::Green);
    if (withInters) {
        for (int i = 0; i < project::common::SIZE_OF_ARRAY_OF_OBSERVATIONS; i++) {
            addDynamicRect({env.get_agent()->get_coords().first - 0.25f, env.get_agent()->get_coords().second + 0.25f, 0.5, 0.5}, sf::Color::Cyan);
        } 
    }
}



void DynamicRectangles::updateAgent(project::env::Agent* agent) {
    float n_x = agent->get_coords().first;
    float n_y = agent->get_coords().second;

    agentRects_[0][0].position.x = n_x - agentR;
    agentRects_[0][0].position.y = n_y + agentR;
    agentRects_[0][1].position.x = n_x + agentR;
    agentRects_[0][1].position.y = n_y + agentR;
    agentRects_[0][2].position.x = n_x + agentR;
    agentRects_[0][2].position.y = n_y - agentR;
    agentRects_[0][3].position.x = n_x - agentR;
    agentRects_[0][3].position.y = n_y - agentR;
}

void DynamicRectangles::updateInters(project::common::State* state) {
    if (withInters) {
        for (int i = 0; i < state->obs_intersect.size(); i++) {
            std::pair<float, float> p = state->obs_intersect[i];
            float n_x = p.first;
            float n_y = p.second;

            dynamicRects_[i][0].position.x = n_x - intersR;
            dynamicRects_[i][0].position.y = n_y + intersR;
            dynamicRects_[i][1].position.x = n_x + intersR;
            dynamicRects_[i][1].position.y = n_y + intersR;
            dynamicRects_[i][2].position.x = n_x + intersR;
            dynamicRects_[i][2].position.y = n_y - intersR;
            dynamicRects_[i][3].position.x = n_x - intersR;
            dynamicRects_[i][3].position.y = n_y - intersR;
        }
    }
}

void DynamicRectangles::draw(sf::RenderTarget& target) const {
    target.draw(background_);
    
    for (const auto& va : staticRects_) {
        target.draw(va);
    }
    for (const auto& va : agentRects_) {
        target.draw(va);
    }
    for (const auto& va : dynamicRects_) {
        target.draw(va);
    }
}

}
