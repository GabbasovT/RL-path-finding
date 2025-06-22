#include <SFML/Graphics.hpp>
#include <vector>
#include "Env.hpp"
#include "Renderer.hpp"

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

DynamicRectangles::DynamicRectangles(project::env::Environment &env) {
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
    addDynamicRect({env.get_agent()->get_coords().first - 1, env.get_agent()->get_coords().second + 1, 2, 2}, sf::Color::Green);
}

void DynamicRectangles::updateAgent(project::env::Agent* agent) {
    float n_x = agent->get_coords().first;
    float n_y = agent->get_coords().second;

    dynamicRects_[0][0].position.x = n_x - 1;
    dynamicRects_[0][0].position.y = n_y + 1;
    dynamicRects_[0][1].position.x = n_x + 1;
    dynamicRects_[0][1].position.y = n_y + 1;
    dynamicRects_[0][2].position.x = n_x + 1;
    dynamicRects_[0][2].position.y = n_y - 1;
    dynamicRects_[0][3].position.x = n_x - 1;
    dynamicRects_[0][3].position.y = n_y - 1;
}

void DynamicRectangles::draw(sf::RenderTarget& target) const {
    target.draw(background_);
    
    for (const auto& va : staticRects_) {
        target.draw(va);
    }
    
    for (const auto& va : dynamicRects_) {
        target.draw(va);
    }
}

}
