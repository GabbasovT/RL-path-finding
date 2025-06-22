#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include "Env.hpp"


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

DynamicRectangles::DynamicRectangles(float w, float h, std::vector<project::env::Box> &objects_, env::Goal &goal, env::Agent &agent) : width_(w), height_(h){
    background_.setFillColor(sf::Color::White);
    background_.setSize({width_, height_});
    for (project::env::Box o : objects_) {
        std::pair<float, float> corn = o.get_left_top();
        std::pair<float, float> w_h = o.get_w_h();
        addStaticRect({corn.first, corn.second, w_h.first, w_h.second}, sf::Color::Grey);
    }
    std::pair<float, float> g_corn = goal.get_left_top();
    std::pair<float, float> g_w_h = goal.get_w_h();
    addStaticRect({g_corn.first, g_corn.second, g_w_h.first, g_w_h.second}, sf::Color::Yellow);
    addDynamicRect({agent.get_coords().first - 1, agent.get_coords().second + 1, 2, 2}, sf::Color::Green);
}

void DynamicRectangles::shiftDinamic(float shift) {
    for (auto &e : dynamicRects_) {
        for (int i = 0; i < 4; i++) {
            e[i].position.x += shift;
        }
    }
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