#ifndef REN_H
#define REN_H

#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include "Env.hpp"

namespace project::ren{

class DynamicRectangles {
    bool withInters = false;
    float intersR = 0.25f;
    float agentR = 1f;
    float width_;
    float height_;
    sf::RectangleShape background_;
    std::vector<sf::VertexArray> staticRects_;
    std::vector<sf::VertexArray> agentRects_;
    std::vector<sf::VertexArray> dynamicRects_;
    void addRectangle(const sf::FloatRect& rect, const sf::Color& color, std::vector<sf::VertexArray>& target);
    void addStaticRect(const sf::FloatRect& rect, const sf::Color& color);
    void addAgentRect(const sf::FloatRect& rect, const sf::Color& color);
    void addDynamicRect(const sf::FloatRect& rect, const sf::Color& color);
public:
    DynamicRectangles(const project::env::Environment& env, bool addInters);
    void updateAgent(const project::env::Agent* agent);
    void updateInters(const project::common::State* state);
    void draw(sf::RenderTarget& target) const;
};

}

#endif
