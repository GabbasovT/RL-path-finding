#ifndef REN_H
#define REN_H

#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include "Env.hpp"

namespace project::ren{

class DynamicRectangles {
    float width_;
    float height_;
    sf::RectangleShape background_;
    std::vector<sf::VertexArray> staticRects_;
    std::vector<sf::VertexArray> dynamicRects_;

    void addRectangle(const sf::FloatRect& rect, const sf::Color& color, std::vector<sf::VertexArray>& target);
    void addStaticRect(const sf::FloatRect& rect, const sf::Color& color);
    void addDynamicRect(const sf::FloatRect& rect, const sf::Color& color);
public:
    DynamicRectangles(project::env::Environment& env);
    void updateAgent(project::env::Agent& agent);
    void draw(sf::RenderTarget& target) const;
};

}

#endif
