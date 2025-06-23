#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <numeric>
#include "Renderer.hpp"

#include "ml/RL.hpp"
#include "environment/Env.hpp"
#include "Config.h"

using namespace project::common;
using namespace rl;

int main() {
    const float MAX_DISTANCE = std::sqrt(project::config::WORLD_WIDTH * project::config::WORLD_WIDTH 
                                        + project::config::WORLD_HEIGHT * project::config::WORLD_HEIGHT);

    project::env::Environment env = project::config::env;

    sf::RenderWindow window(sf::VideoMode(project::config::WORLD_WIDTH, project::config::WORLD_HEIGHT), "RL-path-finding");
    window.setSize(sf::Vector2u(1000, 1000));
    project::ren::DynamicRectangles manager(env, true);

    TD3Agent agent(project::config::ACTOR_LR, project::config::CRITIC_LR, 
                    project::config::GAMMA, project::config::TAU, MAX_DISTANCE);
    ReplayBuffer buffer(300000);

    std::vector<float> episode_rewards;
    int success_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    for (int ep = 0; ep < project::config::EPISODES; ++ep) {
        State s = env.reset();
        manager = project::ren::DynamicRectangles(env, true);
        auto state = agent.preprocess_state(s);
        float ep_reward = 0.0f;
        bool episode_success = false;

        float noise_std = 0.5f * (1.0f - ep / 8000.0f);
        if (noise_std < 0.05f) noise_std = 0.05f;

        for (int t = 0; t < project::config::MAX_STEPS; ++t) {
            auto [action_tensor, _] = agent.select_action(state, noise_std);
            auto action_data = action_tensor.squeeze().data_ptr<float>();
            Action action{{action_data[0], action_data[1]}, 1.0f};

            State s2 = env.do_action(action);
            if (window.isOpen()) {
                sf::Event event;
                while (window.pollEvent(event)) {
                    if (event.type == sf::Event::Closed)
                        window.close();
                }

                manager.updateAgent(env.get_agent());
                manager.updateInters(&s2);

                window.clear();
                manager.draw(window);
                window.display();
            }
            auto next_state = agent.preprocess_state(s2);

            float reward = -0.001f;
            bool done = false;

            switch (s2.env_type) {
                case EnvState::TERMINAL:
                    reward = 500.0f;
                    done = true;
                    episode_success = true;
                    break;
                case EnvState::COLLISION:
                    reward = -100.0f;
                    done = true;
                    break;
                case EnvState::TIMEOUT:
                    reward = -5.0f;
                    done = true;
                    break;
                default:
                    float progress = (s.distance_to_goal - s2.distance_to_goal) / MAX_DISTANCE;
                    reward += 30.0f * progress;
                    if (progress > 0.0f) reward += MAX_DISTANCE *
                        (1.0f / s2.distance_to_goal - 1.0f / s.distance_to_goal);
            }

            buffer.push({
                state,
                action_tensor,
                torch::tensor({reward}, torch::kFloat32),
                next_state,
                torch::tensor({done ? 1.0f : 0.0f}, torch::kFloat32)
            });

            if (buffer.size() > project::config::TRAIN_START_SIZE && t % project::config::TRAIN_INTERVAL == 0) {
                agent.update(buffer, project::config::BATCH_SIZE);
            }

            state = next_state;
            ep_reward += reward;
            if (done) break;
        }

        if (episode_success) success_count++;
        episode_rewards.push_back(ep_reward);

        if (ep % project::config::LOG_INTERVAL == 0 && ep > 0) {
            auto avg_reward = std::accumulate(
                episode_rewards.end() - project::config::LOG_INTERVAL,
                episode_rewards.end(), 0.0f) / project::config::LOG_INTERVAL;

            auto success_rate = success_count * 100.0f / project::config::LOG_INTERVAL;
            success_count = 0;

            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

            std::cout << "Episode " << ep
                      << " | Avg Reward: " << avg_reward
                      << " | Success: " << success_rate << "%"
                      << " | Time: " << elapsed << "s"
                      << " | Noise: " << noise_std
                      << " | Buffer: " << buffer.size() << std::endl;
        }
    }

    return 0;
}