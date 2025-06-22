#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <numeric>
#include "ml/RL.hpp"
#include "environment/Env.hpp"

using namespace project::common;
using namespace rl;

int main() {
    // Параметры среды
    const float WORLD_WIDTH = 100.0f;
    const float WORLD_HEIGHT = 100.0f;
    const float MAX_DISTANCE = std::sqrt(WORLD_WIDTH * WORLD_WIDTH + WORLD_HEIGHT * WORLD_HEIGHT);

    // Параметры обучения
    const int EPISODES = 1000;
    const int MAX_STEPS = 500;
    const int BATCH_SIZE = 128;
    const int LOG_INTERVAL = 50;
    const float ACTOR_LR = 3e-4;
    const float CRITIC_LR = 3e-4;
    const float GAMMA = 0.99f;
    const float TAU = 0.005f;

    // Инициализация среды
    project::env::Agent init_agent(WORLD_WIDTH/2, WORLD_HEIGHT/2);
    project::env::Goal goal(10.0f, 10.0f, 5.0f, 5.0f); // x, y, w, h

    std::vector<project::env::Box> obstacles = {
        project::env::Box(30.0f, 30.0f, 10.0f, 10.0f),
        project::env::Box(70.0f, 70.0f, 15.0f, 15.0f)
    };

    project::env::Environment env(
        obstacles,
        goal,
        init_agent,
        0.0f, 0.0f, // bord_x0, bord_y0
        WORLD_WIDTH, WORLD_HEIGHT // bord_x1, bord_y1
    );

    // Инициализация агента
    TD3Agent agent(ACTOR_LR, CRITIC_LR, GAMMA, TAU, MAX_DISTANCE);
    ReplayBuffer buffer(100000);

    // Статистика
    std::vector<float> episode_rewards;
    int success_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    for (int ep = 0; ep < EPISODES; ++ep) {
        State s = env.reset();
        auto state = agent.preprocess_state(s);
        float ep_reward = 0.0f;
        bool episode_success = false;

        // Параметры исследования (затухающий шум)
        float noise_std = std::max(0.05f, 0.3f * (1.0f - ep / float(EPISODES)));

        for (int t = 0; t < MAX_STEPS; ++t) {
            // Выбор действия
            auto [action_tensor, _] = agent.select_action(state, noise_std);
            auto action_data = action_tensor.squeeze().data_ptr<float>();
            Action action{{action_data[0], action_data[1]}, 1.0f};

            // Шаг среды
            State s2 = env.do_action(action);
            if (s2.env_type == EnvState::TERMINAL) {
                std::cout << t << " " << s2.distance_to_goal << " " << s2.direction_to_goal.first << " " << s2.direction_to_goal.second << std::endl;
            }
            auto next_state = agent.preprocess_state(s2);

            // Награда
            float reward = -0.1f; // Штраф за каждый шаг
            bool done = false;

            switch(s2.env_type) {
                case EnvState::TERMINAL:
                    reward = 10.0f + 5.0f * (1.0f - s2.distance_to_goal/MAX_DISTANCE);
                    done = true;
                    episode_success = true;
                    break;
                case EnvState::COLLISION:
                    reward = -10.0f - 2.0f * s2.distance_to_goal/MAX_DISTANCE;
                    done = true;
                    break;
                case EnvState::TIMEOUT:
                    reward = -5.0f;
                    done = true;
                    break;
                default:
                    // Поощрение за движение к цели
                    reward += 0.5f * (s.distance_to_goal - s2.distance_to_goal)/MAX_DISTANCE;
            }

            // Сохраняем переход
            buffer.push({
                state,
                action_tensor,
                torch::tensor({reward}, torch::kFloat32),
                next_state,
                torch::tensor({done ? 1.0f : 0.0f}, torch::kFloat32)
            });

            // Обучение
            agent.update(buffer, BATCH_SIZE);

            // Обновление состояния
            state = next_state;
            ep_reward += reward;
            if (done) break;
        }

        // Статистика
        if (episode_success) success_count++;
        episode_rewards.push_back(ep_reward);

        // Логирование
        if (ep % LOG_INTERVAL == 0 && ep > 0) {
            auto avg_reward = std::accumulate(
                episode_rewards.end() - LOG_INTERVAL,
                episode_rewards.end(), 0.0f) / LOG_INTERVAL;

            auto success_rate = success_count * 100.0f / LOG_INTERVAL;
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