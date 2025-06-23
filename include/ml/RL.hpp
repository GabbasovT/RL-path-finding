#pragma once

#include <torch/torch.h>
#include <deque>
#include <vector>
#include <random>
#include <utility>
#include "Consts.hpp"
#include "Enums.hpp"
#include "environment/Env.hpp"

namespace rl {
    constexpr int BASE_OBS_SIZE = project::common::SIZE_OF_ARRAY_OF_OBSERVATIONS;
    constexpr int EXTRA_OBS_SIZE = 3;
    constexpr int TOTAL_OBS_SIZE = BASE_OBS_SIZE + EXTRA_OBS_SIZE;
    constexpr int ACT_SIZE = 2;

    struct Transition {
        torch::Tensor state;
        torch::Tensor action;
        torch::Tensor reward;
        torch::Tensor next_state;
        torch::Tensor done;
    };

    class ReplayBuffer {
    public:
        ReplayBuffer(size_t capacity);
        void push(const Transition& transition);
        std::vector<Transition> sample(size_t batch_size);
        size_t size() const;

    private:
        std::deque<Transition> buffer;
        size_t capacity_;
        std::mt19937 rng;
    };

    struct ActorNetImpl : torch::nn::Module {
        torch::nn::Linear fc1, fc2, fc3, fc4, fc5;
        ActorNetImpl();
        torch::Tensor forward(torch::Tensor x);
        void copy_weights(const ActorNetImpl& source);
    };
    TORCH_MODULE(ActorNet);

    struct CriticNetImpl : torch::nn::Module {
        torch::nn::Linear fc1, fc2, fc3, fc4, fc5;
        CriticNetImpl();
        torch::Tensor forward(torch::Tensor state, torch::Tensor action);
        void copy_weights(const CriticNetImpl& source);
    };
    TORCH_MODULE(CriticNet);

    class TD3Agent {
    public:
        TD3Agent(float actor_lr, float critic_lr, float gamma, float tau, float max_distance);
        std::pair<torch::Tensor, torch::Tensor> select_action(torch::Tensor state, float noise_std = 0.1f);
        void update(ReplayBuffer& buffer, int batch_size);
        void save_model(const std::string& actor_path, const std::string& critic1_path, const std::string& critic2_path);
        void load_model(const std::string& actor_path, const std::string& critic1_path, const std::string& critic2_path);
        void set_eval_mode(bool eval);
        torch::Tensor preprocess_state(const project::common::State& state);

        ActorNet actor;
        float max_distance;

    private:
        ActorNet actor_target;
        CriticNet critic1, critic2;
        CriticNet critic1_target, critic2_target;

        torch::optim::Adam actor_optimizer;
        torch::optim::Adam critic1_optimizer;
        torch::optim::Adam critic2_optimizer;

        float gamma;
        float tau;
        int policy_delay = 2;
        int update_step = 0;

        void soft_update(torch::nn::Module& target, const torch::nn::Module& source);
    };
}