#pragma once

#include <torch/torch.h>
#include <deque>
#include <vector>
#include <random>
#include <utility>
#include "Enums.hpp"
#include "Consts.hpp"
#include "Environment.hpp"

namespace rl {

    constexpr int OBS_SIZE = project::common::SIZE_OF_ARRAY_OF_OBSERVATIONS;
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
        torch::nn::Linear fc1, fc2, fc3;
        ActorNetImpl();
        torch::Tensor forward(torch::Tensor x);
    };
    TORCH_MODULE(ActorNet);

    struct CriticNetImpl : torch::nn::Module {
        torch::nn::Linear fc1, fc2, fc3;
        CriticNetImpl();
        torch::Tensor forward(torch::Tensor state, torch::Tensor action);
    };
    TORCH_MODULE(CriticNet);

    class TD3Agent {
    public:
        TD3Agent(float actor_lr, float critic_lr, float gamma, float tau);
        std::pair<torch::Tensor, torch::Tensor> select_action(torch::Tensor state, float noise_std = 0.1f);
        void update(ReplayBuffer& buffer, int batch_size);

        ActorNet actor;

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
    };

}
