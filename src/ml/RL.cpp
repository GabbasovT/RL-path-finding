#include "ml/RL.hpp"
#include <torch/script.h>
#include <algorithm>
#include <numeric>
#include <ranges>

namespace rl {

// ReplayBuffer
ReplayBuffer::ReplayBuffer(size_t capacity) : capacity_(capacity), rng(std::random_device{}()) {}

void ReplayBuffer::push(const Transition& t) {
    if (buffer.size() >= capacity_) buffer.pop_front();
    buffer.push_back(t);
}

std::vector<Transition> ReplayBuffer::sample(size_t batch_size) {
    std::vector<Transition> batch;
    std::shuffle(buffer.begin(), buffer.end(), rng);
    batch_size = std::min(batch_size, buffer.size());
    batch.assign(buffer.begin(), buffer.begin() + batch_size);
    return batch;
}

size_t ReplayBuffer::size() const { return buffer.size(); }

// Actor Network
ActorNetImpl::ActorNetImpl() : fc1(TOTAL_OBS_SIZE, 256), fc2(256, 256), fc3(256, ACT_SIZE) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);

    torch::nn::init::xavier_uniform_(fc1->weight);
    torch::nn::init::constant_(fc1->bias, 0.1);
    torch::nn::init::xavier_uniform_(fc2->weight);
    torch::nn::init::constant_(fc2->bias, 0.1);
    torch::nn::init::xavier_uniform_(fc3->weight);
    torch::nn::init::constant_(fc3->bias, 0.1);
}

torch::Tensor ActorNetImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return torch::tanh(fc3->forward(x));
}

void ActorNetImpl::copy_weights(const ActorNetImpl& source) {
    torch::NoGradGuard no_grad;  // Отключаем градиенты при копировании весов
    for (const auto& p : named_parameters()) {
        p.value().copy_(source.named_parameters()[p.key()]);
    }
}

// Critic Network
CriticNetImpl::CriticNetImpl() : fc1(TOTAL_OBS_SIZE + ACT_SIZE, 256), fc2(256, 256), fc3(256, 1) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);

    torch::nn::init::xavier_uniform_(fc1->weight);
    torch::nn::init::constant_(fc1->bias, 0.1);
    torch::nn::init::xavier_uniform_(fc2->weight);
    torch::nn::init::constant_(fc2->bias, 0.1);
    torch::nn::init::xavier_uniform_(fc3->weight);
    torch::nn::init::constant_(fc3->bias, 0.1);
}

torch::Tensor CriticNetImpl::forward(torch::Tensor state, torch::Tensor action) {
    auto x = torch::cat({state, action}, 1);
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return fc3->forward(x);
}

void CriticNetImpl::copy_weights(const CriticNetImpl& source) {
    torch::NoGradGuard no_grad;  // Отключаем градиенты при копировании весов
    for (const auto& p : named_parameters()) {
        p.value().copy_(source.named_parameters()[p.key()]);
    }
}

// TD3 Agent
TD3Agent::TD3Agent(float actor_lr, float critic_lr, float gamma_, float tau_, float max_distance_)
    : actor(std::make_shared<ActorNetImpl>()),
      actor_target(std::make_shared<ActorNetImpl>()),
      critic1(std::make_shared<CriticNetImpl>()),
      critic2(std::make_shared<CriticNetImpl>()),
      critic1_target(std::make_shared<CriticNetImpl>()),
      critic2_target(std::make_shared<CriticNetImpl>()),
      actor_optimizer(actor->parameters(), actor_lr),
      critic1_optimizer(critic1->parameters(), critic_lr),
      critic2_optimizer(critic2->parameters(), critic_lr),
      gamma(gamma_), tau(tau_), max_distance(max_distance_), update_step(0), policy_delay(2) {

    actor_target->copy_weights(*actor);
    critic1_target->copy_weights(*critic1);
    critic2_target->copy_weights(*critic2);
}

torch::Tensor TD3Agent::preprocess_state(const project::common::State& state) {
    std::vector<float> obs_data;
    obs_data.reserve(TOTAL_OBS_SIZE);

    // Основные наблюдения (лучи)
    obs_data.insert(obs_data.end(), state.obs.begin(), state.obs.end());

    // Направление к цели (нормализованный вектор)
    obs_data.push_back(state.direction_to_goal.first);
    obs_data.push_back(state.direction_to_goal.second);

    // Нормализованная дистанция [0, 1]
    obs_data.push_back(state.distance_to_goal / max_distance);

    return torch::tensor(obs_data, torch::kFloat32).reshape({1, TOTAL_OBS_SIZE});
}

std::pair<torch::Tensor, torch::Tensor> TD3Agent::select_action(torch::Tensor state, float noise_std) {
    actor->eval();
    torch::NoGradGuard no_grad;
    auto action = actor->forward(state);

    if (noise_std > 0.0f) {
        auto noise = torch::randn_like(action) * noise_std;
        noise = noise.clamp(-0.5f, 0.5f);
        action = (action + noise).clamp(-1.0f, 1.0f);
    }

    actor->train();
    return {action, action.norm(2, 1, true)};
}

void TD3Agent::soft_update(torch::nn::Module& target, const torch::nn::Module& source) {
    torch::NoGradGuard no_grad;  // Отключаем градиенты при in-place обновлении весов
    for (const auto& tp : target.named_parameters()) {
        const auto& sp = source.named_parameters()[tp.key()];
        tp.value().data().mul_(1.0f - tau).add_(sp.data(), tau);
    }
}

void TD3Agent::update(ReplayBuffer& buffer, int batch_size) {
    if (buffer.size() < batch_size) return;

    auto batch = buffer.sample(batch_size);

    // Подготовка батча
    std::vector<torch::Tensor> states, actions, rewards, next_states, dones;
    for (const auto& t : batch) {
        states.push_back(t.state);
        // Гарантируем, что action не требует градиента
        actions.push_back(t.action.detach());
        rewards.push_back(t.reward);
        next_states.push_back(t.next_state);
        dones.push_back(t.done);
    }

    auto state_batch = torch::cat(states, 0);       // [batch_size, obs_dim]
    auto action_batch = torch::cat(actions, 0);     // [batch_size, action_dim]
    auto reward_batch = torch::cat(rewards, 0).squeeze(-1);  // [batch_size]
    auto next_state_batch = torch::cat(next_states, 0);     // [batch_size, obs_dim]
    auto done_batch = torch::cat(dones, 0).squeeze(-1);     // [batch_size]

    // Обновление критиков
    torch::Tensor next_action_noise = torch::randn_like(action_batch) * 0.2f;
    next_action_noise = next_action_noise.clamp(-0.5f, 0.5f);

    auto next_actions = actor_target->forward(next_state_batch) + next_action_noise;
    next_actions = next_actions.clamp(-1.0f, 1.0f);

    auto target_q1 = critic1_target->forward(next_state_batch, next_actions);
    auto target_q2 = critic2_target->forward(next_state_batch, next_actions);
    auto target_q = torch::min(target_q1, target_q2);
    auto target_value = reward_batch + gamma * (1.0f - done_batch) * target_q.squeeze();

    auto current_q1 = critic1->forward(state_batch, action_batch).squeeze();
    auto current_q2 = critic2->forward(state_batch, action_batch).squeeze();

    auto critic1_loss = torch::mse_loss(current_q1, target_value.detach());
    auto critic2_loss = torch::mse_loss(current_q2, target_value.detach());

    critic1_optimizer.zero_grad();
    critic1_loss.backward();
    torch::nn::utils::clip_grad_norm_(critic1->parameters(), 1.0);
    critic1_optimizer.step();

    critic2_optimizer.zero_grad();
    critic2_loss.backward();
    torch::nn::utils::clip_grad_norm_(critic2->parameters(), 1.0);
    critic2_optimizer.step();

    // Обновление актора (с задержкой)
    if (++update_step % policy_delay == 0) {
        auto actor_loss = -critic1->forward(state_batch, actor->forward(state_batch)).mean();

        actor_optimizer.zero_grad();
        actor_loss.backward();
        torch::nn::utils::clip_grad_norm_(actor->parameters(), 1.0);
        actor_optimizer.step();

        // Мягкое обновление целевых сетей
        soft_update(*actor_target, *actor);
        soft_update(*critic1_target, *critic1);
        soft_update(*critic2_target, *critic2);
    }
}

}  // namespace rl
