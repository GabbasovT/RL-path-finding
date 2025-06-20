#include "ml/RL.hpp"

namespace rl {

ReplayBuffer::ReplayBuffer(size_t capacity) : capacity_(capacity), rng(std::random_device{}()) {}

void ReplayBuffer::push(const Transition& t) {
    if (buffer.size() >= capacity_) buffer.pop_front();
    buffer.push_back(t);
}

std::vector<Transition> ReplayBuffer::sample(size_t batch_size) {
    std::vector<Transition> batch;
    std::uniform_int_distribution<> dist(0, buffer.size() - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(buffer[dist(rng)]);
    }
    return batch;
}

size_t ReplayBuffer::size() const {
    return buffer.size();
}

ActorNetImpl::ActorNetImpl()
    : fc1(OBS_SIZE, 128), fc2(128, 128), fc3(128, ACT_SIZE) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor ActorNetImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::tanh(fc3->forward(x));
    return x;
}

CriticNetImpl::CriticNetImpl()
    : fc1(OBS_SIZE + ACT_SIZE, 128), fc2(128, 128), fc3(128, 1) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor CriticNetImpl::forward(torch::Tensor state, torch::Tensor action) {
    auto x = torch::cat({state, action}, 1);
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return fc3->forward(x);
}

TD3Agent::TD3Agent(float actor_lr, float critic_lr, float gamma, float tau)
    : actor(std::make_shared<ActorNetImpl>()),
      actor_target(std::make_shared<ActorNetImpl>()),
      critic1(std::make_shared<CriticNetImpl>()),
      critic2(std::make_shared<CriticNetImpl>()),
      critic1_target(std::make_shared<CriticNetImpl>()),
      critic2_target(std::make_shared<CriticNetImpl>()),
      actor_optimizer(actor->parameters(), actor_lr),
      critic1_optimizer(critic1->parameters(), critic_lr),
      critic2_optimizer(critic2->parameters(), critic_lr),
      gamma(gamma), tau(tau) {
    actor_target->load_state_dict(actor->state_dict());
    critic1_target->load_state_dict(critic1->state_dict());
    critic2_target->load_state_dict(critic2->state_dict());
}

std::pair<torch::Tensor, torch::Tensor> TD3Agent::select_action(torch::Tensor state, float noise_std) {
    actor->eval();
    auto action = actor->forward(state);
    if (noise_std > 0.0f) {
        action += torch::normal(0.0, noise_std, action.sizes());
        action = torch::clamp(action, -1.0, 1.0);
    }
    actor->train();
    return {action, action.norm(2, 1, true)};
}

void TD3Agent::update(ReplayBuffer& buffer, int batch_size) {
    if (buffer.size() < batch_size) return;
    auto batch = buffer.sample(batch_size);

    std::vector<torch::Tensor> states, actions, rewards, next_states, dones;
    for (auto& t : batch) {
        states.push_back(t.state);
        actions.push_back(t.action);
        rewards.push_back(t.reward);
        next_states.push_back(t.next_state);
        dones.push_back(t.done);
    }

    auto state_batch = torch::stack(states);
    auto action_batch = torch::stack(actions);
    auto reward_batch = torch::stack(rewards);
    auto next_state_batch = torch::stack(next_states);
    auto done_batch = torch::stack(dones);

    auto next_action = actor_target->forward(next_state_batch);
    next_action += torch::normal(0.0, 0.2, next_action.sizes()).clamp(-0.5, 0.5);
    next_action = torch::clamp(next_action, -1.0, 1.0);

    auto target_q1 = critic1_target->forward(next_state_batch, next_action);
    auto target_q2 = critic2_target->forward(next_state_batch, next_action);
    auto target_q = torch::min(target_q1, target_q2);
    auto y = reward_batch + gamma * (1 - done_batch) * target_q;

    auto q1 = critic1->forward(state_batch, action_batch);
    auto q2 = critic2->forward(state_batch, action_batch);
    auto critic1_loss = torch::mse_loss(q1, y.detach());
    auto critic2_loss = torch::mse_loss(q2, y.detach());

    critic1_optimizer.zero_grad();
    critic1_loss.backward();
    critic1_optimizer.step();

    critic2_optimizer.zero_grad();
    critic2_loss.backward();
    critic2_optimizer.step();

    if (++update_step % policy_delay == 0) {
        auto actor_loss = -critic1->forward(state_batch, actor->forward(state_batch)).mean();
        actor_optimizer.zero_grad();
        actor_loss.backward();
        actor_optimizer.step();

        for (auto& [targ, src] : {
            std::pair<torch::nn::ModulePtr, torch::nn::ModulePtr>{actor_target, actor},
            {critic1_target, critic1}, {critic2_target, critic2}
        }) {
            auto targ_params = targ->named_parameters(true);
            auto src_params = src->named_parameters(true);
            for (auto& kv : targ_params) {
                kv.value().mul_(1 - tau).add_(src_params[kv.key()], tau);
            }
        }
    }
}

}
