#include <torch/torch.h>
#include <iostream>
#include "ml/RL.hpp"
#include "Environment.hpp"

using namespace project::common;
using namespace rl;

int main() {
    rl::TD3Agent agent(1e-3, 1e-3, 0.99f, 0.005f);
    rl::ReplayBuffer buffer(100000);

    Environment env;
    int episodes = 500;
    int max_steps = 200;
    int batch_size = 64;

    for (int ep = 0; ep < episodes; ++ep) {
        State s = env.reset();
        torch::Tensor state = torch::from_blob(s.obs.data(), {1, OBS_SIZE}).clone();
        float ep_reward = 0.0f;

        for (int t = 0; t < max_steps; ++t) {
            auto [action_tensor, len_tensor] = agent.select_action(state, 0.1f);
            auto action_arr = action_tensor.squeeze().data_ptr<float>();
            Action act{{action_arr[0], action_arr[1]}, 1.0f};

            State s2 = env.do_action(act);
            torch::Tensor next_state = torch::from_blob(s2.obs.data(), {1, OBS_SIZE}).clone();

            float reward = -0.1f;
            bool done = false;
            if (s2.env_type == EnvState::TERMINAL) { reward = +10.0f; done = true; }
            else if (s2.env_type == EnvState::COLLISION) { reward = -10.0f; done = true; }

            buffer.push({
                state, action_tensor, torch::tensor({reward}), next_state, torch::tensor({done ? 1.0f : 0.0f})
            });

            agent.update(buffer, batch_size);
            state = next_state;
            ep_reward += reward;

            if (done) break;
        }

        std::cout << "Episode " << ep << ", reward = " << ep_reward << std::endl;
    }

    return 0;
}
