#pragma once
#include "environment/Env.hpp"

namespace project::config {

const float WORLD_WIDTH = 100.0f;
const float WORLD_HEIGHT = 100.0f;

const int EPISODES = 10000;
const int MAX_STEPS = 500;
const int BATCH_SIZE = 512;
const int LOG_INTERVAL = 50;
const float ACTOR_LR = 3e-5;
const float CRITIC_LR = 3e-5;
const float GAMMA = 0.99f;
const float TAU = 0.005f;
const int TRAIN_START_SIZE = 5000;
const int TRAIN_INTERVAL = 1;

const env::Agent init_agent(WORLD_WIDTH / 2, WORLD_HEIGHT / 2);
const env::Goal goal(10.0f, 10.0f, 5.0f, 5.0f);

const std::vector obstacles = {
    env::Box(30.0f, 30.0f, 10.0f, 10.0f),
    env::Box(70.0f, 70.0f, 15.0f, 15.0f)
};

const env::Environment env(
    obstacles,
    goal,
    init_agent,
    0.0f, 0.0f,
    WORLD_WIDTH, WORLD_HEIGHT
);

}
