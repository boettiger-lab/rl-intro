import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from torch import nn as nn

import gym
import gym_fishing

seed = 24
np.random.seed(seed)
env = make_vec_env("fishing-v1", n_envs=4, seed=seed)


# Create a fishing environment
hyper = {
    "batch_size": 64,
    "n_steps": 32,
    "gamma": 0.9999,
    "learning_rate": 0.00037996229760279524,
    "ent_coef": 2.392750198613048e-08,
    "clip_range": 0.3,
    "n_epochs": 10,
    "gae_lambda": 0.92,
    "max_grad_norm": 0.7,
    "vf_coef": 0.4863551514846155,
     "policy_kwargs": {
        "net_arch": [64, 64],
        "activation_fn": nn.ReLU,
    }
}

# Create an agent
agent = sb3.PPO("MlpPolicy", env, seed=seed, **hyper)
# Train the agent
agent.learn(total_timesteps=300000)
agent.save("cache/PPO_tuned")

# Evaluate the trained agent
env = gym.make("fishing-v1")

agent = sb3.PPO.load("cache/PPO_tuned")
agent_sims = env.simulate(agent, reps=100)
agent_policy = env.policyfn(agent, reps=5)

# Plot results
env.plot(agent_sims, "results/fishing_PPO_tuned.png")
env.plot_policy(agent_policy, "results/fishing_PPO_tuned_policy.png")
