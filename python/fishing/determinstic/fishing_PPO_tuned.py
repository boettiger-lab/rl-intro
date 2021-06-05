import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from torch import nn as nn
import os
import gym
import gym_fishing

seed = 24
env = make_vec_env("fishing-v1", n_envs=4, seed=seed, env_kwargs={"sigma": 0.})

# Create a fishing environment
hyper = {
    "batch_size": 16,
    "n_steps": 8,
    "gamma": 0.99,
    "learning_rate": 0.0012512477171887717,
    "ent_coef": 6.908006009288375e-08,
    "clip_range": 0.3,
    "n_epochs": 1,
    "gae_lambda": 0.9,
    "max_grad_norm": 0.7,
    "vf_coef": 0.7688840852349988,
     "policy_kwargs": {
        "net_arch": [256, 256], # should be separate networks(?)
        "activation_fn": nn.ReLU,
    }
}
# Create an agent
agent = sb3.PPO("MlpPolicy", env, seed=seed, **hyper)
# Train the agent

if not os.path.exists("cache/ppo_tuned.zip"):
    agent.learn(total_timesteps=300000)
    agent.save("cache/ppo_tuned")

# Evaluate the trained agent
env = gym.make("fishing-v1", sigma=0.)

agent = sb3.PPO.load("cache/ppo_tuned")
agent_sims = env.simulate(agent, reps=100)
agent_policy = env.policyfn(agent, reps=5)

# Plot results
env.plot(agent_sims, "results/fishing_ppo_tuned.png")
env.plot_policy(agent_policy, "results/fishing_ppo_tuned_policy.png")
