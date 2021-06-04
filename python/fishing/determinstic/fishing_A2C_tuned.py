# Fishing using (untuned) A2C

import numpy as np
from torch import nn as nn
import stable_baselines3 as sb3
import os
import gym
import gym_fishing

seed = 24
np.random.seed(seed)

# Create a fishing environment
env = gym.make("fishing-v1")

hyper = {
    "gamma": 0.98,
    "normalize_advantage": True,
    "max_grad_norm": 0.5,
    "use_rms_prop": False,
    "gae_lambda": 0.92,
    "n_steps": 16,
    "learning_rate": 0.0014125669914905178,
    "ent_coef": 0.000415417442958291,
    "vf_coef": 0.7417182197443206,
    "policy_kwargs": {
        "ortho_init": False,
        "net_arch": [64, 64],
        "activation_fn": nn.ReLU,
    }
}

# Create an A2C agent
agent = sb3.A2C("MlpPolicy", env, seed=seed, **hyper)

# Train
if not os.path.exists("cache/a2c_tuned.zip"):
    agent.learn(total_timesteps=300000)
    agent.save("cache/a2c_tuned")

# Load the saved agent
agent = sb3.A2C.load("cache/a2c_tuned")

# Evaluate
mean, sd = sb3.common.evaluation.evaluate_policy(agent, env)
print("mean score:", mean)

# Plot results
agent_sims = env.simulate(agent, reps=100)
agent_policy = env.policyfn(agent, reps=1)
env.plot(agent_sims, "results/fishing_a2c_tuned.png")
env.plot_policy(agent_policy, "results/fishing_a2c_tuned_policy.png")


