# Fishing using tuned TD3

import os
import numpy as np
import stable_baselines3 as sb3
import gym
import gym_fishing

seed = 24
np.random.seed(seed)

# Create a fishing environment
env = gym.make("fishing-v1")

# Create an agent
agent = sb3.SAC("MlpPolicy", env, seed=seed)

# Train the agent
if not os.path.exists("cache/sac_untuned.zip"):
  agent.learn(total_timesteps=300000)
  agent.save("cache/sac_untuned")

# Evaluate the trained agent
agent = sb3.SAC.load("cache/sac_untuned")
agent_sims = env.simulate(agent, reps=100)
agent_policy = env.policyfn(agent, reps=5)

# Plot results
env.plot(agent_sims, "results/fishing_sac_untuned.png")
env.plot_policy(agent_policy, "results/fishing_sac_untuned_policy.png")



