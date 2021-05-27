# Fishing using untuned TD3

import numpy as np
import stable_baselines3 as sb3
import gym
import gym_fishing

seed = 24
np.random.seed(seed)

# Create a fishing environment
env = gym.make("fishing-v1", sigma=0.1)
# Create the agent
td3 = sb3.TD3("MlpPolicy", env, seed=seed)

# Train the agent
td3.learn(total_timesteps=300000)
td3.save("cache/td3_untuned")

# Simulate management under the trained agent
td3 = sb3.TD3.load("cache/td3_untuned")
td3_sims = env.simulate(td3, reps=100)
td3_policy = env.policyfn(td3, reps=5)

# Example of built-in plotting methods
env.plot(td3_sims, "fishing_td3_untuned.png")
env.plot_policy(td3_policy, "fishing_td3_untuned_policy.png")
