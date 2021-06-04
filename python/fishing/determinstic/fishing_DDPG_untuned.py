# Fishing using untuned ddpg

import numpy as np
import stable_baselines3 as sb3
import gym
import gym_fishing

seed = 24
np.random.seed(seed)

# Create a fishing environment
env = gym.make("fishing-v1", sigma=0.)
# Create the agent
ddpg = sb3.DDPG("MlpPolicy", env, seed=seed)

# Train the agent
ddpg.learn(total_timesteps=300000)
ddpg.save("cache/ddpg_untuned")

# Simulate management under the trained agent
ddpg = sb3.DDPG.load("cache/ddpg_untuned")
ddpg_sims = env.simulate(ddpg, reps=100)
ddpg_policy = env.policyfn(ddpg, reps=5)

# Example of built-in plotting methods
env.plot(ddpg_sims, "results/fishing_ddpg_untuned.png")
env.plot_policy(ddpg_policy, "results/fishing_ddpg_untuned_policy.png")
