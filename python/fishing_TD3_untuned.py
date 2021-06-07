import numpy as np
import stable_baselines3 as sb3
import os
import gym
import gym_fishing

seed = 24

# Create a fishing environment
env = gym.make("fishing-v1", sigma=0.1)

# Initialize the agent to train using TD3 algorithm
agent = sb3.TD3("MlpPolicy", env, seed=24)

# Train the agent
if not os.path.exists("cache/td3_untuned.zip"):
    agent.learn(total_timesteps=300000)
    agent.save("cache/td3_untuned")

# Load the saved agent
agent = sb3.TD3.load("cache/td3_untuned")

# Evaluate agent score using built-in methods:
eval_env = sb3.common.monitor.Monitor(env)
mean, sd = sb3.common.evaluation.evaluate_policy(agent, eval_env)
print("mean score:", mean)

# Simulate management under the trained agent (using deterministic policy)
for i in range(0,10):
  state, action = agent.predict(state, deterministic=True)
  state = env.step(action)

# Or gym_fishing provides additional helper methods for simulation & plotting
td3_sims = env.simulate(agent, reps=100)
td3_policy = env.policyfn(agent, reps=1)

# Example of built-in plotting methods
env.plot(td3_sims, "results/fishing_td3_tuned.png")
env.plot_policy(td3_policy, "results/fishing_td3_tuned_policy.png")
