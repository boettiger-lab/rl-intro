# Fishing using (untuned) A2C

import os
import gym
import gym_fishing
import stable_baselines3 as sb3


# Create a fishing environment
env = gym.make("fishing-v1", sigma=0.1)

# Create an A2C agent
agent = sb3.A2C("MlpPolicy", env, seed=24)

# Train
if not os.path.exists("cache/a2c_untuned.zip"):
    agent.learn(total_timesteps=300000)
    agent.save("cache/a2c_untuned")

# Load the saved agent
agent = sb3.A2C.load("cache/a2c_untuned")

# Evaluate
eval_env = sb3.common.monitor.Monitor(env)
mean, sd = sb3.common.evaluation.evaluate_policy(agent, eval_env)
print("mean score:", mean)

# Plot results
agent_sims = env.simulate(agent, reps=100)
agent_policy = env.policyfn(agent, reps=1)
env.plot(agent_sims, "results/fishing_a2c_untuned.png")
env.plot_policy(agent_policy, "results/fishing_a2c_untuned_policy.png")


