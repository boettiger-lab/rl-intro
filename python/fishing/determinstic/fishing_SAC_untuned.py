import os
import gym
import gym_fishing
import stable_baselines3 as sb3


# Create a fishing environment
env = gym.make("fishing-v1")

# Create an agent
agent = sb3.SAC("MlpPolicy", env, seed=24)

# Train the agent
if not os.path.exists("cache/sac_untuned.zip"):
  agent.learn(total_timesteps=300000)
  agent.save("cache/sac_untuned")

agent = sb3.SAC.load("cache/sac_untuned")

# Evaluate
mean, sd = sb3.common.evaluation.evaluate_policy(agent, sb3.common.monitor.Monitor(env))
print("mean score:", mean)

# Plot results
agent_sims = env.simulate(agent, reps=100)
agent_policy = env.policyfn(agent, reps=2)
env.plot(agent_sims, "results/fishing_sac_untuned.png")
env.plot_policy(agent_policy, "results/fishing_sac_untuned_policy.png")



