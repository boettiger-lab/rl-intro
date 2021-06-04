# Fishing using tuned SB3
import os
import stable_baselines3 as sb3
import gym
import gym_fishing


# Create a fishing environment
env = gym.make("fishing-v1", sigma=0.1)

# Create an agent
hyper = {
    "gamma": 0.95,
    "learning_rate": 0.000115,
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_starts": 0,
    "train_freq": 128,
    "tau": 0.01,
    "policy_kwargs": {"log_std_init": 0.680754, "net_arch": [400, 300]},
}

agent = sb3.SAC("MlpPolicy", env, seed=seed, **hyper)

# Train the agent
if not os.path.exists("cache/sac_tuned,zip"):
  agent.learn(total_timesteps=300000)
  agent.save("cache/sac_tuned")


agent = sb3.SAC.load("cache/sac_tuned")

# Evaluate the trained agent
eval_env = sb3.common.monitor.Monitor(gym.make("fishing-v1"))
mean_reward, std_reward = evaluate_policy(agent, eval_env, n_eval_episodes=100)
print("mean reward:", mean_reward, "std:", std_reward, "tuned_value:", 7.755079)


agent_sims = env.simulate(agent, reps=100)
agent_policy = env.policyfn(agent, reps=5)

print("mean reward from sims:", np.sum(agent_sims["reward"]) / 100)

# Plot results
env.plot(agent_sims, "results/fishing_sac_tuned.png")
env.plot_policy(agent_policy, "results/fishing_sac_tuned_policy.png")
