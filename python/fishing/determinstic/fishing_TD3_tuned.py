# Fishing using tuned TD3
import stable_baselines3 as sb3
import gym
import gym_fishing

seed = 24

# Create a fishing environment
env = gym.make("fishing-v1")

# Define the agent
policy_kwargs = dict(net_arch=[400, 300])  # "big"
noise_std = 0.6656948079225263
action_noise = sb3.common.noise.NormalActionNoise(mean=0, sigma=noise_std)

td3 = sb3.TD3(
    "MlpPolicy",
    env,
    seed=seed,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0001355522450968401,
    gamma=0.995,
    batch_size=128,
    buffer_size=10000,
    train_freq=(128, "step"),
    gradient_steps=128,
    action_noise=action_noise,
)

# Train the agent
td3.learn(total_timesteps=300000)
td3.save("cache/td3_tuned")


# Simulate management under the trained agent
td3 = sb3.TD3.load("cache/td3_tuned")
td3_sims = env.simulate(td3, reps=100)
td3_policy = env.policyfn(td3, reps=5)

# Example of built-in plotting methods
env.plot(td3_sims, "results/fishing_td3_tuned.png")
env.plot_policy(td3_policy, "results/fishing_td3_tuned_policy.png")

