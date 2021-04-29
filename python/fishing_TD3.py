import numpy as np
import stable_baselines3 as sb3
import gym
import gym_fishing

seed = 24
np.random.seed(seed)

# Create a fishing environment
env = gym.make("fishing-v1", sigma=0.1)


policy_kwargs = dict(net_arch=[400, 300])  # "big"
# non-episodic action noise:
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
    train_freq=128,
    n_episodes_rollout=-1,
    gradient_steps=128,
    action_noise=action_noise,
)

td3.learn(total_timesteps=300000)
td3.save("cache/td3")


# Simulate management under the trained agent
td3 = sb3.TD3.load("cache/td3")
td3_sims = env.simulate(td3, reps=500)
td3_policy = env.policyfn(td3, reps=50)
