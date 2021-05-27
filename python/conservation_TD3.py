# Ecological Tipping Points using TD3

import numpy as np
import stable_baselines3 as sb3
import gym
import gym_conservation

seed = 24
np.random.seed(seed)

env = gym.make("conservation-v6")

noise_std = 0.4805935357322933
OU = sb3.common.noise.OrnsteinUhlenbeckActionNoise
action_noise = OU(mean=np.zeros(1), sigma=noise_std * np.ones(1))
model = sb3.TD3(
    "MlpPolicy",
    env,
    verbose=0,
    seed=42,
    gamma=0.995,
    learning_rate=8.315382409902049e-05,
    batch_size=512,
    buffer_size=10000,
    train_freq=1000,
    gradient_steps=1000,
    action_noise=action_noise,
    policy_kwargs={"net_arch": [64, 64]},
)

model.learn(total_timesteps=3000000)
model.save("cache/td3-conservation")

model = sb3.TD3.load("cache/td3-conservation")
TD3_sims = env.simulate(model, reps=100)
TD3_policy = env.policyfn(model, reps=10)
