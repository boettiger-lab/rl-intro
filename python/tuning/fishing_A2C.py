# Fishing using A2C

import numpy as np
import stable_baselines3 as sb3
import gym
import gym_fishing
from torch import nn as nn
from stable_baselines3.common.env_util import make_vec_env


seed = 24
# Create a fishing environment
env = make_vec_env("fishing-v1", n_envs = 4, seed = seed, env_kwargs = {"sigma": 0.1})

# Create an A2C agent
agent = sb3.A2C("MlpPolicy",
              env, 
              seed=seed,
              gamma= 0.995,
              normalize_advantage= False,
              max_grad_norm= 0.7,
              use_rms_prop= False,
              gae_lambda= 0.92,
              n_steps= 8,
              learning_rate= 0.0006987399679768267,
              ent_coef= 3.585469128347999e-05,
              vf_coef= 0.48712388559858977,
              policy_kwargs= dict(activation_fn=nn.ReLU,
                                 ortho_init=True,
                                 net_arch=[dict(pi=[256, 256], vf=[256, 256])]
                                 ))

# Train the agent for a fixed number of timesteps
agent.learn(total_timesteps=300000)

# Save our trained agent for future use
agent.save("cache/fishing_a2c")

# Evaluation
env = gym.make("fishing-v1", sigma= 0.1)
sims = env.simulate(agent, reps = 100)
policy = env.policyfn(agent, reps = 5)

env.plot(sims, "fishing_a2c.png")
env.plot_policy(policy, "fishing_a2c_policy.png")
