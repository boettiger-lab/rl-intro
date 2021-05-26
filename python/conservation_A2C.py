# Fishing using A2C

import numpy as np
import stable_baselines3 as sb3
import gym
import gym_conservation
from torch import nn as nn
from stable_baselines3.common.env_util import make_vec_env

seed = 24
# Create a fishing environment
env = make_vec_env("conservation-v6", n_envs = 4, seed = seed)

# Create an A2C agent
a2c = sb3.A2C("MlpPolicy",
              env, 
              seed=seed,
              gamma= 0.9999,
              normalize_advantage= True,
              max_grad_norm= 0.6,
              use_rms_prop= False,
              gae_lambda= 0.8,
              n_steps= 128,
              learning_rate= 0.021243170810461984,
              ent_coef= 1.7498876702865585e-07,
              vf_coef= 0.12495864179484306,
              policy_kwargs= dict(ortho_init=False))

# Train the agent for a fixed number of timesteps
a2c.learn(total_timesteps=300000)

# Save our trained agent for future use
a2c.save("cache/conservation_a2c")


# Evaluation
env = gym.make("conservation-v6")
sims = env.simulate(agent, reps = 100)
policy = env.policyfn(agent, reps = 5)

env.plot(sims, "conservation_a2c.png")
env.plot_policy(policy, "conservation_a2c_policy.png")
