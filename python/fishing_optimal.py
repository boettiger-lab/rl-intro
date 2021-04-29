# Using provably optimal 'constant escapement' policy instead of RL:

import numpy as np
import gym
import gym_fishing
from gym_fishing import models

seed = 24
np.random.seed(seed)

env = gym.make("fishing-v1", sigma=0.1)
optimal_policy = models.escapement(env)
opt_sims = env.simulate(optimal_policy, reps=500)
opt_policy = env.policyfn(optimal_policy)
