# Using steady-state policy instead of RL:

import numpy as np
import gym
import gym_conservation
from gym_conservation import models

seed = 24
np.random.seed(seed)

env = gym.make("conservation-v6")

# Simulate under the steady-state solution (given the model)
K = 1.5
alpha = 0.001
steadystate = models.fixed_action(env, fixed_action=alpha * 100 * 2 * K)
opt_sims = env.simulate(steadystate, reps=100)
opt_policy = env.policyfn(steadystate)
