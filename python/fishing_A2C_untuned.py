# Fishing using (untuned) A2C

import numpy as np
import stable_baselines3 as sb3
import gym
import gym_fishing

seed = 24
np.random.seed(seed)

# Create a fishing environment
env = gym.make("fishing-v1", sigma=0.1)

# Create an A2C agent
a2c = sb3.A2C("MlpPolicy", env, seed=seed)

# Train the agent for a fixed number of timesteps
a2c.learn(total_timesteps=300000)

# Save our trained agent for future use
a2c.save("cache/a2c")

# Load the saved agent
a2c = sb3.A2C.load("cache/a2c")


# Example manual simulation of the agent
state = env.get_state(0.75)
for i in range(1, 10):
    action, null = a2c.predict(state)
    state, reward, done, info = env.step(action)

## Or more consisely using built-in method:
a2c_sims = env.simulate(a2c, reps=500)

# Estimate the 'Policy function'
a2c_policy = env.policyfn(a2c, reps=50)
