from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym_fishing

seed = 24
env = make_vec_env("fishing-v1", n_envs = 4, seed = seed)
total_timesteps = 3e5

agent = PPO('MlpPolicy', 
          env, 
          seed = seed,
          batch_size= 8,
          n_steps= 64,
          gamma= 0.95,
          learning_rate= 0.009063123943359215,
          ent_coef= 0.002465145165808194,
          clip_range= 0.4,
          n_epochs= 5,
          gae_lambda= 0.98,
          max_grad_norm= 0.8,
          vf_coef= 0.8153389910080954
          )

agent.learn(total_timesteps = 3e5)

# Save our trained agent for future use
agent.save("cache/fishing_ppo")


