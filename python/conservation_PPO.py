from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch import nn as nn
import gym_conservation

seed = 24
env = make_vec_env("conservation-v6", n_envs = 4, seed = seed)

agent = PPO('MlpPolicy', 
            env, 
            seed = seed,
            batch_size= 64,
            n_steps= 256,
            gamma= 0.9,
            learning_rate= 0.0008330851234475645,
            ent_coef= 7.290496088342335e-08,
            clip_range= 0.3,
            n_epochs= 1,
            gae_lambda= 0.95,
            max_grad_norm= 0.5,
            vf_coef= 0.7638921186547223,
            policy_kwargs= dict(activation_fn=nn.ReLU,
                                net_arch=[dict(pi=[256, 256], vf=[256, 256])]
                                )
          )
agent.learn(total_timesteps = 3e5)

# Save our trained agent for future use
agent.save("cache/conservation_ppo")

# Evaluation
env = gym.make("conservation-v6")
sims = env.simulate(agent, reps = 100)
policy = env.policyfn(agent, reps = 5)

env.plot(sims, "conservation_ppo.png")
env.plot_policy(policy, "conservation_ppo_policy.png")


