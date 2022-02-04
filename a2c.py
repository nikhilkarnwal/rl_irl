from turtle import forward
import gym
from pytablewriter import EmptyValueError

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("MountainCarContinuous-v0", n_envs=1)

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("a2c_cartpole")

# del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
print(env.action_space)
print(env.action_space.low)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

class ContinuousHead(nn.Module):
    def __init__(self, scale_prior, mean_prior, noise_std) -> None:
        super().__init__()
        self.scale_prior = nn.Parameter(torch.tensor(scale_prior, dtype=torch.float32), requires_grad=False)
        self.mean_prior = nn.Parameter(torch.tensor(mean_prior, dtype=torch.float32), requires_grad=False)
        self.noise_std = noise_std

    def forward(self, actions):
        mean = F.tanh(actions)
        mean_noise = mean.clone()._normal(0,std=self.noise_std)
        actions = (mean+mean_noise).clamp(-1,1)*self.scale_prior + self.mean_prior
        return actions


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim,out_dim,hid_dim, hid_l) -> None:
        super().__init__()
        self.model = nn.Sequential(
            
        )