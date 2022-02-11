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
        return actions, Normal.log_prob(actions)

def build_mlp(arch,act):
    model = []
    model.append(nn.Linear(arch['in'],arch['hid']))
    if act != None:
        model.append(act())
    for i in range(arch['hid_l']):
        model.append(nn.Linear(arch['hid'],arch['hid']))
        if act!= None:
            model.append(act())
    model.append(nn.Linear(arch['hid'],arch['out']))
    return nn.Sequential(*model)


class Network(nn.Module):
    def __init__(self, in_dim,out_dim,hid_dim, hid_l) -> None:
        super().__init__()
        self.model = build_mlp(arch={
            'in':in_dim,
            'out':out_dim,
            'hid':hid_dim,
            'hid_l':hid_l
        }, act=nn.Tanh)

    def forward(self,obs):
        action = self.model(obs)
        return action

class ContinuousPolicy(nn.Module):

    def __init__(self, obs_shape, action_shape, action_space) -> None:
        super().__init__()
        self.policy = Network(obs_shape, action_shape, 64,3)
        self.value = Network(obs_shape,1,64,3)
        policy_head_cfg = {}
        if action_space.is_bounded():
            low = action_space.low
            high = action_space.high
            scale_prior = (high - low) / 2
            bias_prior = (low + high) / 2
            policy_head_cfg['scale_prior'] = scale_prior
            policy_head_cfg['mean_prior'] = bias_prior
        self.head = ContinuousHead(**policy_head_cfg, noise_std=0.2)
        self.value_optim = torch.optim.Adam(self.value.parameters)
        self.policy_optim = torch.optim.Adam({'params':self.policy.parameters(),'params':self.head.parameters()})

    def forward(self, obs):
        acts = self.policy(obs)
        values = self.value(obs)
        acts, prob = self.head(acts)

        return (acts, prob), values

    def update(self, obs,next_obs, rew, acts):
        logs ={}
        # train value function
        with torch.no_grad():
            expected_v = rew + self.value(next_obs)
        self.value.train()
        actual_v = self.value(obs)
        self.value_optim.zero_grad()
        v_loss  = F.mse_loss(expected_v, actual_v)
        logs['v_loss'] = v_loss.item()
        v_loss.backward()
        self.value_optim.step()


        # train policy
        with torch.no_grad():
            adv = rew + self.value(next_obs) - self.value(obs)

        self.policy.train()
        self.head.train()
        self.policy_optim.zero_grad()

        actual_acts, log_prob = self.head(self.policy(obs))

        act_loss = torch.mean(log_prob*adv)
        logs['act_loss'] = act_loss.item()
        act_loss.backward()
        self.policy_optim.step()

        return logs






