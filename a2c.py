from calendar import day_abbr
from distutils.log import info
from turtle import forward
import gym
from matplotlib.pyplot import axis
import numpy as np

from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# env = make_vec_env("MountainCarContinuous-v0", n_envs=1)

# # model = A2C("MlpPolicy", env, verbose=1)
# # model.learn(total_timesteps=25000)
# # model.save("a2c_cartpole")

# # del model # remove to demonstrate saving and loading

# # model = A2C.load("a2c_cartpole")

# # obs = env.reset()
# # while True:
# #     action, _states = model.predict(obs)
# #     obs, rewards, dones, info = env.step(action)
# #     env.render()
# print(env.action_space)
# print(env.action_space.low)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from tqdm import tqdm, trange

class ContinuousHead(nn.Module):
    def __init__(self, scale_prior, mean_prior, noise_std) -> None:
        super().__init__()
        self.scale_prior = nn.Parameter(torch.tensor(scale_prior, dtype=torch.float32), requires_grad=False)
        self.mean_prior = nn.Parameter(torch.tensor(mean_prior, dtype=torch.float32), requires_grad=False)
        self.noise_std = noise_std

    def forward(self, actions):
        mean = F.tanh(actions)
        mean_noise = Normal(torch.zeros_like(mean),self.noise_std).sample(mean.shape)
        actions = (mean+mean_noise).clamp(-1,1)*self.scale_prior + self.mean_prior
        return actions, Normal(self.mean_prior, self.scale_prior).log_prob(actions)

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
        self.value_optim = torch.optim.Adam(self.value.parameters())
        self.policy_optim = torch.optim.Adam(iter( self.policy.parameters()))

    def forward(self, obs):
        acts = self.policy(obs)
        values = self.value(obs)
        acts, prob = self.head(acts)

        return (acts, torch.sum(prob,dim=-1)), values

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
    
    def predict(self, obs):
        with torch.no_grad():
            ret = self.forward(obs)
        return (ret[0].cpu().to_numpy())


from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer

def collect_rollout(env: gym.Env, buff: ReplayBuffer, timesteps, policy):
    curr_steps = 0
    while curr_steps < timesteps:
        obs  = env.reset()
        ep_start = True
        while True:
            obs_tensor = torch.tensor(obs,dtype=torch.double, device=buff.device)
            (action, log_prob), value = policy.predict(torch.FloatTensor(obs).to(buff.device))
            action = action[0].cpu().detach().numpy()
            log_prob = log_prob[0].cpu().detach()
            value = value[0].cpu().detach()
            next_obs, rew, done, info = env.step(action)
            print(obs.shape, log_prob.shape)
            buff.add(obs=obs,next_obs= next_obs,action= action,reward= [rew], done= [done], infos=[info])
            ep_start = False
            if done:
                break;
            curr_steps += 1
            obs = next_obs
    return curr_steps




class A2CLearner:
    def __init__(self, env: gym.Env, rollout_buff: ReplayBuffer = None, buff_size=1000000, batch_size = 256, logger= None) -> None:
        self.env = env
        self.rollout_buff = rollout_buff
        self.batch_size = batch_size
        self.logger = logger
        self.buff_size = buff_size
        self.device = 'cuda'
        self._setup_model()
        
    def _setup_model(self):
        if self.rollout_buff is None:
            self.rollout_buff = ReplayBuffer(
                self.buff_size, self.env.observation_space, 
                self.env.action_space, self.device)

        self.policy = ContinuousPolicy(self.env.observation_space.shape[0],self.env.action_space.shape[0],self.env.action_space).to(self.device)


    def train(self, epochs=10, total_timesteps=1000000):
        pbar = trange(total_timesteps)
        pbar.set_description('Training Policy')
        curr_ts = 0
        while curr_ts < total_timesteps:
            curr_ts += collect_rollout(self.env, self.rollout_buff, self.batch_size, self.policy)
            curr_logs = {}
            for i in range(epochs):
                data = self.rollout_buff.sample(self.batch_size)

                _logs = self.policy.update(data.observations.to(dtype=torch.float32), data.next_observations, data.rewards, data.actions)
                for (k,v) in _logs.items():
                    if k not in curr_logs.keys():
                        curr_logs[k] = []
                    curr_logs[k].append(v)
            pbar.update(curr_ts)
            pbar.set_postfix({k:np.mean(v) for (k,v) in curr_logs.items()})
        pbar.close()
        print("Done!")
            


        




