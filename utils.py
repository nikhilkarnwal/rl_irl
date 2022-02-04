from logging import info
from typing import Any, Dict, ItemsView, List, Union
from argparse import Action
import glob
import os.path as osp
from random import shuffle
import gym
from gym.core import ObservationWrapper

from imitation.algorithms.base import AnyTransitions
from imitation.data.types import Trajectory, TrajectoryWithRew
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch

def add_absorbing_state(obs, horizon=2):
    obs_shape = list(obs.shape)
    new_obs_shape = obs_shape.copy()
    #expand horizon and one dim for absorbing condition
    new_obs_shape[0]=horizon
    new_obs_shape[-1]+=1
    final_obs = np.zeros(new_obs_shape)

    # copy original
    final_obs[:obs_shape[0],:obs_shape[1]] = obs

    # put 1 in absorbing state dim, 0 for other 
    final_obs[:,-1]=1
    final_obs[:obs_shape[0],-1]=0

    return final_obs

def add_action_for_absorbing_states(actions, horizon=1):
    new_shape = list(actions.shape)
    new_shape[0]=horizon
    final_actions = np.zeros(new_shape)
    final_actions[:actions.shape[0],:] = actions
    return final_actions

class IRLASWrapper(ObservationWrapper):

    def __init__(self, env,i) -> None:
        super().__init__(env)
        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            shape=(obs_space.shape[0] + 1,),
            low=obs_space.low[0],
            high=obs_space.high[0])
        self.reset_done = False
        self.curr_len = 0
        self.absorbing_cnt=0

    def step(self, action):
        # self.curr_len+=1
        # if self.reset_done:
        #     self.absorbing_cnt +=1
        #     return self.get_absorbing_state(), 1, self.absorbing_cnt==2, {}
        observation, reward, done, info = super().step(action)
        # self.reset_done = done
        return observation, reward, done, info

    def observation(self,obs):
        return self.get_non_abosrbing_state(obs)

    def get_non_abosrbing_state(self, obs):
        return np.concatenate([obs, [0]], -1)

    def get_absorbing_state(self):
        obs = np.zeros(self.observation_space.shape)
        obs[-1] = 1
        return obs

    def reset(self, **kwargs):
        self.reset_done = False
        self.curr_len = 0
        self.absorbing_cnt=0
        return super().reset(**kwargs)

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps

    def get_state(self):
        return self.env.get_state()

    def get_obs(self):
        return self.observation(self.env._get_obs())
        
class ReplayBufferAS(ReplayBuffer):

    def __init__(self, buffer_size: int, observation_space, action_space, device: Union[torch.device, str] = "cpu", n_envs: int = 1, optimize_memory_usage: bool = False, handle_timeout_termination: bool = True,ep_max_len=200):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.max_len = ep_max_len
        self.episode_ts = 0

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        # print(obs.shape)
        self.episode_ts +=1
        if done[0] and (self.episode_ts+1)<=self.max_len:
            ab_state = np.zeros_like(obs)
            ab_state[:,-1]=1
            # add (T-1) to (Abs)
            super().add(obs, ab_state, action, reward, done, infos)
            # add (Abs) to (Abs)
            super().add(ab_state, ab_state, np.zeros_like(action), np.zeros_like(reward), np.zeros_like(done), infos)
            #reset current episode len to 0
            self.episode_ts=0
            return
        return super().add(obs, next_obs, action, reward, done, infos)


class ReplayBufferFHAS(ReplayBuffer):

    def __init__(self, buffer_size: int, observation_space, action_space, device: Union[torch.device, str] = "cpu", n_envs: int = 1, optimize_memory_usage: bool = False, handle_timeout_termination: bool = True,ep_max_len=200):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.max_len = ep_max_len
        self.episode_ts = 0

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        # print(obs.shape)
        self.episode_ts +=1
        if done[0] and (self.episode_ts+1)<=self.max_len:
            ab_state = np.zeros_like(obs)
            ab_state[:,-1]=1
            # add (T-1) to (Abs)
            super().add(obs, ab_state, action, reward, done, infos)
            # add (Abs) to (Abs)
            while (self.episode_ts+1)<=self.max_len:
                super().add(ab_state, ab_state, np.zeros_like(action), np.zeros_like(reward), np.zeros_like(done), infos)
                self.episode_ts+=1
            #reset current episode len to 0
            self.episode_ts=0
            return
        return super().add(obs, next_obs, action, reward, done, infos)


class TrajReplayABS:
    def __init__(self, capacity, horizon=1):
        self.horizon = horizon


    def process(self, main_memory):
        final_trajs = []
        for i in range(len(main_memory)):
            if self.horizon < main_memory[i].obs.shape[0]:
                self.horizon += main_memory[i].obs.shape[0]
            final_obs = add_absorbing_state(main_memory[i].obs, self.horizon+1)
            final_actions = add_action_for_absorbing_states(main_memory[i].acts, self.horizon)
            final_rewards = np.zeros(self.horizon, dtype=main_memory[i].rews.dtype)
            final_rewards[:main_memory[i].rews.shape[0]]=main_memory[i].rews

            final_trajs.append(TrajectoryWithRew(obs=final_obs,
                                                    acts=final_actions,
                                                    rews=final_rewards, infos=None, terminal=True))

        print(f'Processed-{len(main_memory)} trajs using backbone')
        return final_trajs