import argparse

import gym
from matplotlib import pyplot as plt
import numpy as np
from train_rl import build_env
from a2c import A2CLearner
from imitation.data.types import TrajectoryWithRew, load
from imitation.data import rollout
from train_rl import load_reward_model


parser = argparse.ArgumentParser(description='Run RL training code')
# Configurations
parser.add_argument('--abs', action='store_true', default=False)
parser.add_argument('--env', type=str, default="CartPole-v1")
args = parser.parse_args()
name = args.env

# venv, nenv, env = build_env(name, args)

env = gym.make(name)

print(env._max_episode_steps)
print(env.observation_space)
print(env.action_space)

# model = A2CLearner(env)
# model.train(2,100000)
# model.eval(100)

# def kld_loss():
#     kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),-1))
#     return kld 


trajs = load("/media/biswas/D/rl_irl/test_env/CartPole-v1/30_12_2021-17_58_15/trajs")
# trajs = rollout.flatten_trajectories(list(trajs))
# for j in range(3):    
#     obs = trajs[5+j].obs
#     print(obs.shape)
#     for i in range(obs.shape[1]):
#         plt.subplot(3,obs.shape[1],obs.shape[1]*j+(i+1))
#         plt.plot(np.arange(obs.shape[0]),obs[:,i])
# # plt.plot(np.arange(obs.shape[0]-1),trajs[0].acts)
# plt.show()
from gym.envs.classic_control.cartpole import CartPoleEnv
obs = env.reset()
id=0
print(type(env),isinstance(env,CartPoleEnv))
expert = True
if expert:
    env.env.state = trajs[id].obs[0]
    obs = trajs[id].obs[0]
    print(env.env.state)
    print(trajs[id].obs[0])


rew_model = load_reward_model("/media/biswas/D/rl_irl/test_env/cartpole/vae/logs/MLPVAE/version_7/checkpoints/epoch=24-step=1000.ckpt")
rew_model.setup()

cnt=0
curr_len=0
rew=0
rews=[]
lens = []
n_rews = []
n_rew = []
len_act = 1000
cid=0
while cnt < 5:
    # env.render(mode='human')
    # env.env.mj_render()
    # action = np.random.rand(*env.action_space.sample().shape) 
    # obs = np.random.rand(*env.observation_space.sample().shape)
    action = env.action_space.sample()
    # obs = env.observation_space.sample()

    n_rew.append(rew_model.get_rew(np.concatenate((obs, [action]), -1)))
    if expert:
        obs, action = (trajs[id].obs[cid],trajs[id].acts[cid])

    if cid >= len_act:
        action = env.action_space.sample()

    # n_rew.append(rew_model.get_rew(np.concatenate((obs, [action]), -1)))
    obs,r,d,info = env.step(action)
    # print(obs,env.env.state, trajs[id].obs[cid+1])
    cid+=1
    # time.sleep(0.01)
    # print(r,d)
    curr_len+=1
    rew+=r
    if d or curr_len==200:
        obs = env.reset()
        id+=1
        if expert:    
                env.env.state = trajs[id].obs[0]
                obs = trajs[id].obs[0]
            
        cnt+=1
        # id=0
        lens.append(curr_len)
        curr_len=0
        cid=0
        rews.append(rew)
        rew=0
        n_rews.extend(n_rew)
        n_rew = []

print(np.mean(lens),np.std(lens))

print(np.mean(rews),np.std(rews))

# print('Obs', np.mean(obs_all,0), np.var(obs_all,0))
# print(np.mean(n_rews, axis=0))
# print(len(n_rews), [len(r) for r in n_rews])
# print(np.mean(n_rews))
# plt.plot(np.arange(len(n_rews)), n_rews)
# plt.show()
# if expert:
#     plt.savefig('expert_all.png')
# else:
#     plt.savefig('sample_all.png')

plt.hist(n_rews,density=True)
plt.savefig('sample_all.png')
# plt.hist(np.asarray(obs_all)[:,4],density=True)
# plt.show()