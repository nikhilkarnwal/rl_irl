import time
from git import os
import gym
import d4rl
from gym.wrappers import Monitor
from matplotlib import pyplot as plt
import numpy as np
from train_rl import load_reward_model
# Create the environment
env_name = 'door-expert-v1'
env = gym.make(env_name)

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

work_dir = "/media/biswas/D/d4rl/"+env_name+"/"
os.environ['D4RL_DATASET_DIR']=work_dir
d4rl.set_dataset_path(work_dir)
if not os.path.exists(work_dir):
    print('Creating dir-', work_dir)
    os.makedirs(work_dir)
# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
# print(dataset['observations'][0]) # An N x dim_observation Numpy array of observations
# print(dataset.keys())

# # Alternatively, use d4rl.qlearning_dataset which
# # also adds next_observations.
# dataset = d4rl.qlearning_dataset(env)
# print(dataset.keys())
# print(dataset['observations'].shape)
env.render = env.env.sim.render
env = Monitor(env,work_dir,force=True)
obs = env.reset()
id=0

expert = True
if expert:    
    env.set_env_state({
        'qpos':dataset['infos/qpos'][0],
        'qvel':dataset['infos/qvel'][0],
        'door_body_pos':dataset['infos/door_body_pos'][0]})

rew_model = load_reward_model("/media/biswas/D/rl_irl/test_env/adroit/vae/logs/MLPVAE/version_19/checkpoints/epoch=13-step=10892.ckpt")
rew_model.setup()

cnt=0
len=0
rew=0
rews=[]
lens = []
n_rews = []
n_rew = []
len_act = 25
cid=0
while cnt < 50:
    # env.render(mode='human')
    # env.env.mj_render()
    # action = np.random.rand(*env.action_space.sample().shape) 
    # obs = np.random.rand(*env.observation_space.sample().shape)
    action = env.action_space.sample()
    obs = env.observation_space.sample()

    if expert:
        obs, action = (dataset['observations'][id],dataset['actions'][id])

    if cid >= len_act:
        action = env.action_space.sample()

    n_rew.append(rew_model.get_rew(np.concatenate((obs, action), -1)))
    obs,r,d,info = env.step(action)
    id+=1
    cid+=1
    # time.sleep(0.01)
    # print(r,d)
    len+=1
    rew+=r
    if d:
        obs = env.reset()
        if expert:    
            env.set_env_state({
            'qpos':dataset['infos/qpos'][id],
            'qvel':dataset['infos/qvel'][id],
            'door_body_pos':dataset['infos/door_body_pos'][id]})
            
        cnt+=1
        # id=0
        lens.append(len)
        len=0
        cid=0
        rews.append(rew)
        rew=0
        n_rews.append(n_rew)
        n_rew = []
print(np.mean(lens),np.std(lens))

print(np.mean(rews),np.std(rews))

rew_model.close()
print(np.mean(n_rews, axis=0))
print(np.mean(n_rews), np.std(n_rews))
plt.plot(np.arange(200), np.mean(n_rews, axis=0))
# plt.show()
if expert:
    plt.savefig('expert_all.png')
else:
    plt.savefig('sample_all.png')
    # plt.savefig('non_random_sample_all.png')
# import mujoco_py
# from mujoco_py import load_model_from_path, MjSim, MjViewer