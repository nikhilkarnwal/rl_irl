import time
from git import os
import gym
import d4rl
from gym.wrappers import Monitor
import numpy as np 
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
env.reset()
id=0
env.set_env_state({
    'qpos':dataset['infos/qpos'][0],
    'qvel':dataset['infos/qvel'][0],
    'door_body_pos':dataset['infos/door_body_pos'][0]})
cnt=0
len=0
rew=0
rews=[]
lens = []
while cnt < 30:
    # env.render(mode='human')
    # env.env.mj_render()
    obs,r,d,info = env.step(dataset['actions'][id])
    id+=1
    # time.sleep(0.01)
    # print(r,d)
    len+=1
    rew+=r
    if d:
        env.reset()
        env.set_env_state({
        'qpos':dataset['infos/qpos'][id],
        'qvel':dataset['infos/qvel'][id],
        'door_body_pos':dataset['infos/door_body_pos'][id]})
        cnt+=1
        # id=0
        lens.append(len)
        len=0
        rews.append(rew)
        rew=0
print(np.mean(lens),np.std(lens))

print(np.mean(rews),np.std(rews))

# import mujoco_py
# from mujoco_py import load_model_from_path, MjSim, MjViewer