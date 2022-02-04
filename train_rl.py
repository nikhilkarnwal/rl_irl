"""Trains BC, GAIL and AIRL models on saved CartPole-v1 demonstrations."""

import argparse
from distutils.log import info
import os
import numpy as np
import gym
from imitation.data.types import TrajectoryWithRew, save, load
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from datetime import datetime
import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util
from stable_baselines3.common.noise import NormalActionNoise
import stable_baselines3.common.utils as s3utils
import yaml
from utils import ReplayBufferFHAS, TrajReplayABS, ReplayBufferAS, IRLASWrapper
import d4rl
from imitation.data import types
gen_algo_cfg = {}

irl_cfg = {}

cfg = {}

def build_env(name,args):
    env = gym.make(name)
    env.render = env.env.sim.render
    n_envs = gen_algo_cfg.pop('n_envs',1)
    n_ts = gen_algo_cfg.pop('n_timesteps',100000)
    print(f"Spaces- obs : {env.observation_space}, action : {env.action_space}")
    if args.abs :
        nenv = IRLASWrapper(gym.make(name),0)
        venv = util.make_vec_env(name, n_envs=1,post_wrappers=[IRLASWrapper])
    else:
        nenv = env
        venv = util.make_vec_env(name, n_envs=1)
    return venv, nenv, env

def run_gail(name, transitions, args, work_dir):
    # # Load pickled test demonstrations.
    # with open("./final.pkl", "rb") as f:
    #     # This is a list of `imitation.data.types.Trajectory`, where
    #     # every instance contains observations and actions for a single expert
    #     # demonstration.
    #     trajectories = pickle.load(f)

    # # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # # This is a more general dataclass containing unordered
    # # (observation, actions, next_observation) transitions.
    # transitions = rollout.flatten_trajectories(trajectories)
    if isinstance(transitions,list):
        print('Traj-shape:',transitions[0].obs.shape)
    else:
        print('Traj-shape:',transitions.obs.shape)

    venv, nenv, env = build_env(name,args)

    tempdir_path = pathlib.Path(work_dir)
    print(
        f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    # Train BC on expert data.
    # BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
    # dictionaries containing observations and actions.
    # bc_logger = logger.configure(tempdir_path / "BC/")
    # bc_trainer = bc.BC(
    #     observation_space=venv.observation_space,
    #     action_space=venv.action_space,
    #     demonstrations=transitions,
    #     custom_logger=bc_logger,
    # )
    # bc_trainer.train(n_epochs=1)

    # Train GAIL on expert data.
    # GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
    # iterates over dictionaries containing observations, actions, and next_observations.
    gail_logger = logger.configure(tempdir_path / "GAIL/")
    # setting replay buffer
    replay_buffer_kwargs={'ep_max_len': env._max_episode_steps}
    if args.abs:
        replay_bf_cls = ReplayBufferFHAS
    else:
        replay_bf_cls = None
        replay_buffer_kwargs = None
    action_noise = None
    n_actions = env.action_space.shape[-1]
    if args.explore:
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    if args.gen == 'ppo':
        gen_algo = PPO(env=venv,tensorboard_log=work_dir, **gen_algo_cfg)
    else:
        gen_algo = sb3.SAC(
            env=venv,tensorboard_log=work_dir, **gen_algo_cfg, 
            replay_buffer_class=replay_bf_cls, replay_buffer_kwargs=replay_buffer_kwargs, action_noise=action_noise, ent_coef="auto_5")
    # gen_algo.learn(100000)
    # gen_algo.collect_rollouts()
    # gail_trainer = airl.AIRL(
    #     venv=venv,
    #     demonstrations=transitions,
    #     demo_batch_size=irl_cfg['demo_batch_size'],
    #     gen_algo=gen_algo,
    #     custom_logger=gail_logger, n_disc_updates_per_round=irl_cfg['round'],
    #     normalize_reward=False, normalize_obs=False,
    #     init_tensorboard_graph=True,
    #     allow_variable_horizon=True,gen_train_timesteps = irl_cfg['gen_train_timesteps'],
    #     disc_opt_kwargs={'lr': irl_cfg['lr']},
    # )
    gail_trainer = gail.GAIL(
        venv=venv,
        demonstrations=None,
        demo_batch_size=irl_cfg['demo_batch_size'],
        gen_algo=gen_algo,
        custom_logger=gail_logger, n_disc_updates_per_round=irl_cfg['round'],
        normalize_reward=False, normalize_obs=False,
        init_tensorboard_graph=True,
        allow_variable_horizon=True,gen_train_timesteps = irl_cfg['gen_train_timesteps'],
        disc_opt_kwargs={
            'lr': irl_cfg['lr'],
            'betas':(0.95, 0.999),}
            # 'weight_decay':10,},
    )
    gail_trainer.set_demonstrations(transitions)
    callbks = []
    callbks.append(EvalCallback(
            venv,
            best_model_save_path=work_dir,
            n_eval_episodes=20,
            log_path=work_dir,
            eval_freq=20000
        ))
    gail_trainer.gen_callback = [*callbks,gail_trainer.gen_callback]
    gail_trainer.allow_variable_horizon = True
    gail_trainer.train(total_timesteps=int(irl_cfg['ts']))

    # Train AIRL on expert data.
    # airl_logger = logger.configure(tempdir_path / "AIRL/")
    # airl_trainer = airl.AIRL(
    #     venv=venv,
    #     demonstrations=transitions,
    #     demo_batch_size=32,
    #     gen_algo=gen_algo,
    #     custom_logger=airl_logger,
    # )
    # airl_trainer.allow_variable_horizon=True
    # airl_trainer.train(total_timesteps=2048*50)
    gen_algo.save(f"{work_dir}/gen_model.pth")
    run_traj(name,args,gen_algo,20,work_dir, args.abs)


# gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024)
# gen_algo.learn(2048)
# venv = gen_algo.env
# obs = venv.reset()
# for i in range(100):
#     obs_tensor = obs_as_tensor(obs,'cuda')
#     actions, values, log_probs = gen_algo.policy.forward(obs_tensor)
#     a = actions.cpu().numpy()
#     obs, rew, done, info = venv.step(a)
#     print(f'obs{obs}, rew{rew}, done{done}, info{info}')
#     gen_algo.ep_info_buffer.clear()
#     gen_algo._update_info_buffer(info)
# print(gen_algo.ep_info_buffer)

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def run_gen(name,args , work_dir="temp"):
    # Parallel environments
    n_envs = gen_algo_cfg.pop('n_envs')
    n_ts = gen_algo_cfg.pop('n_timesteps')
    

    # env = venv
    env = util.make_vec_env(name, n_envs=1)
    print(gen_algo_cfg)
    gen_algo = None
    callbks = []
    callbks.append(EvalCallback(
            env,
            best_model_save_path=work_dir,
            n_eval_episodes=20,
            log_path=work_dir,
            eval_freq=5000
        ))
    if args.gen == 'ppo':
        model = PPO(env=env,tensorboard_log=work_dir, **gen_algo_cfg)
        gen_algo = PPO
    else:
        model = sb3.SAC(env=env,tensorboard_log=work_dir, **gen_algo_cfg)
        gen_algo = sb3.SAC
    
    if args.resume_model != None:
        del model
        model = gen_algo.load(args.resume_model,env=env, print_system_info=True, force_reset=True)
        model.tensorboard_log = work_dir
        print('Model loaded')
    model.learn(total_timesteps=n_ts,eval_freq=10000,eval_log_path=work_dir, callback=callbks)
    model.save(f"{work_dir}/model")
    return model

import tqdm
def init(name, args):
    venv, nenv, env = build_env(name,args)
    cfg['ep_max_len'] = env._max_episode_steps
    return env

def get_traj(name,args, model, num, render=5):
    trajs = []
    traj_len = []
    traj_rew = []
    venv, nenv, env = build_env(name,args)
    for i in tqdm.tqdm(range(num),total=num, desc='Creating Trajs'):
        obs_list = []
        rew_list = []
        actions_list = []
        obs = env.reset()
        cnt = 0
        while cnt < env.spec.max_episode_steps:
            # new_obs = np.zeros_like(obs)
            # new_obs[:,:obs.shape[1]] = obs
            # obs = new_obs
            obs = np.expand_dims(obs,axis=0)
            action, _states = model.predict(obs)
            obs_list.append(obs[0])
            actions_list.append(action[0])
            if i < render:
                env.render()
            obs, rewards, dones, info = env.step(action[0])
            rew_list.append(rewards)
            cnt += 1
            if dones:
                # print(cnt)
                break
        # if np.sum(rew_list) < 3500:
        #     continue
        obs_list.append(obs)
        term = True
        traj_len.append(cnt)
        traj_rew.append(np.sum(rew_list))
        if cnt == env.spec.max_episode_steps:
            term = False
        trajs.append(TrajectoryWithRew(obs=np.array(obs_list), acts=np.array(actions_list),
                                       rews=np.reshape(rew_list, [len(rew_list), ]), infos=None, terminal=True))
    env.close()
    print(f'Mean len-{np.mean(traj_len)}, Rew - {np.mean(traj_rew)}')
    return trajs

from gym.wrappers import Monitor 
def run_traj(name, args, model, num, work_dir,abs=False):
    venv, nenv, env = build_env(name,args)
    env = Monitor(env,work_dir,force=True)
    # Parallel environments
    # venv = make_vec_env(name, n_envs=1,monitor_dir=work_dir)
    for i in tqdm.tqdm(range(num),total=num, desc='Saving Videos'):
        obs = env.reset()
        cnt = 0
        while cnt < env.spec.max_episode_steps:
            if abs:
                new_obs = np.zeros((obs.shape[0]+1))
                new_obs[:obs.shape[0]] = obs
                obs = new_obs
            obs = np.expand_dims(obs,axis=0)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action[0])
            cnt += 1
            if dones:
                break
    env.close()


def test_traj(trajs):
    traj_len = []
    traj_rew = []
    for traj in trajs:
        # if traj.obs.shape[0] <800:
        #     continue
        traj_len.append(traj.obs.shape[0])
        traj_rew.append(np.sum(traj.rews))
    print(f"Total traj- {len(traj_len)}")
    print(f"Testing traj: Mean Len - {np.mean(traj_len)}, Mean Rew - {np.mean(traj_rew)}")
    print(f"Testing traj: Std Len - {np.std(traj_len)}, Std Rew - {np.std(traj_rew)}")

def load_params(args):
    global irl_cfg
    with open(args.config_file, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if args.env in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[args.env]
        else:
            raise ValueError(f"Hyperparameters not found for {args.gen}-{args.env}")
        if args.irl and args.irl in list(hyperparams_dict.keys()):
            irl_cfg = hyperparams_dict[args.irl]
        else:
            print('Info-No irl info is found in config')

    # Sort hyperparams that will be saved
    # saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    print("Default hyperparameters for environment (ones being tuned will be overridden):")
    print(hyperparams, irl_cfg)
    global gen_algo_cfg
    gen_algo_cfg = hyperparams
    global sac_algo
    gen_algo_cfg = hyperparams
    return hyperparams, irl_cfg

def test_gail(name, args, work_dir):
    # # Load pickled test demonstrations.
    # with open("./final.pkl", "rb") as f:
    #     # This is a list of `imitation.data.types.Trajectory`, where
    #     # every instance contains observations and actions for a single expert
    #     # demonstration.
    #     trajectories = pickle.load(f)

    # # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # # This is a more general dataclass containing unordered
    # # (observation, actions, next_observation) transitions.
    # transitions = rollout.flatten_trajectories(trajectories)
    venv, nenv, env = build_env(name,args)

    if args.gen == 'ppo':
        model = PPO(env=venv,tensorboard_log=work_dir, **gen_algo_cfg)
        gen_algo = PPO
    else:
        model = sb3.SAC(env=venv,tensorboard_log=work_dir, **gen_algo_cfg)
        gen_algo = sb3.SAC
    
    if args.resume_model != None:
        del model
        model = gen_algo.load(args.resume_model,env=venv, print_system_info=True, force_reset=True)
        model.tensorboard_log = work_dir
        print('Model loaded')
    return model

def load_adroit(env,f_name):
    data = env.get_dataset(f_name)
    data = d4rl.qlearning_dataset(env, dataset=data)
    traj = types.Transitions(
        obs=data['observations'], acts=data['actions'],
        next_obs=data['next_observations'],infos=data['rewards'],dones=data['terminals'])
    return traj

def main():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")

    parser = argparse.ArgumentParser(description='Run RL training code')
    # Configurations
    parser.add_argument('--gen', help='gn algo', type=str, default='ppo')
    parser.add_argument('--irl', type=str, default=None)
    parser.add_argument('--env', type=str, default="Walker2d-v2")
    parser.add_argument('--trajs', type=str, default=None)
    parser.add_argument('--resume_model', type=str, default=None)
    parser.add_argument('--gen_trajs', action='store_true', default=False)
    parser.add_argument('--test_trajs', action='store_true', default=False)
    parser.add_argument('--save_video', action='store_true', default=False)

    parser.add_argument('--explore', action='store_true', default=False)

    parser.add_argument('--test_model', action='store_true', default=False)
    parser.add_argument('--abs', action='store_true', default=False)
    parser.add_argument('--num_trajs', help='num of traj to gen',
                        type=int, default=1000)
    parser.add_argument('--rl_ts', help='num of traj to gen',
                        type=int, default=500000)
    parser.add_argument('--irl_ts', help='num of traj to gen',
                        type=int, default=500000)

    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()
    name = args.env

    env = init(name,args)

    if args.config_file != None:
        params = load_params(args)
    # name = "CartPole-v1"
    work_dir = f"/media/biswas/D/rl_irl/test_env/{name}/{dt_string}/"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        print(f"Creating dir-{work_dir}")

    print(f"Storing at {work_dir}")
    with open(f'{work_dir}/desc_file.txt', 'w') as fd:
        fd.write(f"{args.__str__()}, {params}")

    if args.test_model:
        model = test_gail(name,args,work_dir)

    if args.gen_trajs:
        model = run_gen(name, args, work_dir)
        trajs = get_traj(name, args, model, args.num_trajs, 0)
        args.trajs = 'trajs'
        save(f"{work_dir}/{args.trajs}", trajs)
        print("Saved trajs")
    elif args.trajs != None:
        trajs = load_adroit(env, args.trajs)
        # trajs = load(args.trajs)
    
    if args.abs and args.trajs != None:
        traj_abs = TrajReplayABS(cfg['ep_max_len'])
        trajs = traj_abs.process(trajs)

    if args.save_video:
        run_traj(name, args,model,20,work_dir, args.test_model and args.abs)

    if args.test_trajs:
        test_traj(trajs)
        return 

    if args.irl:
        run_gail(name, trajs, args, work_dir)


if __name__ == "__main__":
    main()