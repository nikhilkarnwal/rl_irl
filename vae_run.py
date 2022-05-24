import os
import gym
import yaml
import argparse
import numpy as np
from pathlib import Path, PurePath
from vae_mlp import MLPVAE
from vae_experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataModule
from pytorch_lightning.plugins import DDPPlugin
import d4rl
from imitation.data import types
from imitation.data.types import TrajectoryWithRew, load
from imitation.data import rollout
from gauss_mlp import GaussMLP

def build_env(name):
    env = gym.make(name)
    env.render = env.env.sim.render
    return env
    
def load_adroit(env,f_name):
    data = env.get_dataset(f_name)
    data = d4rl.qlearning_dataset(env, dataset=data)
    traj = types.Transitions(
        obs=data['observations'], acts=data['actions'],
        next_obs=data['next_observations'],infos=data['rewards'],dones=data['terminals'])
    return traj


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

print(config)

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
if config['model_name'] == "mlpvae":
    model_clss = MLPVAE
elif config['model_name'] == "gaussmlp":
    model_clss = GaussMLP

model =  model_clss(**config['model_params'])
print(model)
experiment = VAEXperiment(model,config['exp_params'])

# env = build_env(config["env"])
# trajs  = load_adroit(env, config['traj_file'])
trajs = rollout.flatten_trajectories(list(load(config['traj_file'])))
data = VAEDataModule(trajs, 1024)

data.setup()
print("Data Size - ", data.__len__())

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


# Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['logging_params']['name']} =======")
runner.fit(experiment, datamodule=data)
