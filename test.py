import argparse
from train_rl import build_env

parser = argparse.ArgumentParser(description='Run RL training code')
# Configurations
parser.add_argument('--abs', action='store_true', default=False)
parser.add_argument('--env', type=str, default="Hopper-v3")
args = parser.parse_args()
name = args.env

venv, env = build_env(name, args)

print(env._max_episode_steps)