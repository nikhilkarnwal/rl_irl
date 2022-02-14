import argparse
from train_rl import build_env
from a2c import A2CLearner

parser = argparse.ArgumentParser(description='Run RL training code')
# Configurations
parser.add_argument('--abs', action='store_true', default=False)
parser.add_argument('--env', type=str, default="Hopper-v3")
args = parser.parse_args()
name = args.env

venv, nenv, env = build_env(name, args)

print(env._max_episode_steps)
print(env.observation_space.shape)
print(env.action_space.shape)

model = A2CLearner(env)
model.train(2,1000)