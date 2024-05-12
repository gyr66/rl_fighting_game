import torch
from stable_baselines3 import PPO
import argparse

from fightingice_env import FightingiceEnv

def env_creator():
    return FightingiceEnv()

def eval(args):
  env = env_creator()
  policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 512])
  model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=128, batch_size=64)
  model.set_parameters(args.checkpoint)

  for _ in range(args.num_episode):
    obs, _ = env.reset()
    while True:
      action, _ = model.predict(obs)
      obs, reward, done, truncated, _ = env.step(action)
      if done:
         break
  env.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, default="./checkpoint/mcts/best_model")
  parser.add_argument("--num_episode", type=int, default=10)

  args = parser.parse_args()

  eval(args)






