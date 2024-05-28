from stable_baselines3 import DQN
import argparse

from fightingice_env import FightingiceEnv
from opponent_pool import OpponentPool


def eval(args):
  env = FightingiceEnv(OpponentPool())
  
  model = DQN.load(args.checkpoint, env=env)

  for _ in range(args.num_episode):
    obs, _ = env.reset(p2="MctsAi")
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

