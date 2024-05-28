import logging
import coloredlogs

import torch
from stable_baselines3 import PPO

from coach import Coach
from fightingice_env import FightingiceEnv
from model_wrapper import ModelWrapper
from config import Config

log = logging.getLogger(__name__)
 
coloredlogs.install(level="INFO")

import sys

sys.setrecursionlimit(100000)



def main():
    env = FightingiceEnv()
    env.reset() # Start the engine. We only need to simulate, doesn't need to actually step in the enviroment. So reset can be only called once.
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 512])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=256, batch_size=64)
    p_model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=256, batch_size=64)
    model_wrapper = ModelWrapper(model)
    p_model_wrapper = ModelWrapper(p_model)

    c = Coach(env, model_wrapper,p_model_wrapper)

    if Config.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            Config.load_folder_file[0],
            Config.load_folder_file[1],
        )
        model_wrapper.load_checkpoint(Config.load_folder_file[0], Config.load_folder_file[1])
        p_model_wrapper.load_checkpoint(Config.load_folder_file[0], Config.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":    
    main()
