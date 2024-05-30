import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import argparse

from fightingice_env import FightingiceEnv
from callback import SaveCallback
from opponent_pool import OpponentPool


def env_creator():
    oppent_pool = OpponentPool()
    return FightingiceEnv(oppent_pool)


def main(args):
    env = make_vec_env(env_creator, n_envs=args.num_env, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_creator, n_envs=1, vec_env_cls=SubprocVecEnv)
    if args.load_model is not None:
        print("Loading model from: ", args.load_model, flush=True)
        model = DQN.load(args.load_model, env=env, verbose=1, batch_size=args.batch_size, exploration_initial_eps=0.1) # Set the exploration_initial_eps to 0.1
    else:
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 512, 128])
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=args.batch_size,
        )
    logger = configure(args.log_path, ["stdout", "csv"])
    model.set_logger(logger)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_path,
        log_path=args.log_path,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    save_callback = SaveCallback(save_freq=args.save_freq, save_dir=args.save_path)

    print("num envs: ", model.n_envs, flush=True)
    model.learn(
        total_timesteps=args.total_timesteps, log_interval=1, callback=[eval_callback, save_callback]
    )
    model.save(os.path.join(args.save_path, f"dqn_{args.total_timesteps}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_env", type=int, default=16)
    parser.add_argument("--eval_freq", type=int, default=128)
    parser.add_argument("--save_freq", type=int, default=512)
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./checkpoint/dqn")
    parser.add_argument("--log_path", type=str, default="./sb3_log")

    args = parser.parse_args()

    main(args)
