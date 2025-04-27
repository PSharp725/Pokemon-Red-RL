import sys
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from red_gym_env import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def train_ppo_agent():
    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "ppo_runs"
    sess_path = Path(sess_id)
    sess_path.mkdir(exist_ok=True)

    env_config = {
        'headless': True,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': './init.state',
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': False,
        'session_path': sess_path,
        'gb_path': './PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25,
    }

    print(env_config)

    num_cpu = 32
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2,
        save_path=sess_path,
        name_prefix="poke"
    )
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    train_steps_batch = ep_length // 64

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=train_steps_batch,
        batch_size=512,
        n_epochs=1,
        gamma=0.997,
        ent_coef=0.01,
        tensorboard_log=sess_path
    )

    print(model.policy)

    model.learn(
        total_timesteps=(ep_length) * num_cpu * 10000,
        callback=CallbackList(callbacks),
        tb_log_name="poke_ppo"
    )

    # Save final model
    model.save(sess_path / "final_ppo_model")

    print("Training finished and final model saved.")

if __name__ == "__main__":
    train_ppo_agent()
