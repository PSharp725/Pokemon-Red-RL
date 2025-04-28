import sys
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from red_gym_env import RedGymEnv
from run_config import EP_LENGTH, NUM_CPU, REWARD_SCALE, EXPLORE_WEIGHT
from stream_agent_wrapper import StreamWrapper
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def train_dqn_agent():
    use_wandb_logging = False
    ep_length = EP_LENGTH
    sess_id = "dqn_runs"
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
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': './PokemonRed.gb',
        'debug': False,
        'reward_scale': REWARD_SCALE,
        'explore_weight': EXPLORE_WEIGHT,
    }

    print(env_config)

    env = DummyVecEnv([make_env(0, env_config)])  # DQN requires single-env

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2,
        save_path=sess_path,
        name_prefix="poke"
    )
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=100,
        batch_size=16,
        tau=1.0,
        train_freq=4,
        target_update_interval=500,
        gamma=0.997,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        tensorboard_log=sess_path
    )

    print(model.policy)

    model.learn(
        total_timesteps=ep_length * 10000,
        callback=CallbackList(callbacks),
        tb_log_name="poke_dqn"
    )

    model.save(sess_path / "final_dqn_model")

    print("Training finished and final model saved.")

if __name__ == "__main__":
    train_dqn_agent()
