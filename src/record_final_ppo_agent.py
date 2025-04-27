import sys
import os
from pathlib import Path

from stable_baselines3 import PPO
from red_gym_env import RedGymEnv

def main():
    sess_path = Path("runs")
    model_path = sess_path / "final_model.zip"
    assert model_path.exists(), "❌ No final model found. Train it first."

    env_config = {
        'headless': True,
        'save_final_state': True,  # Important: save map/screenshots
        'early_stop': False,
        'action_freq': 24,
        'init_state': './init.state',
        'max_steps': 2048 * 5,  # 5 rollouts for demo
        'print_rewards': True,
        'save_video': True,
        'fast_video': True,
        'session_path': sess_path / "final_rollout",
        'gb_path': './PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25,
    }

    # Ensure output directory exists
    (sess_path / "final_rollout").mkdir(exist_ok=True)

    env = RedGymEnv(env_config)

    model = PPO.load(model_path, env=env)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, _, done, _ = env.step(action)
        env.save_and_print_info(done, obs)

    print("Final rollout finished and video saved!")

if __name__ == "__main__":
    main()
