import sys
import os
from pathlib import Path


from red_gym_env import RedGymEnv
from agents.random_agent import RandomAgent

def main():
    sess_id = "random_runs"
    sess_path = Path(sess_id)
    
    env_config = {
        'headless': True,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': './init.state',
        'max_steps': 2048 * 80,
        'save_video': True,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': './PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25,
        'print_rewards': True,
    }
    
    env = RedGymEnv(env_config)
    agent = RandomAgent(env.action_space)

    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, _, done, _ = env.step(action)
        env.save_and_print_info(done, obs)

    print("Finished RandomAgent rollout.")

if __name__ == "__main__":
    main()