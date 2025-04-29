from pathlib import Path

from stable_baselines3 import A2C
from red_gym_env import RedGymEnv

def main():
    sess_path = Path("a2c_runs")
    model_path = sess_path / "poke_84541440_steps.zip"
    assert model_path.exists(), "No final model found. Train it first."

    env_config = {
        'headless': True,  # Change to see the agent play durning training
        'save_final_state': True,  # We will have alot of saves as checkpoints
        'early_stop': False,    # Stop training when the agent is done
        'action_freq': 24,                      # How many frames to skip
        'init_state': './init.state',           # Initial state to load
        'max_steps': 2048 * 2,            # Max steps to run the agent  
        'print_rewards': True,       # Print rewards    
        'save_video': True,     # Save video of the agent playing
        'fast_video': True,     # Save the video with fast movement (skip frames)
        'session_path': sess_path / "final_rollout",        # Path to save the files
        'gb_path': './PokemonRed.gb',                       # Path to the gameboy file   
        'debug': False,        # Debug mode (will print a lot of stuff)
        'reward_scale': 0.5,                            # Scale the reward
        'explore_weight': 0.25,                   # Exploration weight  
    }

    # Ensure output directory exists
    (sess_path / "final_rollout").mkdir(exist_ok=True)                          

    env = RedGymEnv(env_config)  # Create the environment

    model = A2C.load(model_path, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})   # Load the model         
    obs, _ = env.reset() # Reset the environment
    done = False                

    while not done:
        action, _states = model.predict(obs, deterministic=False)   # Get the action from the model       
        obs, reward, _, done, _ = env.step(action)  # Step the environment                                 
        env.save_and_print_info(done, obs)          # Save the state and print the info

    print("Final rollout finished and video saved!")

if __name__ == "__main__":
    main()
