# --- START OF FILE test_drone_env.py ---

import gym
import numpy as np
import time

# IMPORTANT: Adjust this import path based on how your project is structured
# and how you installed your custom environment.
# Option 1: If installed as a package
# from drone_2d_custom_gym_env_package.drone_2d_custom_gym_env import Drone2dEnv
# Option 2: If drone_2d_env.py is in the same directory or accessible via PYTHONPATH
try:
    from drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv
except ImportError:
    print("Error importing Drone2dEnv. Make sure the environment file is accessible.")
    print("Attempting relative import...")
    # Adjust relative path if needed (e.g., if test file is outside the package)
    from drone_2d_custom_gym_env_package.drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv


# --- Test Configuration ---
NUM_EPISODES = 5       # How many episodes to run
MAX_STEPS_PER_EPISODE = 750 # Max steps before terminating an episode
RENDER_SIM = True      # Set to True to visualize the simulation
RENDER_DELAY_S = 0.02  # Small delay (in seconds) between frames if rendering

# --- Environment Options ---
# Choose the configuration you want to test
ENABLE_WIND_TEST = True
MOVING_PLATFORM_TEST = True
PLATFORM_SPEED_TEST = 2.5 # Meters per second
INITIAL_RANDOM_RANGE_TEST = 10.0 # Meters (+/- from center)
MAX_TILT_ANGLE_DEG_TEST = 90.0 # Degrees

def run_test():
    """Runs the test loop for the Drone2dEnv."""
    print("--- Starting Drone Environment Test ---")

    # Convert degrees to radians for the environment parameter
    max_tilt_rad = np.radians(MAX_TILT_ANGLE_DEG_TEST)

    # Initialize the environment with chosen options
    try:
        env = Drone2dEnv(
            render_sim=RENDER_SIM,
            max_steps=MAX_STEPS_PER_EPISODE,
            moving_platform=MOVING_PLATFORM_TEST,
            platform_speed=PLATFORM_SPEED_TEST,
            initial_pos_random_range_m=INITIAL_RANDOM_RANGE_TEST,
            enable_wind=ENABLE_WIND_TEST,
            max_allowed_tilt_angle_rad=max_tilt_rad,
            # Keep other render options (path, shade) as their defaults or specify them
            # render_path=True,
            # render_shade=True,
        )
    except Exception as e:
        print(f"\n!!! Error creating environment: {e} !!!")
        print("Please check the environment's __init__ method and parameters.")
        return

    print("\n--- Environment Initialized ---")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    # print(f"Sample Observation: {env.observation_space.sample()}") # Can be useful for checking bounds

    total_steps_all_episodes = 0

    for episode in range(NUM_EPISODES):
        print(f"\n--- Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        try:
            # Reset the environment for a new episode
            obs = env.reset()
            print(f"Initial Observation sample: {obs[:4]}...") # Print start of obs
        except Exception as e:
            print(f"\n!!! Error resetting environment: {e} !!!")
            print("Skipping episode.")
            continue

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Render the current state (if enabled)
            if RENDER_SIM:
                try:
                    env.render()
                    if RENDER_DELAY_S > 0:
                        time.sleep(RENDER_DELAY_S)
                except Exception as e:
                    print(f"\n!!! Error rendering environment: {e} !!!")
                    print("Disabling rendering for this episode.")
                    # Optionally disable rendering completely: RENDER_SIM = False
                    break # Stop this episode if rendering fails badly

            # --- Action Selection ---
            # For testing, use random actions. Replace with your agent's policy later.
            action = env.action_space.sample()
            # print(f"Step {step_count}: Action = {action}") # Uncomment for debugging action values

            # --- Step the Environment ---
            try:
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step_count += 1
                total_steps_all_episodes += 1
            except Exception as e:
                print(f"\n!!! Error during env.step() at step {step_count}: {e} !!!")
                print("Terminating episode.")
                done = True # Force episode termination
                info = info if 'info' in locals() else {} # Use existing info if available
                info['error'] = f"Exception during step: {e}"

            # Optional: Print step info periodically
            # if step_count % 100 == 0:
            #     print(f"  Step: {step_count}, Reward: {reward:.3f}, Done: {done}")

            # Check if max steps reached (env should handle this with 'done', but as a backup)
            if step_count >= MAX_STEPS_PER_EPISODE and not done:
                print("Warning: Max steps reached in test loop, but env not 'done'.")
                done = True # Force termination in test script

        # --- Episode End ---
        print(f"--- Episode {episode + 1} Finished ---")
        print(f"Steps taken: {step_count}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Info: {info}")

        # Render the final frame after done (if rendering)
        if RENDER_SIM:
             try: env.render(); time.sleep(0.5) # Pause briefly on final frame
             except: pass # Ignore render errors on final frame

    # --- Cleanup ---
    try:
        env.close()
        print("\nEnvironment Closed.")
    except Exception as e:
        print(f"\nError closing environment: {e}")

    print(f"\n--- Test Finished ---")
    print(f"Total steps across all episodes: {total_steps_all_episodes}")


if __name__ == "__main__":
    run_test()