# ppo_project/main.py

import os
from config import PPOConfig
from ppo_agent import PPOAgent

def main():
    """
    Main function to initialize and run the PPO agent training.
    """
    # Instantiate the PPO configuration
    config = PPOConfig()

    # Create directories for logs and saved models if they don't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Initialize the PPO Agent
    agent = PPOAgent(config)

    # Start training
    agent.train()

    # After training, optionally evaluate the final policy
    # Note: For evaluation, you might want to load a specific trained model
    # if you're not evaluating the very last one.
    # agent.evaluate(num_episodes=10, render=True) # Set render to True to visualize
                                                  # if your environment supports it.
                                                  # Ensure env.make() has render_mode="human"
                                                  # if you intend to render.


if __name__ == "__main__":
    main()
