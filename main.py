# ppo_project/main.py

import os
import argparse
from config import PPOConfig
from ppo_agent import PPOAgent
from a2c_agent import A2CAgent # Import the A2C agent

def main():
    """
    Main function to initialize and run the selected RL agent training.
    """
    parser = argparse.ArgumentParser(description="Train a Reinforcement Learning Agent (PPO or A2C).")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"],
                        help="Which algorithm to train: 'ppo' or 'a2c'.")
    args = parser.parse_args()

    # Instantiate the PPO configuration
    config = PPOConfig()

    # Create directories for logs and saved models if they don't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    agent = None
    if args.algo == "ppo":
        # Override config parameters with PPO specific ones
        config.N_EPOCHS = config.PPO_N_EPOCHS
        config.MINIBATCH_SIZE = config.PPO_MINIBATCH_SIZE
        print(f"Initializing PPO Agent with N_EPOCHS={config.N_EPOCHS}, MINIBATCH_SIZE={config.MINIBATCH_SIZE}")
        agent = PPOAgent(config)
    elif args.algo == "a2c":
        # Override config parameters with A2C specific ones
        config.N_EPOCHS = config.A2C_N_EPOCHS
        config.MINIBATCH_SIZE = config.A2C_MINIBATCH_SIZE
        print(f"Initializing A2C Agent with N_EPOCHS={config.N_EPOCHS}, MINIBATCH_SIZE={config.MINIBATCH_SIZE}")
        agent = A2CAgent(config)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}. Choose 'ppo' or 'a2c'.")

    # Start training
    # agent.train()

    # After training, optionally evaluate the final policy
    # To evaluate, uncomment the lines below and optionally load a specific model
    # Example: Load a PPO model saved at step 1,000,000 and evaluate
    model_path = os.path.join(config.SAVE_DIR, f"{args.algo}_agent_step_184320.pth") 
    agent.load_model(model_path) 
    agent.evaluate(num_episodes=10, render=True) 


if __name__ == "__main__":
    main()
