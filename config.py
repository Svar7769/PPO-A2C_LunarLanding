# ppo_project/config.py

import torch

class PPOConfig:
    # Environment
    ENV_NAME = "LunarLanderContinuous-v3"
    SEED = 42

    # Global Training Parameters
    # IMPORTANT: Increase TOTAL_TIMESTEPS for meaningful learning
    TOTAL_TIMESTEPS = 200_000 # Increased for better learning (start with 2M-5M for LunarLander)
    N_STEPS = 2048              # Number of steps to collect per policy update (rollout length)
    GAMMA = 0.99                # Discount factor
    GAE_LAMBDA = 0.95           # Lambda for Generalized Advantage Estimation

    # Policy and Value Network Architecture
    # These are general, can be overridden by algorithm-specific LRs if needed
    ACTOR_LR = 3e-4             # Default learning rate for the actor network
    CRITIC_LR = 3e-4            # Default learning rate for the critic network
    HIDDEN_SIZES = [64, 64]     # Sizes of hidden layers in actor/critic networks

    # PPO Specific Parameters
    PPO_N_EPOCHS = 10           # Number of optimization epochs per policy update for PPO
    PPO_MINIBATCH_SIZE = 64     # Size of minibatches for PPO optimization
    CLIP_EPSILON = 0.2          # Clipping parameter for PPO
    
    # A2C Specific Parameters
    A2C_N_EPOCHS = 1            # Number of optimization epochs per policy update for A2C (typically 1)
    # A2C often uses larger minibatches (or the full rollout) since N_EPOCHS is 1
    # Try setting this to N_STEPS (2048) or a larger value like 256/512 for A2C
    A2C_MINIBATCH_SIZE = 256    # Changed from 64 for potentially better A2C stability
    # A2C specific learning rates (optional, but good for independent tuning)
    A2C_ACTOR_LR = 1e-4         # Often a bit lower for A2C due to direct gradients
    A2C_CRITIC_LR = 3e-4        # Can be same or slightly higher than actor LR

    # Common Algorithm Coefficients
    ENTROPY_COEFF = 0.01        # Coefficient for entropy regularization
    VALUE_LOSS_COEFF = 0.5      # Coefficient for value function loss

    # Logging and Saving
    LOG_DIR = "runs"            # Directory for TensorBoard logs
    SAVE_DIR = "trained_models" # Directory to save trained models
    LOG_INTERVAL = 10           # Log average rewards/losses every N updates (rollouts)
    # IMPORTANT: Adjust SAVE_INTERVAL to ensure models save within TOTAL_TIMESTEPS
    # If N_STEPS=2048, setting SAVE_INTERVAL=1 means save every 2048 steps.
    # Setting SAVE_INTERVAL=10 means save every 20480 steps.
    SAVE_INTERVAL = 10          # Changed from 100 to save more frequently, e.g., every 20,480 steps

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        print(f"Using device: {self.DEVICE}")

