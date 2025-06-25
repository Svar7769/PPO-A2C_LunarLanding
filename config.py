# ppo_project/config.py

import torch

class PPOConfig:
    # Environment
    ENV_NAME = "LunarLanderContinuous-v2"
    SEED = 42

    # Training
    TOTAL_TIMESTEPS = 1_000_000 # Total interactions with the environment
    N_STEPS = 2048              # Number of steps to collect per policy update
    N_EPOCHS = 10               # Number of optimization epochs per policy update
    MINIBATCH_SIZE = 64         # Size of minibatches for optimization
    GAMMA = 0.99                # Discount factor
    GAE_LAMBDA = 0.95           # Lambda for Generalized Advantage Estimation

    # Policy and Value Network
    ACTOR_LR = 3e-4             # Learning rate for the actor network
    CRITIC_LR = 3e-4            # Learning rate for the critic network
    HIDDEN_SIZES = [64, 64]     # Sizes of hidden layers in actor/critic networks

    # PPO Specific
    CLIP_EPSILON = 0.2          # Clipping parameter for PPO
    ENTROPY_COEFF = 0.01        # Coefficient for entropy regularization
    VALUE_LOSS_COEFF = 0.5      # Coefficient for value function loss

    # Logging and Saving
    LOG_DIR = "runs"            # Directory for TensorBoard logs
    SAVE_DIR = "trained_models" # Directory to save trained models
    LOG_INTERVAL = 10           # Log every N updates
    SAVE_INTERVAL = 100         # Save model every N updates

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        print(f"Using device: {self.DEVICE}")

# You can instantiate the config in other files like this:
# from config import PPOConfig
# config = PPOConfig()