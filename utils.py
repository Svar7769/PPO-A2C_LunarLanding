# ppo_project/utils.py

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def set_seeds(seed):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")

def make_env(env_name, seed):
    """Helper function to create and seed an environment."""
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env