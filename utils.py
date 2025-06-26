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

def make_env(env_name, seed, render_mode="human"): # ADDED render_mode parameter
    """
    Helper function to create and seed an environment.
    Args:
        env_name (str): Name of the Gymnasium environment.
        seed (int): Seed for environment randomness.
        render_mode (str, optional): The render_mode to pass to gym.make().
                                     Common values: "human", "rgb_array".
                                     Defaults to None (no rendering).
    """
    env = gym.make(env_name, render_mode=render_mode) # Pass render_mode here
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
