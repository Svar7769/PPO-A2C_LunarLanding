# ppo_project/networks.py

import torch
import torch.nn as nn
from torch.distributions import Normal

from config import PPOConfig
from utils import init_weights

class ActorNetwork(nn.Module):
    """
    Implements the Actor (Policy) network for continuous action spaces.
    It takes the state as input and outputs the mean and standard deviation
    of a Gaussian distribution, from which actions are sampled.
    """
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super(ActorNetwork, self).__init__()

        # Define the layers of the neural network
        layers = []
        # Input layer
        layers.append(nn.Linear(observation_dim, hidden_sizes[0]))
        layers.append(nn.ReLU()) # Using ReLU activation function

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # Combine all layers into a sequential model
        self.fc_layers = nn.Sequential(*layers)

        # Output layer for the mean of the Gaussian policy
        # The output dimension should match the action_dim
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)

        # Log standard deviation of the Gaussian policy.
        # It's common to learn log_std and then exponentiate it to get std,
        # ensuring std is always positive. Initializing it as a trainable
        # parameter ensures it can be learned.
        # We use a small initial value (e.g., -0.5) to keep actions somewhat
        # deterministic at the start, allowing exploration to expand as needed.
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        # Apply orthogonal initialization to weights to help with training stability
        self.apply(init_weights)

    def forward(self, state):
        """
        Forward pass through the actor network.
        Calculates the mean and standard deviation for the policy distribution.
        """
        # Pass state through the shared fully connected layers
        x = self.fc_layers(state)

        # Get the mean of the Gaussian distribution
        mean = self.mean_layer(x)

        # Calculate standard deviation by exponentiating log_std.
        # Clamp log_std to prevent extremely small or large values for stability.
        # PPOConfig.MAX_ACTION_STD and MIN_ACTION_STD can be added to config.py
        # if more precise control over the std range is needed.
        std = torch.exp(self.log_std.expand_as(mean))

        # Create a Normal distribution object
        # This allows us to sample actions and compute log probabilities
        policy_distribution = Normal(mean, std)

        return policy_distribution


class CriticNetwork(nn.Module):
    """
    Implements the Critic (Value) network.
    It takes the state as input and outputs a single value,
    representing the estimated value of that state.
    """
    def __init__(self, observation_dim, hidden_sizes):
        super(CriticNetwork, self).__init__()

        # Define the layers of the neural network
        layers = []
        # Input layer
        layers.append(nn.Linear(observation_dim, hidden_sizes[0]))
        layers.append(nn.ReLU()) # Using ReLU activation function

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # Combine all layers into a sequential model
        self.fc_layers = nn.Sequential(*layers)

        # Output layer for the state-value estimate (a single scalar)
        self.value_layer = nn.Linear(hidden_sizes[-1], 1)

        # Apply orthogonal initialization to weights
        self.apply(init_weights)

    def forward(self, state):
        """
        Forward pass through the critic network.
        Calculates the estimated value of the input state.
        """
        # Pass state through the fully connected layers
        x = self.fc_layers(state)

        # Get the state-value estimate
        value = self.value_layer(x)

        return value
