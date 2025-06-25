# ppo_project/buffer.py

import torch
import numpy as np
from config import PPOConfig

class RolloutBuffer:
    """
    A buffer for storing trajectories collected from the environment.
    It's used to store states, actions, rewards, log probabilities of actions,
    and value predictions. It also computes advantages using GAE and returns.
    """
    def __init__(self, config: PPOConfig):
        self.config = config
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = [] # Whether an episode terminated (True) or truncated (False)
        self.values = [] # Value predictions from the critic for each state
        self.log_probs = [] # Log probabilities of chosen actions

        self.returns = [] # Discounted returns
        self.advantages = [] # Generalized Advantage Estimates

        self.current_ptr = 0 # Pointer for the current position in the buffer

    def add(self, state, action, reward, done, value, log_prob):
        """
        Adds a single experience step to the buffer.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.current_ptr += 1

    def compute_returns_and_advantages(self, last_value):
        """
        Computes discounted returns and Generalized Advantage Estimates (GAE)
        for the collected trajectories.

        Args:
            last_value (torch.Tensor): The value prediction of the last state
                                       in the trajectory, or 0 if the episode ended.
        """
        # Convert lists to tensors for computation
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.config.DEVICE)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.config.DEVICE)
        dones = torch.tensor(self.dones, dtype=torch.bool, device=self.config.DEVICE)

        # Initialize advantage and return tensors
        # Add the last value to the end of the values tensor for computation
        # If the episode truly ended (done is True), the last_value should be 0.
        # Otherwise, it's the critic's prediction for the last state.
        extended_values = torch.cat((values, last_value.unsqueeze(0)))
        
        # GAE calculation
        advantages = torch.zeros_like(rewards, device=self.config.DEVICE)
        last_gae_lam = 0 # Stores the previous GAE value for iterative calculation

        # Iterate backward through the trajectory to compute GAE
        for t in reversed(range(len(rewards))):
            # If the episode ended at this step, next_value is 0 for reward calculation
            # and the GAE term is reset.
            if t == len(rewards) - 1: # Last step in the current segment
                next_non_terminal = 1.0 - dones[-1].float()
                next_value = last_value
            else: # Not the last step in the segment
                next_non_terminal = 1.0 - dones[t+1].float()
                next_value = extended_values[t+2] # extended_values includes last_value at the end

            # TD error (delta)
            # δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
            delta = rewards[t] + self.config.GAMMA * next_value * next_non_terminal - values[t]
            
            # A_t = δ_t + γ * λ * A_{t+1} * (1 - done_{t+1})
            # (1 - done_{t+1}) term handles episode termination properly
            advantages[t] = last_gae_lam = delta + self.config.GAMMA * self.config.GAE_LAMBDA * next_non_terminal * last_gae_lam
        
        # Calculate returns (target values for the critic)
        # Returns are simply advantages + current state values
        self.returns = advantages + values
        self.advantages = advantages
        
        # Normalize advantages to stabilize training. This is a common practice.
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


    def get_batches(self):
        """
        Generates minibatches of collected data for training.
        Shuffles the data and yields batches.
        """
        # Convert lists of experiences into PyTorch tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.config.DEVICE)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.config.DEVICE)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.config.DEVICE)
        values = torch.tensor(np.array(self.values), dtype=torch.float32, device=self.config.DEVICE)
        
        # Returns and advantages are already tensors from compute_returns_and_advantages
        returns = self.returns
        advantages = self.advantages

        # Get total number of experiences
        batch_size = len(self.states)
        
        # Create indices and shuffle them
        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        # Yield minibatches
        for start_idx in range(0, batch_size, self.config.MINIBATCH_SIZE):
            end_idx = min(start_idx + self.config.MINIBATCH_SIZE, batch_size)
            batch_indices = indices[start_idx:end_idx]

            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                values[batch_indices], # Old value predictions (for value loss debugging if needed)
                returns[batch_indices],
                advantages[batch_indices]
            )

    def clear(self):
        """
        Clears the buffer after a policy update.
        On-policy algorithms like PPO typically use fresh data for each update.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.current_ptr = 0

