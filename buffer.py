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
                                       This tensor is expected to be a 1D tensor of size 1 (e.g., torch.Size([1])).
        """
        # Convert lists to tensors for computation
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.config.DEVICE)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.config.DEVICE)
        dones = torch.tensor(self.dones, dtype=torch.bool, device=self.config.DEVICE)

        # Initialize advantage and return tensors
        # Concatenate values with the last_value for GAE calculation.
        # Both 'values' and 'last_value' should now be 1D tensors, making concatenation smooth.
        extended_values = torch.cat((values, last_value)) # FIX: Removed .unsqueeze(0) here
        
        # GAE calculation
        advantages = torch.zeros_like(rewards, device=self.config.DEVICE)
        last_gae_lam = 0 # Stores the previous GAE value for iterative calculation

        # Iterate backward through the trajectory to compute GAE
        for t in reversed(range(len(rewards))):
            # Determine the value of the next state (V(s_{t+1}))
            # If the episode ended at this step, next_value is 0 for reward calculation
            # and the GAE term is reset.
            if dones[t]: # If the episode terminated at this step 't'
                next_non_terminal = 0.0
                next_value = torch.tensor(0.0, device=self.config.DEVICE) # No future value if done
            else: # Episode did not terminate, use value of next state
                next_non_terminal = 1.0
                # FIX: Index should be t+1 for the next value in extended_values
                # extended_values contains values[0...N-1] and last_value at N
                next_value = extended_values[t+1] 

            # TD error (delta_t) = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
            delta = rewards[t] + self.config.GAMMA * next_value * next_non_terminal - values[t]
            
            # GAE (A_t) = delta_t + gamma * lambda * A_{t+1} * (1 - done_{t+1})
            advantages[t] = last_gae_lam = delta + self.config.GAMMA * self.config.GAE_LAMBDA * next_non_terminal * last_gae_lam
        
        # Calculate returns (target values for the critic)
        # Returns are simply advantages + current state values
        self.returns = advantages + values
        self.advantages = advantages
        
        # Normalize advantages to stabilize training. This is a common practice.
        # Add a small epsilon for numerical stability to prevent division by zero
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

