# ppo_project/ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import datetime

from config import PPOConfig
from networks import ActorNetwork, CriticNetwork
from buffer import RolloutBuffer
from utils import make_env, set_seeds

class PPOAgent:
    """
    Implements the Proximal Policy Optimization (PPO) agent.
    This class handles environment interaction, data collection,
    network updates, and logging.
    """
    def __init__(self, config: PPOConfig):
        self.config = config
        self.env = make_env(self.config.ENV_NAME, self.config.SEED)
        set_seeds(self.config.SEED)

        # Get observation and action space dimensions from the environment
        self.observation_dim = self.env.observation_space.shape[0]
        # For continuous action spaces, action_dim is the number of continuous actions
        self.action_dim = self.env.action_space.shape[0]

        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(
            self.observation_dim, self.action_dim, self.config.HIDDEN_SIZES
        ).to(self.config.DEVICE)
        self.critic = CriticNetwork(
            self.observation_dim, self.config.HIDDEN_SIZES
        ).to(self.config.DEVICE)

        # Initialize optimizers for Actor and Critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.CRITIC_LR)

        # Initialize the RolloutBuffer
        self.buffer = RolloutBuffer(self.config)

        # Setup TensorBoard writer for logging training progress
        log_dir_path = os.path.join(self.config.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{self.config.ENV_NAME}")
        self.writer = SummaryWriter(log_dir=log_dir_path)
        print(f"TensorBoard logs will be saved to: {log_dir_path}")

        self.global_step = 0 # Counter for total environment interactions

    def collect_rollout(self):
        """
        Collects a trajectory (rollout) of experiences from the environment
        for a specified number of steps (self.config.N_STEPS).
        This method is currently only called by the evaluation function,
        the main training loop directly implements data collection for more control.
        """
        # Reset buffer before collecting new data
        self.buffer.clear()
        
        # Get initial state
        state, info = self.env.reset(seed=self.config.SEED)
        
        # Loop to collect N_STEPS experiences
        for _ in range(self.config.N_STEPS):
            # global_step is incremented in the train loop, not here
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)

            # Get action distribution and value prediction from networks
            with torch.no_grad(): # No need to compute gradients during data collection
                policy_dist = self.actor(state_tensor)
                value = self.critic(state_tensor).item() # FIX: Use .item() to get scalar value

            # Sample action from the policy distribution
            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action).sum(axis=-1) # Sum log_probs for multi-dimensional actions

            # Clamp actions to environment's action space bounds if necessary
            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)


            # Take action in the environment
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Store the experience in the buffer
            self.buffer.add(state, action_np, reward, done, value, log_prob.item())

            state = next_state # Update current state

            # If episode ends, reset environment and continue collecting
            if done:
                state, info = self.env.reset(seed=self.config.SEED)
                
        # After collecting N_STEPS, get the value of the last state
        last_state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            last_value = self.critic(last_state_tensor).squeeze(0) # Keep on device, as it's used in buffer computations

        # Compute returns and advantages based on the collected rollout
        self.buffer.compute_returns_and_advantages(last_value)

    def update_networks(self):
        """
        Performs the PPO network updates using the data in the rollout buffer.
        Iterates over the data for N_EPOCHS using mini-batches.
        """
        # Get batches from the buffer
        data_batches = list(self.buffer.get_batches()) # Convert to list to iterate multiple times

        # Store losses for logging
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []

        # Perform N_EPOCHS of optimization
        for epoch in range(self.config.N_EPOCHS):
            # Shuffle batches for each epoch to ensure diverse mini-batches
            np.random.shuffle(data_batches) 

            for states, actions, old_log_probs, old_values, returns, advantages in data_batches:
                # --- Update Critic Network ---
                # Get current value predictions from the critic
                current_values = self.critic(states).squeeze(-1) # Ensure values are 1D: (MINIBATCH_SIZE,)

                # Critic loss: Mean Squared Error between predicted values and computed returns
                # Both current_values and returns should now be (MINIBATCH_SIZE,)
                critic_loss = nn.functional.mse_loss(current_values, returns)
                
                # Backpropagation for critic
                self.critic_optimizer.zero_grad() # Clear previous gradients
                critic_loss.backward()            # Compute gradients
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # Clip gradients to prevent explosion
                self.critic_optimizer.step()      # Update network parameters
                
                critic_losses.append(critic_loss.item()) # Store loss value

                # --- Update Actor Network ---
                # Get current policy distribution and log probabilities
                current_policy_dist = self.actor(states)
                current_log_probs = current_policy_dist.log_prob(actions).sum(axis=-1) # Sum for continuous actions

                # Calculate the ratio of new policy to old policy probabilities
                # ratio = pi_theta(a|s) / pi_theta_old(a|s)
                ratio = torch.exp(current_log_probs - old_log_probs)

                # Clipped surrogate objective components
                # Term 1: r_t * A_t
                surr1 = ratio * advantages
                # Term 2: clip(r_t, 1-eps, 1+eps) * A_t
                surr2 = torch.clamp(ratio, 1.0 - self.config.CLIP_EPSILON, 1.0 + self.config.CLIP_EPSILON) * advantages

                # PPO policy loss: take the minimum of the two surrogate objectives
                # We negate because PyTorch optimizers perform gradient *descent*,
                # but PPO's objective is a maximization problem.
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy regularization for exploration
                # Adds a bonus proportional to the policy's entropy
                entropy = current_policy_dist.entropy().mean()
                
                # Total actor loss (policy loss - entropy bonus)
                actor_loss = policy_loss - self.config.ENTROPY_COEFF * entropy

                # Backpropagation for actor
                self.actor_optimizer.zero_grad() # Clear previous gradients
                actor_loss.backward()             # Compute gradients
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # Clip gradients
                self.actor_optimizer.step()       # Update network parameters

                actor_losses.append(policy_loss.item()) # Store policy loss (excluding entropy term)
                entropies.append(entropy.item())

                # Calculate approximate fraction of samples clipped for monitoring
                clipped = (ratio < 1 - self.config.CLIP_EPSILON) | (ratio > 1 + self.config.CLIP_EPSILON)
                clip_fractions.append(clipped.float().mean().item())


        # Log average losses and metrics for this update step to TensorBoard
        self.writer.add_scalar("Loss/Actor_Loss", np.mean(actor_losses), self.global_step)
        self.writer.add_scalar("Loss/Critic_Loss", np.mean(critic_losses), self.global_step)
        self.writer.add_scalar("Metrics/Entropy", np.mean(entropies), self.global_step)
        self.writer.add_scalar("Metrics/Clip_Fraction", np.mean(clip_fractions), self.global_step)

    def train(self):
        """
        Main training loop for the PPO agent.
        """
        print(f"Starting training for {self.config.ENV_NAME} with {self.config.TOTAL_TIMESTEPS} total timesteps.")
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        episode_count = 0

        # Initial environment reset for the training loop
        state, info = self.env.reset(seed=self.config.SEED)
        
        while self.global_step < self.config.TOTAL_TIMESTEPS:
            # Store global step at the start of the rollout collection for logging interval
            initial_global_step = self.global_step
            
            # Clear the buffer at the start of each new data collection segment
            self.buffer.clear() 
            
            # Collect data for N_STEPS (or until total_timesteps is reached)
            for _ in range(self.config.N_STEPS):
                if self.global_step >= self.config.TOTAL_TIMESTEPS:
                    break

                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)

                with torch.no_grad():
                    policy_dist = self.actor(state_tensor)
                    # FIX: Use .item() to get a scalar value from the critic's prediction
                    value = self.critic(state_tensor).item() 

                action = policy_dist.sample()
                # Sum log probabilities for multi-dimensional continuous actions
                log_prob = policy_dist.log_prob(action).sum(axis=-1)

                action_np = action.squeeze(0).cpu().numpy()
                # Clamp actions to be within the environment's valid range
                action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

                next_state, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                # Add the collected experience to the buffer
                self.buffer.add(state, action_np, reward, done, value, log_prob.item())

                state = next_state # Update current state for next step
                current_episode_reward += reward
                current_episode_length += 1
                self.global_step += 1 # Increment global step counter

                if done:
                    # If episode terminated, log episode stats
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    episode_count += 1
                    
                    self.writer.add_scalar("Episode/Reward", current_episode_reward, episode_count)
                    self.writer.add_scalar("Episode/Length", current_episode_length, episode_count)
                    
                    print(f"Global Step: {self.global_step}, Episode: {episode_count}, Reward: {current_episode_reward:.2f}, Length: {current_episode_length}")
                    
                    # Reset episode tracking variables and environment
                    current_episode_reward = 0
                    current_episode_length = 0
                    state, info = self.env.reset(seed=self.config.SEED) 

            # After collecting N_STEPS (or hitting TOTAL_TIMESTEPS),
            # get the value of the last state for GAE bootstrapping.
            # If the last step in the buffer was a terminal state, last_value remains 0.0.
            last_value = torch.tensor([0.0], dtype=torch.float32, device=self.config.DEVICE) 
            if self.buffer.current_ptr > 0 and not self.buffer.dones[-1]: 
                last_state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)
                with torch.no_grad():
                    last_value = self.critic(last_state_tensor).squeeze(0) # Keep as a tensor on device

            # Compute returns and advantages for the collected rollout
            self.buffer.compute_returns_and_advantages(last_value)

            # Perform network updates using the collected data
            self.update_networks()

            # Logging average rewards and lengths for the update interval to TensorBoard
            # This logs a rolling average over the last N_STEPS or N_EPISODES collected
            if (self.global_step - initial_global_step) % self.config.LOG_INTERVAL == 0:
                if episode_count > 0: 
                    # Calculate average reward/length over a recent window of episodes
                    # or just for the episodes that completed within this N_STEPS rollout.
                    # For simplicity, using all collected episodes so far, or the last LOG_INTERVAL ones.
                    avg_reward_window = np.mean(episode_rewards[-self.config.LOG_INTERVAL:] if len(episode_rewards) >= self.config.LOG_INTERVAL else episode_rewards)
                    avg_length_window = np.mean(episode_lengths[-self.config.LOG_INTERVAL:] if len(episode_lengths) >= self.config.LOG_INTERVAL else episode_lengths)
                    self.writer.add_scalar("Training/Avg_Reward_Window", avg_reward_window, self.global_step)
                    self.writer.add_scalar("Training/Avg_Length_Window", avg_length_window, self.global_step)
            
            # Save the model periodically based on global steps / N_STEPS (i.e., every few updates)
            if (self.global_step // self.config.N_STEPS) % self.config.SAVE_INTERVAL == 0 and self.global_step > 0:
                self.save_model(os.path.join(self.config.SAVE_DIR, f"ppo_agent_step_{self.global_step}.pth"))

        print("Training finished.")
        self.writer.close() # Close the TensorBoard writer
        self.env.close()    # Close the environment

    def save_model(self, path):
        """Saves the actor and critic model states."""
        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'global_step': self.global_step,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Loads the actor and critic model states."""
        checkpoint = torch.load(path, map_location=self.config.DEVICE) # Load to the correct device
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"Model loaded from {path} at global step {self.global_step}")

    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluates the trained agent for a specified number of episodes.
        """
        print(f"\n--- Starting evaluation for {num_episodes} episodes ---")
        # Create a new environment for evaluation to ensure isolation and fresh state
        eval_env = make_env(self.config.ENV_NAME, self.config.SEED + 100) # Use a different seed for evaluation
        
        total_rewards = []
        for i in range(num_episodes):
            state, info = eval_env.reset() # Reset evaluation environment
            episode_reward = 0
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)
                
                with torch.no_grad(): # Evaluation doesn't need gradient computation
                    policy_dist = self.actor(state_tensor)
                    action = policy_dist.sample() # Sample action from the policy
                
                action_np = action.squeeze(0).cpu().numpy()
                # Clamp actions to environment's bounds, essential for continuous control
                action_np = np.clip(action_np, eval_env.action_space.low, eval_env.action_space.high) 

                next_state, reward, terminated, truncated, info = eval_env.step(action_np)
                done = terminated or truncated

                episode_reward += reward
                state = next_state
                
                # If rendering is desired, ensure env.make() in make_env had render_mode="human"
                # if render:
                #     eval_env.render() # Uncomment if you've set up rendering

            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {i+1}: Reward = {episode_reward:.2f}")

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"--- Evaluation Complete: Avg Reward = {avg_reward:.2f} +/- {std_reward:.2f} ---")
        eval_env.close() # Close evaluation environment
        return avg_reward, std_reward

