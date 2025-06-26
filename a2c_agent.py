# ppo_project/a2c_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import datetime

from config import PPOConfig # We'll reuse PPOConfig, but define A2C specific params
from networks import ActorNetwork, CriticNetwork
from buffer import RolloutBuffer
from utils import make_env, set_seeds

class A2CAgent:
    """
    Implements the Advantage Actor-Critic (A2C) agent.
    This class handles environment interaction, data collection,
    network updates, and logging, serving as a baseline for PPO.
    """
    def __init__(self, config: PPOConfig): # Still uses PPOConfig for shared parameters
        self.config = config
        self.env = make_env(self.config.ENV_NAME, self.config.SEED)
        set_seeds(self.config.SEED)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Actor and Critic networks are the same architecture as PPO
        self.actor = ActorNetwork(
            self.observation_dim, self.action_dim, self.config.HIDDEN_SIZES
        ).to(self.config.DEVICE)
        self.critic = CriticNetwork(
            self.observation_dim, self.config.HIDDEN_SIZES
        ).to(self.config.DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.CRITIC_LR)

        # RolloutBuffer is also reused (GAE is common in A2C too)
        self.buffer = RolloutBuffer(self.config)

        # Setup TensorBoard writer for logging training progress
        log_dir_path = os.path.join(self.config.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-A2C-{self.config.ENV_NAME}")
        self.writer = SummaryWriter(log_dir=log_dir_path)
        print(f"TensorBoard logs for A2C will be saved to: {log_dir_path}")

        self.global_step = 0

    def collect_rollout(self):
        """
        Collects a trajectory (rollout) of experiences from the environment
        for a specified number of steps (self.config.N_STEPS).
        This method is identical to PPO's data collection.
        """
        self.buffer.clear()
        state, info = self.env.reset(seed=self.config.SEED)
        
        for _ in range(self.config.N_STEPS):
            self.global_step += 1
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)

            with torch.no_grad():
                policy_dist = self.actor(state_tensor)
                value = self.critic(state_tensor).item() 

            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action).sum(axis=-1) 

            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(state, action_np, reward, done, value, log_prob.item())

            state = next_state
            if done:
                state, info = self.env.reset(seed=self.config.SEED)
                
        last_state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            last_value = self.critic(last_state_tensor).squeeze(0)

        self.buffer.compute_returns_and_advantages(last_value)

    def update_networks(self):
        """
        Performs the A2C network updates using the data in the rollout buffer.
        A2C typically performs a single optimization pass over the collected data.
        """
        # Get batches from the buffer (A2C usually uses one big batch, or minibatches)
        # For simplicity and consistency with PPO's buffer, we still iterate through batches
        # but the effective N_EPOCHS will be set to 1 in config for A2C.
        data_batches = list(self.buffer.get_batches())

        actor_losses = []
        critic_losses = []
        entropies = []
        # No clip_fractions for A2C as there's no clipping mechanism.

        # A2C typically performs 1 epoch of optimization over the collected data.
        # This loop will technically run for self.config.N_EPOCHS, which should be 1 for A2C.
        for epoch in range(self.config.N_EPOCHS): 
            np.random.shuffle(data_batches) 

            for states, actions, old_log_probs, old_values, returns, advantages in data_batches:
                # --- Update Critic Network (identical to PPO) ---
                current_values = self.critic(states).squeeze(-1) 
                critic_loss = nn.functional.mse_loss(current_values, returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) 
                self.critic_optimizer.step()
                
                critic_losses.append(critic_loss.item())

                # --- Update Actor Network (A2C specific policy loss) ---
                current_policy_dist = self.actor(states)
                current_log_probs = current_policy_dist.log_prob(actions).sum(axis=-1)

                # A2C Policy Loss: -log_prob * advantage
                # We negate because it's gradient descent on a maximization objective
                policy_loss = -(current_log_probs * advantages).mean()

                # Entropy regularization for exploration (identical to PPO)
                entropy = current_policy_dist.entropy().mean()
                
                # Total actor loss: policy loss - entropy bonus
                actor_loss = policy_loss - self.config.ENTROPY_COEFF * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) 
                self.actor_optimizer.step()

                actor_losses.append(policy_loss.item())
                entropies.append(entropy.item())

        # Log average losses and metrics
        self.writer.add_scalar("Loss/Actor_Loss", np.mean(actor_losses), self.global_step)
        self.writer.add_scalar("Loss/Critic_Loss", np.mean(critic_losses), self.global_step)
        self.writer.add_scalar("Metrics/Entropy", np.mean(entropies), self.global_step)
        # No clip_fraction for A2C

    def train(self):
        """
        Main training loop for the A2C agent.
        This loop is identical to PPO's, handling data collection and updates.
        """
        print(f"Starting training for A2C on {self.config.ENV_NAME} with {self.config.TOTAL_TIMESTEPS} total timesteps.")
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        episode_count = 0

        state, info = self.env.reset(seed=self.config.SEED)
        
        while self.global_step < self.config.TOTAL_TIMESTEPS:
            initial_global_step = self.global_step
            self.buffer.clear() 
            
            for _ in range(self.config.N_STEPS):
                if self.global_step >= self.config.TOTAL_TIMESTEPS:
                    break

                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)

                with torch.no_grad():
                    policy_dist = self.actor(state_tensor)
                    value = self.critic(state_tensor).item() 

                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action).sum(axis=-1)

                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

                next_state, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.add(state, action_np, reward, done, value, log_prob.item())

                state = next_state
                current_episode_reward += reward
                current_episode_length += 1
                self.global_step += 1

                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    episode_count += 1
                    
                    self.writer.add_scalar("Episode/Reward", current_episode_reward, episode_count)
                    self.writer.add_scalar("Episode/Length", current_episode_length, episode_count)
                    
                    print(f"Global Step: {self.global_step}, Episode: {episode_count}, Reward: {current_episode_reward:.2f}, Length: {current_episode_length}")
                    
                    current_episode_reward = 0
                    current_episode_length = 0
                    state, info = self.env.reset(seed=self.config.SEED) 

            last_value = torch.tensor([0.0], dtype=torch.float32, device=self.config.DEVICE) 
            if self.buffer.current_ptr > 0 and not self.buffer.dones[-1]: 
                last_state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)
                with torch.no_grad():
                    last_value = self.critic(last_state_tensor).squeeze(0) 
            
            self.buffer.compute_returns_and_advantages(last_value)
            self.update_networks()

            if (self.global_step - initial_global_step) % self.config.LOG_INTERVAL == 0:
                if episode_count > 0: 
                    avg_reward_window = np.mean(episode_rewards[-self.config.LOG_INTERVAL:] if len(episode_rewards) >= self.config.LOG_INTERVAL else episode_rewards)
                    avg_length_window = np.mean(episode_lengths[-self.config.LOG_INTERVAL:] if len(episode_lengths) >= self.config.LOG_INTERVAL else episode_lengths)
                    self.writer.add_scalar("Training/Avg_Reward_Window", avg_reward_window, self.global_step)
                    self.writer.add_scalar("Training/Avg_Length_Window", avg_length_window, self.global_step)
            
            if (self.global_step // self.config.N_STEPS) % self.config.SAVE_INTERVAL == 0 and self.global_step > 0:
                self.save_model(os.path.join(self.config.SAVE_DIR, f"a2c_agent_step_{self.global_step}.pth"))

        print("Training finished.")
        self.writer.close()
        self.env.close()

    def save_model(self, path):
        """Saves the actor and critic model states."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'global_step': self.global_step,
        }, path)
        print(f"A2C Model saved to {path}")

    def load_model(self, path):
        """Loads the actor and critic model states."""
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"A2C Model loaded from {path} at global step {self.global_step}")

    def evaluate(self, num_episodes=10, render=True):
        """
        Evaluates the trained A2C agent for a specified number of episodes.
        """
        print(f"\n--- Starting A2C evaluation for {num_episodes} episodes ---")
        eval_env = make_env(self.config.ENV_NAME, self.config.SEED + 100) 
        
        total_rewards = []
        for i in range(num_episodes):
            state, info = eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    policy_dist = self.actor(state_tensor)
                    action = policy_dist.sample() 
                
                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, eval_env.action_space.low, eval_env.action_space.high) 

                next_state, reward, terminated, truncated, info = eval_env.step(action_np)
                done = terminated or truncated

                episode_reward += reward
                state = next_state
                
                # if render:
                #     eval_env.render()

            total_rewards.append(episode_reward)
            print(f"A2C Evaluation Episode {i+1}: Reward = {episode_reward:.2f}")

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"--- A2C Evaluation Complete: Avg Reward = {avg_reward:.2f} +/- {std_reward:.2f} ---")
        eval_env.close()
        return avg_reward, std_reward

