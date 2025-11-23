#!/usr/bin/env python

# %%capture
# !pip install --upgrade pip
# !pip install gymnasium[atari] ale-py autorom[accept-rom-license] opencv-python

from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import ale_py
import AutoROM

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(),
            nn.Linear(512, 1) # Output a single scalar value
        )

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values
        conv_out = self.conv(x).view(x.size(0), -1)
        policy_logits = self.actor(conv_out)
        value = self.critic(conv_out)
        return policy_logits, value

print("✅ ActorCritic network class defined successfully!")

class A3CAgent:
    def __init__(self, obs_shape, n_actions, lr=1e-4, gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.n_actions = n_actions

        self.model = ActorCritic(obs_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(state)

        # Get probabilities from logits
        policy = F.softmax(policy_logits, dim=-1)

        # Create categorical distribution
        m = torch.distributions.Categorical(policy)

        # Sample an action
        action = m.sample()

        # Return action and its log probability
        return action.item(), m.log_prob(action)

    def compute_losses(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Forward pass through the network
        policy_logits, values = self.model(states)
        _, next_values = self.model(next_states)
        
        target_values = rewards + self.gamma * next_values * (1 - dones)

        advantages = target_values.detach() - values # A = R - V(s)

        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = policy_dist.log_prob(actions)
        policy_loss = -(log_probs * advantages.squeeze()).mean()

        value_loss = F.mse_loss(values, target_values.detach())

        entropy_loss = -policy_dist.entropy().mean()

        total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

        return total_loss

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

print("✅ A3CAgent class defined successfully!")

def preprocess(obs):
    # Convert RGB (210x160x3) to grayscale + resize (84x84)
    import cv2
    obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
    return obs_resized

def train_beamrider(episodes=200, log_interval=10, save_path="beamrider_a3c.pth"):
    from datetime import datetime
    print("Training started ...!")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"beamrider_a3c_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training_log.csv")

    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("episode,total_reward,avg_reward_10,best_reward\n")

    env = gym.make("BeamRiderNoFrameskip-v4", render_mode=None)
    obs, _ = env.reset()
    obs = preprocess(obs)
    state_shape = (4, 84, 84)
    n_actions = env.action_space.n

    agent = A3CAgent(state_shape, n_actions)
    reward_history = []
    best_avg_reward = -float("inf")

    # Frame stack for preprocessing
    frame_stack = deque(maxlen=4)

    for ep in range(episodes):
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        next_states_buffer = []
        dones_buffer = []

        obs, _ = env.reset()
        initial_processed_obs = preprocess(obs)
        frame_stack.clear()
        for _ in range(4):
            frame_stack.append(initial_processed_obs)

        total_reward = 0
        done = False

        while not done:
            current_state = np.stack(frame_stack, axis=0)
            action, log_prob = agent.select_action(current_state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_processed_obs = preprocess(obs)

            # Create next state for buffer
            next_frame_stack_for_buffer = deque(frame_stack, maxlen=4) # Copy current frame_stack
            next_frame_stack_for_buffer.append(next_processed_obs)
            next_state = np.stack(next_frame_stack_for_buffer, axis=0)

            # Store transition for the trajectory
            states_buffer.append(current_state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            next_states_buffer.append(next_state)
            dones_buffer.append(done)

            # Update current frame_stack for next step
            frame_stack.append(next_processed_obs)

            total_reward += reward

        if len(states_buffer) > 0:
            loss = agent.compute_losses(states_buffer, actions_buffer, rewards_buffer, next_states_buffer, dones_buffer)
            agent.optimize(loss)

        reward_history.append(total_reward)

        recent_rewards = reward_history[-10:] if len(reward_history) >= 10 else reward_history
        avg_r_10 = float(np.mean(recent_rewards))

        is_best = avg_r_10 > best_avg_reward
        if is_best:
            best_avg_reward = avg_r_10
            torch.save({
                "policy_net_state_dict": agent.model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "reward_history": reward_history,
                "episode": ep + 1,
                "best_avg_reward": best_avg_reward,
            }, os.path.join(log_dir, "best_model.pth"))

        with open(log_file_path, "a") as f:
            f.write(f"{ep+1},{total_reward},{avg_r_10},{best_avg_reward}\n")

        print(f"Episode {ep+1}: Reward = {total_reward:.1f}, Avg10 = {avg_r_10:.2f}, BestAvg10 = {best_avg_reward:.2f}")

        if (ep + 1) % log_interval == 0:
            ckpt_path = os.path.join(log_dir, f"checkpoint_ep{ep+1}.pth")
            torch.save({
                "policy_net_state_dict": agent.model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "reward_history": reward_history,
                "episode": ep + 1,
                "best_avg_reward": best_avg_reward,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    env.close()
    plt.plot(reward_history)
    plt.title("A3C on BeamRider-v4")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

train_beamrider(episodes=100, log_interval=10)

log_file_path = "runs/beamrider_a3c_20251120_165144/training_log.csv"
log_df = pd.read_csv(log_file_path)

plt.figure(figsize=(10, 6))
plt.plot(log_df["episode"], log_df["avg_reward_10"], label="Avg 10-Episode Reward")
plt.title("A3C Training Progress on BeamRider-v4")
plt.xlabel("Episode")
plt.ylabel("Average 10-Episode Reward")
plt.legend()
plt.grid(True)
plt.show()