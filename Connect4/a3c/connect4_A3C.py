import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def flatten(obs):
    return torch.tensor(obs.flatten(), dtype=torch.float32)

class A3C_Model(nn.Module):
    def __init__(self, input_dim=42, n_actions=7):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

    def act(self, obs, available_actions=None):
        logits, value = self.forward(obs)
        
        # Mask illegal actions
        if available_actions is not None:
            if len(available_actions) == 0:
                available_actions = list(range(logits.size(-1)))
            mask = torch.ones(logits.size(-1)) * float('-inf')
            mask[available_actions] = 0
            logits = logits + mask
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

class SelfPlayOpponent:
    """
    Stores older snapshots of the agent and samples from them.
    If empty, plays random.
    """
    def __init__(self):
        self.opponents = []

    def add(self, model_state):
        self.opponents.append(model_state)

    def select_action(self, obs, model_class):
        if len(self.opponents) == 0:
            # Random if no opponent yet
            legal = [i for i in range(7) if obs[0, i] == 0]
            return np.random.choice(legal)

        # Sample a random past model
        snapshot = np.random.choice(self.opponents)
        opponent_model = model_class()
        opponent_model.load_state_dict(snapshot)
        opponent_model.eval()

        obs_flat = flatten(obs)
        logits, _ = opponent_model(obs_flat)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs).item()

        # Ensure legal
        legal = [i for i in range(7) if obs[0, i] == 0]
        if action not in legal:
            action = np.random.choice(legal)
        return action

class Connect4Env:
    def __init__(self, opponent=None):
        self.rows, self.cols = 6, 7
        self.opponent = opponent 
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.done = False
        return self.board.copy()

    def available_actions(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def drop(self, col, player):
        if self.board[0, col] != 0:
            return False
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = player
                return True
        return False

    def check_win(self, p):
        b = self.board
        R, C = self.rows, self.cols

        # Horizontal
        for r in range(R):
            for c in range(C - 3):
                if np.all(b[r, c:c + 4] == p):
                    return True

        # Vertical
        for r in range(R - 3):
            for c in range(C):
                if np.all(b[r:r + 4, c] == p):
                    return True

        # Diagonal down-right
        for r in range(R - 3):
            for c in range(C - 3):
                if all(b[r + i, c + i] == p for i in range(4)):
                    return True

        # Diagonal up-right
        for r in range(3, R):
            for c in range(C - 3):
                if all(b[r - i, c + i] == p for i in range(4)):
                    return True

        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def three_in_row(self, p):
        b = self.board
        R, C = self.rows, self.cols
        count = 0

        for r in range(R):
            for c in range(C - 2):
                if np.sum(b[r, c:c+3] == p) == 3:
                    count += 1

        for r in range(R - 2):
            for c in range(C):
                if np.sum(b[r:r+3, c] == p) == 3:
                    count += 1

        for r in range(R - 2):
            for c in range(C - 2):
                if all(b[r + i, c + i] == p for i in range(3)):
                    count += 1

        for r in range(2, R):
            for c in range(C - 2):
                if all(b[r - i, c + i] == p for i in range(3)):
                    count += 1

        return count

    def step(self, action):
        # ----- Agent Move -----
        # Check legality first
        if action not in self.available_actions():
            # Illegal move: penalize and retry (do not end episode, do not let opponent play)
            return self.board.copy(), -0.5, False, {"illegal": True}

        self.drop(action, 1)

        # Shaping: 3-in-a-row
        reward = 0.1 * self.three_in_row(1)
        reward -= 0.1 * self.three_in_row(2)

        # Win?
        if self.check_win(1):
            return self.board.copy(), reward + 1.0, True, {"winner": 1}

        # Draw?
        if self.is_draw():
            return self.board.copy(), reward, True, {"winner": 0}

        # ----- Opponent Move -----
        if self.opponent:
            if isinstance(self.opponent, SelfPlayOpponent):
                opp_action = self.opponent.select_action(self.board.copy(), A3C_Model)
            else:
                # Fallback random
                avail = self.available_actions()
                if not avail: # Should be covered by is_draw, but safety check
                     return self.board.copy(), reward, True, {"winner": 0}
                opp_action = np.random.choice(avail)

            self.drop(opp_action, 2)

            # Opponent shaping
            reward -= 0.1 * self.three_in_row(2)

            # Loss?
            if self.check_win(2):
                return self.board.copy(), reward - 1.0, True, {"winner": 2}

            # Draw?
            if self.is_draw():
                return self.board.copy(), reward, True, {"winner": 0}

        return self.board.copy(), reward, False, {}

gamma = 0.99
lam = 0.95
lr = 3e-4
entropy_coef = 0.05  # Increase for exploration

def save_checkpoint(model, ep, run_dir):
    filename = os.path.join(run_dir, f"checkpoint_ep{ep}.pth")
    torch.save(model.state_dict(), filename)
    print(f" Saved checkpoint: {filename}")

def save_best_model(model, best_model_path):
    torch.save(model.state_dict(), best_model_path)
    print(f" New best model saved to {best_model_path}")

def log_to_csv(log_file, ep, reward, avg10, best):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ep, reward, avg10, best])

def compute_gae(rewards, values, dones):
    advantages, gae = [], 0
    
    values = [v.detach().item() if hasattr(v, 'detach') else float(v) for v in values]
    values = values + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + values[i] for i, adv in enumerate(advantages)]
    
    return [torch.tensor(a, dtype=torch.float32) for a in advantages], \
           [torch.tensor(r, dtype=torch.float32) for r in returns]

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class Worker(mp.Process):
    def __init__(self, global_model, optimizer, rank, global_ep, res_queue, run_dir, max_episodes):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.rank = rank
        self.global_ep = global_ep
        self.res_queue = res_queue
        self.run_dir = run_dir
        self.max_episodes = max_episodes
        
        # Local env and model
        self.env = Connect4Env(opponent=SelfPlayOpponent()) 
        self.local_model = A3C_Model()
        
    def run(self):
        total_step = 1
        while self.global_ep.value < self.max_episodes:
            self.local_model.load_state_dict(self.global_model.state_dict())
            
            obs = self.env.reset()
            done = False
            ep_reward = 0
            
            states, actions, logprobs, vals, rewards, dones = [], [], [], [], [], []
            
            while not done:
                s = flatten(obs)
                available_actions = self.env.available_actions()
                a, lp, v = self.local_model.act(s, available_actions=available_actions)
                
                states.append(s)
                actions.append(a)
                logprobs.append(lp)
                vals.append(v)
                
                obs, r, done, info = self.env.step(a)
                rewards.append(r)
                dones.append(done)
                
                ep_reward += r
                total_step += 1
            
            # Calculate GAE and Loss
            vals_detached = [v.detach() for v in vals]
            adv, ret = compute_gae(rewards, vals_detached, dones)
            
            adv = torch.stack(adv).detach()
            ret = torch.stack(ret).detach()
            
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            else:
                adv = adv - adv.mean()
            
            logprobs_stack = torch.stack(logprobs)
            vals_stack = torch.stack(vals).squeeze()
            
            policy_loss = -(logprobs_stack * adv).mean()
            value_loss = F.mse_loss(vals_stack, ret)
            
            states_stack = torch.stack(states)
            logits, _ = self.local_model(states_stack)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            
            # Update Global Model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 0.5)
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()
            
            # Update global episode count
            with self.global_ep.get_lock():
                self.global_ep.value += 1
                current_ep = self.global_ep.value
                
            self.res_queue.put(ep_reward)
            
            if self.rank == 0:
                if current_ep % 10 == 0:
                    print(f"Episode {current_ep}: Reward={ep_reward:.2f}")
                
                if current_ep % 100 == 0:
                    save_path = os.path.join(self.run_dir, f"checkpoint_ep{current_ep}.pth")
                    torch.save(self.global_model.state_dict(), save_path)

def train_connect4_a3c(episodes=1000, n_workers=4):
    # Run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"connect4_A3C_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training_log.csv")

    # Initialize CSV
    with open(log_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "avg_reward_10", "best_reward"])
    
    global_model = A3C_Model()
    global_model.share_memory()
    
    optimizer = SharedAdam(global_model.parameters(), lr=lr)
    
    global_ep = mp.Value('i', 0)
    res_queue = mp.Queue()
    
    workers = [Worker(global_model, optimizer, i, global_ep, res_queue, log_dir, episodes) for i in range(n_workers)]
    
    [w.start() for w in workers]
    
    res = []
    best_avg_reward = -float("inf")
    
    while global_ep.value < episodes:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            current_ep = len(res)
            avg10 = np.mean(res[-10:])
            
            if avg10 > best_avg_reward:
                best_avg_reward = avg10
            
            # Log to CSV
            with open(log_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_ep, r, avg10, best_avg_reward])
            
    [w.join() for w in workers]
    
    # Save final model
    torch.save(global_model.state_dict(), os.path.join(log_dir, "best_model.pth"))
    
    return res, np.mean(res[-100:])

if __name__ == '__main__':
    
    reward_history, best_avg_reward = train_connect4_a3c(episodes=1000, n_workers=4)

    # Plotting logic
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        print("No 'runs' directory found.")
    else:
        # Get all subdirectories in runs/
        subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        
        if not subdirs:
            print("No run directories found in 'runs/'.")
        else:
            # Sort by modification time (latest last)
            latest_run = max(subdirs, key=os.path.getmtime)
            print(f"Analyzing latest run: {latest_run}")
            
            log_file = os.path.join(latest_run, "training_log.csv")
            
            if not os.path.exists(log_file):
                print(f"No training_log.csv found in {latest_run}")
            else:
                try:
                    df = pd.read_csv(log_file)
                    
                    # Plotting
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot Avg Reward (Last 10)
                    ax.plot(df['episode'], df['avg_reward_10'], label='Avg Reward (Last 10)', color='blue', alpha=0.7)
                    
                    # Plot Best Avg Reward
                    ax.plot(df['episode'], df['best_reward'], label='Best Avg Reward', color='green', linestyle='--', linewidth=2)
                    
                    ax.set_title(f"Connect4 A3C")
                    ax.set_xlabel("Episode")
                    ax.set_ylabel("Reward")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Save and Display
                    plot_path = os.path.join(latest_run, "training_plot.png")
                    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"Plot saved to {plot_path}")
                    
                except Exception as e:
                    print(f"Error reading or plotting log: {e}")
