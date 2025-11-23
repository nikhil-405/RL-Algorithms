import numpy as np
import torch
import torch.nn as nn
import random
import os

# Set the path to your trained model here
MODEL_PATH = r"best_model.pth"

# Define the Connect4 environment
class Connect4:
    def __init__(self):
        self.rows, self.cols = 6, 7
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.done = False
        self.current_player = 1
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

    def step(self, action, player=1):
        """Apply a move for the specified player without automatic opponent response"""
        if self.done:
            return self.board.copy(), 0, True, {}

        # Apply the move
        if action not in self.available_actions():
             return self.board.copy(), -1.0, True, {"illegal": True}
        
        self.drop(action, player)

        # Check win
        if self.check_win(player):
            self.done = True
            reward = 1.0 if player == 1 else -1.0
            return self.board.copy(), reward, True, {"winner": player}

        # Check draw
        if self.is_draw():
            self.done = True
            return self.board.copy(), 0.0, True, {"winner": 0}

        return self.board.copy(), 0.0, False, {}

# Define the PPO Model
class PPO_Model(nn.Module):
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
            mask = torch.ones(logits.size(-1)) * float('-inf')
            mask[available_actions] = 0
            logits = logits + mask
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

# Helper functions
def flatten(obs):
    return torch.tensor(obs.flatten(), dtype=torch.float32)

def render(board):
    print("\nBoard:")
    chars = {0: '.', 1: 'X', 2: 'O'}
    for r in board:
        print(" ".join(chars[c] for c in r))
    print("0 1 2 3 4 5 6")

# Load the trained model
model = PPO_Model()

if os.path.exists(MODEL_PATH):
    print(f"Loading model from: {MODEL_PATH}")
    try:
        # Try loading as state dict directly
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        # Fallback if it was saved as a checkpoint dict
        checkpoint = torch.load(MODEL_PATH)
        if "policy_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["policy_state_dict"])
        else:
            model.load_state_dict(checkpoint)
else:
    print(f"Model not found at {MODEL_PATH}. Using random weights.")

model.eval()
print("Model loaded successfully!")

def agent_move(board, available_actions):
    obs = flatten(board)
    with torch.no_grad():
        action, _, _ = model.act(obs, available_actions=available_actions)
    return action

def play_vs_agent():
    env = Connect4()
    board = env.reset()
    
    print("\n=== Connect4 vs AI Agent ===")
    print("Enter column numbers 0-6, or 'q' to quit")
    print("You are 'O' (Player 2), Agent is 'X' (Player 1)\n")
    render(board)
    
    while True:
        # Agent Turn (Player 1 - X)
        available = env.available_actions()
        a = agent_move(board, available)
        print(f"\nAgent plays column {a}")
        board, r, d, info = env.step(a, player=1)
        render(board)
        if d:
            if info.get("illegal"):
                print("⚠️ Agent made illegal move! Trying again...")
                env.done = False  # Reset done flag
                continue
            print("\n🎮 Agent wins!" if r>0 else "\n🤝 Draw!")
            break

        # Human Turn (Player 2 - O)
        try:
            user_input = input("\nYour Move (0-6, or 'q' to quit): ").strip().lower()
            if user_input == 'q':
                print("Thanks for playing!")
                break
            human = int(user_input)
            if human < 0 or human > 6:
                print("Invalid column! Enter 0-6")
                continue
        except ValueError:
            print("Invalid input! Enter a number 0-6 or 'q' to quit")
            continue
        except KeyboardInterrupt:
            print("\n\nThanks for playing!")
            break
            
        board, r, d, _ = env.step(human, player=2)
        render(board)
        if d:
            if "illegal" in _:
                print("❌ Illegal move! Column is full. Try another column.")
                env.done = False  # Reset done flag to continue game
                continue
            print("\n🎉 You win!" if r < 0 else "\n🤝 Draw!")
            break

if __name__ == "__main__":
    play_vs_agent()