import random
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """Deep Q-Network for scoring experiment candidate actions."""
    def __init__(self, state_dim: int = 8, hidden_dim: int = 64, max_actions: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNPolicy:
    """
    Deep Q-Learning Policy for Intelligent Experiment Selection.
    """
    def __init__(
        self,
        state_dim: int = 8,
        hidden_dim: int = 64,
        max_actions: int = 10,
        lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.92,
        memory_capacity: int = 1000,
        batch_size: int = 16,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.max_actions = max_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.q_net = QNetwork(state_dim, hidden_dim, max_actions).to(self.device)
        self.target_net = QNetwork(state_dim, hidden_dim, max_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=memory_capacity)
        self.total_steps = 0

    def select_action(self, state: np.ndarray, num_available_actions: int) -> int:
        """Selects action using epsilon-greedy strategy."""
        num_valid = min(self.max_actions, max(1, num_available_actions))
        
        if random.random() < self.epsilon:
            return random.randint(0, num_valid - 1)
            
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)[0, :num_valid]
            return int(torch.argmax(q_values).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ):
        self.memory.append((state, action, reward, next_state, done))

    def update(
        self,
        state: Optional[np.ndarray] = None,
        action: Optional[int] = None,
        reward: Optional[float] = None,
        next_state: Optional[np.ndarray] = None,
        done: bool = False
    ) -> Optional[float]:
        """Performs a mini-batch Q-learning optimization step."""
        if state is not None and action is not None and reward is not None and next_state is not None:
            self.store_transition(state, action, reward, next_state, done)
            
        if len(self.memory) < self.batch_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        s_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        a_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        r_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        ns_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        d_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        curr_q = self.q_net(s_t).gather(1, a_t)
        
        with torch.no_grad():
            next_q = self.target_net(ns_t).max(1)[0].unsqueeze(1)
            target_q = r_t + (1.0 - d_t) * self.gamma * next_q
            
        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_steps += 1
        if self.total_steps % 10 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())
