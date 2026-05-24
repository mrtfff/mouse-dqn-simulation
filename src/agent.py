import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class QNetwork(nn.Module):
    """Nöron çizimine uygun optimize edilmiş Sinir Ağı"""
    def __init__(self, state_dim=3, action_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim=3, action_dim=2, agent_id=0):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def get_q_values(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t).cpu().numpy()[0]
        return q_values

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q = self.model(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            next_q = self.model(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (self.gamma * next_q * (1 - dones_t))
            
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 1.0)