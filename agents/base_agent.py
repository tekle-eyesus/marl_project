# type: ignore
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BaseAgent:
    def __init__(self, obs_space, action_space, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.obs_size = obs_space.shape[0]
        self.action_size = action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.update_target()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        ).to(self.device)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        action = torch.argmax(q_values).item()  # Convert tensor -> Python int

        # Make sure it's in [0, action_dim - 1]
        action = int(action)
        action = max(0, min(action, self.action_size - 1))
        return action


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
