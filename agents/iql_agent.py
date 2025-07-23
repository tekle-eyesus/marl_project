from agents.base_agent import BaseAgent
import random
import numpy as np
import torch

class IQLAgent(BaseAgent):
    def __init__(self, obs_space, action_space, lr=1e-3, gamma=0.95):
        super().__init__(obs_space, action_space, lr, gamma)
        self.replay_buffer = []

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        minibatch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Compute Q targets
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute Q predictions
        q_values = self.q_network(states).gather(1, actions)

        # Loss and optimize
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
