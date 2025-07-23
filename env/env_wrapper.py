import numpy as np
from pettingzoo.mpe import simple_spread_v3
from gymnasium import spaces

class MultiAgentEnv:
    def __init__(self, num_agents=3, max_cycles=25, continuous_actions=False):
        self.env = simple_spread_v3.env(N=num_agents, max_cycles=max_cycles, continuous_actions=continuous_actions)
        self.env.reset(seed=42)
        self.agents = self.env.agents
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.num_agents = num_agents

    def reset(self):
        self.env.reset(seed=None)
        # debug 
        # # Collect initial observations for all agents
        # for agent in self.agents:
        #     print(f"{agent} action_space: {self.env.action_space(agent)}")

        obs = {agent: self.env.observe(agent) for agent in self.agents}
        return obs

    def step(self, actions):
        for agent in self.agents:
            action = actions[agent]
            # Step the environment for this agent
            self.env.step(action)

        # Collect observations for all agents
        obs = {agent: self.env.observe(agent) for agent in self.agents}
        rewards = self.env.rewards
        terminations = self.env.terminations
        truncations = self.env.truncations
        dones = {agent: terminations[agent] or truncations[agent] for agent in self.agents}
        infos = self.env.infos
        return obs, rewards, dones, infos



    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close()
