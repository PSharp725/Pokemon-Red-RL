import numpy as np
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        super().__init__(action_space)

    def select_action(self, observation):
        return self.action_space.sample()