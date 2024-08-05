import numpy as np
from collections import defaultdict

class Rewards:
    
    def __init__(self):
        self.rewards = defaultdict(float)
    
    def get(self, state):
        return self.rewards[state]
    
    def set(self, state, reward):
        self.rewards[state] = reward
    