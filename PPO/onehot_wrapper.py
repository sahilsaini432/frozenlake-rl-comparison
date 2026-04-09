"""
Converst discrete integer states into one-hot vectors for MLP
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# FrozenLake gives discrete ids, need to convert to one hot vectors 
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_states = env.observation_space.n
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (self.n_states,), dtype = np.float32)
 
    def observation(self, obs):
        one_hot = np.zeros(self.n_states, dtype = np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot