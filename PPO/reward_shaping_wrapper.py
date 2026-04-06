import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, map_size=8, step_penalty=0.0, manhattan_scale=0.0, gamma=0.99):
        super().__init__(env)
        self.map_size = map_size
        self.step_penalty = step_penalty
        self.manhattan_scale = manhattan_scale
        self.gamma = gamma
        self.goal_row = map_size - 1
        self.goal_col = map_size - 1
        self._current_state = 0

    def _get_row_col(self, state):
        return state // self.map_size, state % self.map_size

    def _manhattan_potential(self, state):
        row, col = self._get_row_col(state)
        return -(abs(row - self.goal_row) + abs(col - self.goal_col))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_state = int(np.argmax(obs))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_state = int(np.argmax(obs))

        shaping = 0.0

        if self.step_penalty > 0.0 and not terminated:
            shaping -= self.step_penalty

        if self.manhattan_scale > 0.0:
            phi_s = self._manhattan_potential(self._current_state)
            phi_s_next = self._manhattan_potential(next_state)
            shaping += self.manhattan_scale * (self.gamma * phi_s_next - phi_s)

        self._current_state = next_state
        return obs, reward + shaping, terminated, truncated, info