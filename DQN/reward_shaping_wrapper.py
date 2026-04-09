from __future__ import annotations

import gymnasium as gym
import numpy as np


class FrozenLakeRewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        step_penalty: float = -0.01,
        closer_bonus: float = 0.05,
        farther_penalty: float = -0.05,
        goal_reward: float = 1.0,
        hole_penalty: float = -1.0,
    ):
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.Discrete):
            raise TypeError(
                "FrozenLakeRewardShapingWrapper expects Discrete obs (wrap FrozenLake before OneHot)."
            )
        desc = np.asarray(self.env.unwrapped.desc)
        self._desc = desc
        self._nrow, self._ncol = desc.shape
        self._goal_rc = self._find_cell(desc, b"G")
        self.step_penalty = float(step_penalty)
        self.closer_bonus = float(closer_bonus)
        self.farther_penalty = float(farther_penalty)
        self.goal_reward = float(goal_reward)
        self.hole_penalty = float(hole_penalty)
        self._last_state: int | None = None

    @staticmethod
    def _find_cell(desc: np.ndarray, mark: bytes) -> tuple[int, int]:
        for r in range(desc.shape[0]):
            for c in range(desc.shape[1]):
                if desc[r, c] == mark:
                    return r, c
        raise ValueError(f"No cell {mark!r} found in FrozenLake desc")

    def _state_to_rc(self, state: int) -> tuple[int, int]:
        s = int(state)
        return s // self._ncol, s % self._ncol

    def _manhattan_to_goal(self, state: int) -> int:
        r, c = self._state_to_rc(state)
        gr, gc = self._goal_rc
        return abs(r - gr) + abs(c - gc)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_state = int(np.asarray(obs).item())
        return obs, info

    def step(self, action):
        if self._last_state is None:
            raise RuntimeError("FrozenLakeRewardShapingWrapper: reset() must be called before step().")
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        s_old = self._last_state
        s_new = int(np.asarray(obs).item())
        self._last_state = s_new

        d_old = self._manhattan_to_goal(s_old)
        d_new = self._manhattan_to_goal(s_new)

        shaped = self.step_penalty
        if d_new < d_old:
            shaped += self.closer_bonus
        elif d_new > d_old:
            shaped += self.farther_penalty

        r, c = self._state_to_rc(s_new)
        ch = self._desc[r, c]
        in_goal = ch == b"G" or (isinstance(ch, str) and ch == "G")
        in_hole = ch == b"H" or (isinstance(ch, str) and ch == "H")

        done = terminated or truncated
        if terminated:
            if in_goal or _base_reward >= 1.0:
                shaped += self.goal_reward
            elif in_hole:
                shaped += self.hole_penalty

        info = dict(info)
        if done:
            on_goal = bool(in_goal or (terminated and _base_reward >= 1.0))
            info["is_goal"] = on_goal
            info["is_hole"] = bool(terminated and in_hole)

        return obs, shaped, terminated, truncated, info
