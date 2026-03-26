import numpy as np
from copy import deepcopy


class RandomRollout:
    """Random rollout policy — samples actions uniformly at random."""

    def __init__(self, sim_env, max_rollout_depth):
        self.sim_env = sim_env
        self.max_rollout_depth = max_rollout_depth

    def __call__(self, node) -> float:
        cloned_env = self._clone_env_state(node.state)
        cumulative_reward = 0.0
        for _ in range(self.max_rollout_depth):
            action = cloned_env.action_space.sample()
            _, reward, done, _, _ = cloned_env.step(action)
            cumulative_reward += reward
            if done:
                break
        return cumulative_reward

    def _clone_env_state(self, state):
        cloned_env = deepcopy(self.sim_env)
        cloned_env.reset()
        cloned_env.unwrapped.s = state
        return cloned_env


class ValueNetworkRollout:
    """Blends a value function V(s) with a rollout estimate."""

    def __init__(self, value_fn, rollout_policy, lam=0.5):
        self.value_fn = value_fn
        self.rollout_policy = rollout_policy
        self.lam = lam

    def __call__(self, node) -> float:
        v = self.value_fn(node.state)
        r = self.rollout_policy(node)
        return self.lam * r + (1.0 - self.lam) * v


class EpsilonGreedyRollout:
    """Epsilon-greedy rollout — with probability epsilon take a random action,
    otherwise take greedy action (minimize Manhattan distance to goal with hole penalty)."""

    def __init__(self, sim_env, max_rollout_depth, grid_size, epsilon=0.5):
        self.sim_env = sim_env
        self.max_rollout_depth = max_rollout_depth
        self.grid_size = grid_size
        self.epsilon = epsilon

    def __call__(self, node) -> float:
        cloned_env = self._clone_env_state(node.state)
        cumulative_reward = 0.0
        for _ in range(self.max_rollout_depth):
            if np.random.random() < self.epsilon:
                action = cloned_env.action_space.sample()
            else:
                action = self._greedy_action(cloned_env.unwrapped.s)
            _, reward, done, _, _ = cloned_env.step(action)
            cumulative_reward += reward
            if done:
                break
        return cumulative_reward

    def _greedy_action(self, state) -> int:
        goal = self.grid_size * self.grid_size - 1
        goal_row, goal_col = divmod(goal, self.grid_size)

        scores = []
        for action in range(self.sim_env.action_space.n):
            expected_score = 0
            for prob, next_state, reward, done in self.sim_env.unwrapped.P[state][action]:
                next_row, next_col = divmod(next_state, self.grid_size)
                dist = abs(next_row - goal_row) + abs(next_col - goal_col)
                hole_penalty = -10 if (done and reward == 0) else 0
                expected_score += prob * (-dist + hole_penalty)
            scores.append(expected_score)

        max_score = max(scores)
        best_actions = [a for a, s in enumerate(scores) if s == max_score]
        return np.random.choice(best_actions)

    def _clone_env_state(self, state):
        cloned_env = deepcopy(self.sim_env)
        cloned_env.reset()
        cloned_env.unwrapped.s = state
        return cloned_env
