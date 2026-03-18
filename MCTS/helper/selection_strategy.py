import math
import numpy as np


class UCTStrategy:
    """UCT with reward normalization to [0, 1] based on observed min/max rewards in the tree."""

    def __init__(self, exploration_constant):
        self.exploration_constant = exploration_constant
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def reset(self):
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def update(self, reward):
        self.min_value = min(self.min_value, reward)
        self.max_value = max(self.max_value, reward)

    def score(self, node) -> float:
        if node.visits == 0:
            return float("inf")
        q = node.value / node.visits
        value_range = self.max_value - self.min_value
        normalized_q = (q - self.min_value) / value_range if value_range > 1e-8 else 0.5
        return normalized_q + self.exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)

    def best_child(self, node):
        scores = [self.score(child) for child in node.children]
        max_score = max(scores)
        best_children = [child for child, s in zip(node.children, scores) if s == max_score]
        return np.random.choice(best_children)


class UCB1Strategy:
    """UCB1 — Q(v) + C * sqrt(ln(N_parent) / N_v). No reward normalization."""

    def __init__(self, exploration_constant):
        self.exploration_constant = exploration_constant

    def reset(self):
        pass

    def update(self, _):
        pass

    def score(self, node) -> float:
        if node.visits == 0:
            return float("inf")
        q = node.value / node.visits
        return q + self.exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)

    def best_child(self, node):
        scores = [self.score(child) for child in node.children]
        max_score = max(scores)
        best_children = [child for child, s in zip(node.children, scores) if s == max_score]
        return np.random.choice(best_children)


class PUCTStrategy_Heuristic:
    """PUCT with heuristic priors based on Manhattan distance to the goal.
    Actions that reduce distance to the goal get higher prior probability."""

    def __init__(self, exploration_constant, grid_size):
        self.exploration_constant = exploration_constant
        self.grid_size = grid_size
        self.goal_row = grid_size - 1
        self.goal_col = grid_size - 1

    def reset(self):
        pass

    def update(self, _):
        pass

    def compute_prior(self, state, action):
        """Compute a heuristic prior for an action based on how much it reduces
        Manhattan distance to the goal. Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP."""
        row, col = divmod(state, self.grid_size)
        dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_row = max(0, min(self.grid_size - 1, row + dr))
        new_col = max(0, min(self.grid_size - 1, col + dc))
        current_dist = abs(row - self.goal_row) + abs(col - self.goal_col)
        new_dist = abs(new_row - self.goal_row) + abs(new_col - self.goal_col)
        # Reward actions that decrease distance, penalize those that increase it
        return current_dist - new_dist + 1  # shift so all values >= 0

    def _ensure_priors(self, node):
        """Lazily assign heuristic priors to all children once the node is fully expanded."""
        if not hasattr(node, '_heuristic_priors_set'):
            raw = [self.compute_prior(node.state, child.action) for child in node.children]
            total = sum(raw)
            for child, r in zip(node.children, raw):
                child.prior = r / total if total > 0 else 1.0 / len(node.children)
            node._heuristic_priors_set = True

    def score(self, node) -> float:
        q = node.value / node.visits if node.visits > 0 else 0.0
        return q + self.exploration_constant * node.prior * math.sqrt(node.parent.visits) / (1 + node.visits)

    def best_child(self, node):
        self._ensure_priors(node)
        scores = [self.score(child) for child in node.children]
        max_score = max(scores)
        best_children = [child for child, s in zip(node.children, scores) if s == max_score]
        return np.random.choice(best_children)


class PUCTStrategy_Uniform:
    """PUCT — Q(v) + C * P(a) * sqrt(N_parent) / (1 + N_v). Used in AlphaGo/AlphaZero.
    Requires nodes to have a `prior` attribute."""

    def __init__(self, exploration_constant):
        self.exploration_constant = exploration_constant

    def reset(self):
        pass

    def update(self, _):
        pass

    def score(self, node) -> float:
        q = node.value / node.visits if node.visits > 0 else 0.0
        return q + self.exploration_constant * node.prior * math.sqrt(node.parent.visits) / (1 + node.visits)

    def best_child(self, node):
        scores = [self.score(child) for child in node.children]
        max_score = max(scores)
        best_children = [child for child, s in zip(node.children, scores) if s == max_score]
        return np.random.choice(best_children)
