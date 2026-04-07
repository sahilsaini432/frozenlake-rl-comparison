import numpy as np
from helper.node import Node


class RobustChild:
    """Select the most visited child (robust child) with random tiebreaking."""

    def __call__(self, root: Node) -> Node:
        max_visits = max((child.visits for child in root.children), default=0)
        best_children = [child for child in root.children if child.visits == max_visits]
        return np.random.choice(best_children)


class MaxValue:
    """Select the child with the highest average value (Q) with random tiebreaking."""

    def __call__(self, root: Node) -> Node:
        def avg_value(child):
            return child.value / child.visits if child.visits > 0 else float("-inf")

        max_q = max(avg_value(child) for child in root.children)
        best_children = [child for child in root.children if avg_value(child) == max_q]
        return np.random.choice(best_children)


class SoftmaxVisits:
    """Sample final action proportionally to softmax over visit counts.

    temperature > 1 → more uniform (more exploration)
    temperature < 1 → more peaked on the most visited child (more greedy)
    temperature → 0 → equivalent to RobustChild
    """

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, root: Node) -> Node:
        children = root.children
        visits = np.array([c.visits for c in children], dtype=float)
        scaled = visits / self.temperature
        scaled -= scaled.max()  # numerical stability
        weights = np.exp(scaled)
        probs = weights / weights.sum()
        return children[np.random.choice(len(children), p=probs)]
