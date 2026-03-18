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
