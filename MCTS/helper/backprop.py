from helper.node import Node


class StandardBackprop:
    """Standard backpropagation — propagate reward up the tree, incrementing visits."""

    def __call__(self, node: Node, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class MaxBackprop:
    """Max backup — propagate the best reward seen, not the cumulative sum."""

    def __call__(self, node: Node, reward: float):
        while node is not None:
            node.visits += 1
            # Recover running max from previous value/visits, then update.
            # This keeps node.value / node.visits == max(all rewards seen),
            # so selection strategies that compute Q = value/visits stay correct.
            prev_max = node.value / (node.visits - 1) if node.visits > 1 else float("-inf")
            node.value = max(prev_max, reward) * node.visits
            node = node.parent
