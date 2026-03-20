from helper.node import Node


class StandardBackprop:
    """Standard backpropagation — propagate reward up the tree, incrementing visits."""

    def __call__(self, node: Node, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class MaxBackprop:
    """Max backup — propagate the best reward seen, not the cumulative sum.

    Stores the running max in node.value and sets node.visits so that
    value / visits == max_reward, keeping Q-value computation in selection
    strategies correct (they all compute node.value / node.visits).

    Better than averaging when reward is sparse and a single success matters
    more than the average outcome (e.g. FrozenLake with rare goal reaches).
    """

    def __call__(self, node: Node, reward: float):
        while node is not None:
            node.visits += 1
            # Recover running max from previous value/visits, then update.
            # This keeps node.value / node.visits == max(all rewards seen),
            # so selection strategies that compute Q = value/visits stay correct.
            prev_max = node.value / (node.visits - 1) if node.visits > 1 else float("-inf")
            node.value = max(prev_max, reward) * node.visits
            node = node.parent
