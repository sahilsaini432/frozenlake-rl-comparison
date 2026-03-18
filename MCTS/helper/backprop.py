from helper.node import Node


class StandardBackprop:
    """Standard backpropagation — propagate reward up the tree, incrementing visits."""

    def __call__(self, node: Node, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
