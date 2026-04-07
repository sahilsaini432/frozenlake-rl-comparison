class Node:
    """Represents a single node in the MCTS search tree."""

    def __init__(self, state, parent, action, prior=0.0):
        self.state = state
        self.parent: Node = parent
        self.action = action
        self.prior = prior
        self.children: list[Node] = []
        self.visits = 0  # Number of times this node was visited
        self.value = 0.0  # Total value (reward) accumulated from simulations passing through this node
        self.untried_actions = []
        self.done = False  # Whether this node represents a terminal state
        self.terminal_reward = 0.0  # Reward stored at creation for terminal nodes

    def is_fully_expanded(self):
        # If no untried actions remain, this node is fully expanded
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.done
