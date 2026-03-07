import gymnasium as gym
import numpy as np
import math
from copy import deepcopy


class MCTSNode:
    """Represents a single node in the MCTS search tree."""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        self.done = False  # Whether this node represents a terminal state

    def is_fully_expanded(self):
        # If no untried actions remain, this node is fully expanded
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.done

    def ucb1(self, exploration_constant=math.sqrt(2)):
        """
        Calculate the UCB1 value for this node.

        The UCB1 formula balances exploitation (average value) with
        exploration (bonus for less-visited nodes):
            UCB1 = (value / visits) + c * sqrt(ln(parent.visits) / visits)

        Args:
            exploration_constant: The exploration parameter c (default sqrt(2)).

        Returns:
            The UCB1 score for this node.
        """
        pass

    def best_child(self, exploration_constant=math.sqrt(2)):
        """
        Select the child node with the highest UCB1 value.

        Args:
            exploration_constant: The exploration parameter c passed to UCB1.

        Returns:
            The child MCTSNode with the highest UCB1 score.
        """
        pass


class MCTS:
    """Monte Carlo Tree Search algorithm for OpenAI Gymnasium environments."""

    def __init__(self, env, num_simulations=1000, exploration_constant=math.sqrt(2), max_rollout_depth=100):
        """
        Initialize the MCTS algorithm.

        Args:
            env: An OpenAI Gymnasium environment instance.
            num_simulations: Number of MCTS iterations to run per action selection.
            exploration_constant: The exploration parameter c for UCB1.
            max_rollout_depth: Maximum number of steps in a rollout before stopping.
        """
        self.env = env
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_rollout_depth = max_rollout_depth

    def search(self, state):
        """
        Run the full MCTS algorithm from the given state and return the best action.

        This is the main entry point. It creates the root node, runs
        num_simulations iterations of select -> expand -> rollout -> backpropagate,
        then returns the action leading to the most-visited child.

        Args:
            state: The current environment state to search from.

        Returns:
            The best action to take from the current state.
        """
        pass

    def select(self, node):
        """
        Selection phase: traverse the tree from the root to a leaf node.

        Starting from the given node, repeatedly select the best child
        (via UCB1) until reaching a node that is not fully expanded or
        is terminal.

        Args:
            node: The root MCTSNode to start selection from.

        Returns:
            The selected MCTSNode (a leaf or not-fully-expanded node).
        """
        pass

    def expand(self, node):
        """
        Expansion phase: add a new child node to the tree.

        Pick one untried action from the node, simulate it in the environment
        to get the resulting state, and create a new child MCTSNode.

        Args:
            node: The MCTSNode to expand.

        Returns:
            The newly created child MCTSNode.
        """
        pass

    def rollout(self, node):
        """
        Simulation/rollout phase: play out a random episode from the node's state.

        From the given node's state, repeatedly take random actions until
        reaching a terminal state or the max_rollout_depth. Use the
        environment's step function to simulate.

        Args:
            node: The MCTSNode to start the rollout from.

        Returns:
            The cumulative reward obtained during the rollout.
        """
        pass

    def backpropagate(self, node, reward):
        """
        Backpropagation phase: propagate the rollout result up the tree.

        Starting from the given node, walk up to the root, incrementing
        the visit count and adding the reward to the value of each node
        along the path.

        Args:
            node: The MCTSNode where the rollout ended (leaf node).
            reward: The reward obtained from the rollout.
        """
        pass

    def get_action_probabilities(self, root):
        """
        Compute action probabilities based on visit counts of the root's children.

        After all simulations, calculate the probability of each action
        proportional to how many times each child was visited. This can
        be used for training or for stochastic action selection.

        Args:
            root: The root MCTSNode after search is complete.

        Returns:
            A numpy array of action probabilities over the action space.
        """
        pass

    def clone_env_state(self, state):
        """
        Create a deep copy of the environment at the given state.

        MCTS requires simulating actions without modifying the real environment.
        This method should clone/restore the environment so that rollouts
        and expansions don't affect the true game state.

        Args:
            state: The environment state to clone.

        Returns:
            A deep copy of the environment set to the given state.
        """
        pass
