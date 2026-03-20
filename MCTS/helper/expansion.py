from ast import List
from copy import deepcopy
import gymnasium as gym
from helper.node import Node


class FullExpansion:
    """Expand all children at once on the first visit to a node.
    On the first call, every untried action is simulated and all child nodes
    are added to node.children simultaneously. One child (the last created) is
    returned for rollout/value evaluation. Subsequent calls won't happen because
    the node is immediately fully expanded.
    """

    def __init__(self, sim_env: gym.Env, prior=0.0):
        self.sim_env = sim_env
        self.prior = prior

    def __call__(self, node: Node) -> List[tuple[Node, float]]:
        actions = list(node.untried_actions)  # copy before clearing
        node.untried_actions.clear()  # mark fully expanded immediately

        children = []
        for action in actions:
            cloned_env = self._clone_env_state(node.state)
            next_state, step_reward, done, _, _ = cloned_env.step(action)

            child_node = Node(state=next_state, parent=node, action=action, prior=self.prior)
            is_wall_hit = next_state == node.state and not done
            child_node.done = done or is_wall_hit
            if is_wall_hit or (done and step_reward == 0):
                child_node.terminal_reward = -1.0
            else:
                child_node.terminal_reward = step_reward if child_node.done else 0.0
            child_node.untried_actions = [] if child_node.done else list(range(self.sim_env.action_space.n))

            node.children.append(child_node)
            children.append((child_node, step_reward))

        return children

    def _clone_env_state(self, state) -> gym.Env:
        cloned_env = deepcopy(self.sim_env)
        cloned_env.reset()
        cloned_env.unwrapped.s = state
        return cloned_env


class StandardExpansion:
    """Standard MCTS expansion — pop an untried action, simulate it, create a child node.
    Treats wall hits and holes as terminal with -1.0 reward."""

    def __init__(self, sim_env: gym.Env, prior=0.0):
        self.sim_env = sim_env
        self.prior = prior

    def __call__(self, node: Node) -> List[tuple[Node, float]]:
        action = node.untried_actions.pop()
        cloned_env = self._clone_env_state(node.state)

        next_state, step_reward, done, truncated, info = cloned_env.step(action)
        child_node = Node(state=next_state, parent=node, action=action, prior=self.prior)

        # Treat wall hits (same state, not done) as terminal to prevent infinite cycles
        is_wall_hit = next_state == node.state and not done
        child_node.done = done or is_wall_hit
        if is_wall_hit or (done and step_reward == 0):
            # Wall hits and holes both get -1.0 so the tree avoids them equally.
            child_node.terminal_reward = -1.0
        else:
            child_node.terminal_reward = step_reward if child_node.done else 0.0

        # Initialize the child node's untried actions (empty if terminal)
        child_node.untried_actions = [] if child_node.done else list(range(self.sim_env.action_space.n))

        node.children.append(child_node)
        return [(child_node, step_reward)]

    def _clone_env_state(self, state) -> gym.Env:
        cloned_env = deepcopy(self.sim_env)
        cloned_env.reset()
        cloned_env.unwrapped.s = state
        return cloned_env
