import gymnasium as gym
import numpy as np
from copy import deepcopy
from MCTS.helper.selection_strategy import UCTStrategy
from MCTS.helper.node import Node

# Rollout policy: Epsilon-greedy (greedy = minimize Manhattan distance to goal with hole penalty via one-step lookahead)
# Final Action selection: Most visited child (robust child)
# Selection policy: UCT with reward normalization to [0, 1] based on observed min/max rewards in the tree (to handle different reward scales across environments)


class MCTS_GreedyRollout:
    """Monte Carlo Tree Search algorithm"""

    def __init__(
        self,
        env: gym.Env,
        num_simulations,
        exploration_constant,
        max_rollout_depth,
        epsilon=0.5,
        verbose=False,
    ):
        # Headless sim env for cloning — strip render_mode so simulations don't render
        sim_kwargs = {k: v for k, v in env.unwrapped.spec.kwargs.items() if k != "render_mode"}
        self.sim_env: gym.Env = gym.make(env.unwrapped.spec.id, **sim_kwargs)
        self.num_simulations = num_simulations
        self.max_rollout_depth = max_rollout_depth
        self.epsilon = epsilon  # Probability of random action in rollout
        self.strategy = UCTStrategy(exploration_constant)
        self.grid_size = env.unwrapped.nrow
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)

    # Decides the best action to take from the given state by running MCTS simulations
    def search(self, state):
        self.strategy.reset()
        # Create the root node for the current state
        root = Node(state=state, parent=None, action=None)
        # No actions have been tried from the root yet, so initialize the untried actions to all possible actions
        root.untried_actions = list(range(self.sim_env.action_space.n))

        for _ in range(self.num_simulations):
            # Selection: Start from the root and select child nodes until we reach a node that is not fully expanded or is terminal
            node = self.select(root)
            if node.is_terminal():
                self.backpropagate(node, node.terminal_reward)
            elif not node.is_fully_expanded():
                child_node, step_reward = self.expand(node)
                node.children.append(child_node)
                # For terminal children (goal/hole), use the step reward directly —
                # rollout from a terminal state won't reproduce the reward that was
                # given for arriving there (e.g. goal gives reward=1 on arrival only)
                reward = (
                    child_node.terminal_reward
                    if child_node.is_terminal()
                    else step_reward + self.rollout(child_node)
                )
                self.strategy.update(reward)
                self.backpropagate(child_node, reward)

        # Get best child from the root — random tiebreaking when visits are equal
        max_visits = max((child.visits for child in root.children), default=0)
        best_children = [child for child in root.children if child.visits == max_visits]
        best_child = np.random.choice(best_children)

        if self.verbose:
            action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
            self.log(f"\n--- Search from state {state} (root visits={root.visits}) ---")
            for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
                q = child.value / child.visits if child.visits > 0 else 0.0
                ucb1 = self.strategy.score(child) if child.visits > 0 else float("inf")
                terminal_tag = (
                    f" [TERMINAL reward={child.terminal_reward:.2f}]" if child.is_terminal() else ""
                )
                chosen_tag = " <-- CHOSEN" if child is best_child else ""
                self.log(
                    f"  action={action_names.get(child.action, child.action):<5} "
                    f"-> state={child.state:<4} "
                    f"visits={child.visits:<6} "
                    f"Q={q:+.6f}  "
                    f"UCT={ucb1:+.6f}"
                    f"{terminal_tag}{chosen_tag}"
                )

        # choose the action of the most visited child
        return best_child.action

    def select(self, node: Node) -> Node:
        current_Node = node
        # Find a leaf node to expand: keep selecting the best child until we find a node that is not fully expanded or is terminal
        while current_Node.is_fully_expanded() and not current_Node.is_terminal():
            best_node = self.strategy.best_child(current_Node)

            if best_node is None:
                break  # No children, return the current leaf node
            current_Node = best_node
        return current_Node

    def expand(self, node: Node) -> tuple["Node", float]:
        # Pick an untried action
        action = node.untried_actions.pop()
        cloned_env = self.clone_env_state(node.state)

        # Simulate the action in the cloned environment
        next_state, step_reward, done, truncated, info = cloned_env.step(action)
        if step_reward == 1 and done:
            self.log("Hit a goal during expansion!")
        elif done:
            self.log(f"Hit a hole during expansion! State: {next_state}")
        child_node = Node(state=next_state, parent=node, action=action)

        # Treat wall hits (same state, not done) as terminal to prevent infinite cycles
        is_wall_hit = next_state == node.state and not done
        child_node.done = done or is_wall_hit
        if is_wall_hit or (done and step_reward == 0):
            # Wall hits and holes both get -1.0 so the tree avoids them equally.
            # Without this, holes (reward=0) look better than valid paths (Q slightly
            # negative from wall hit penalties leaking up the tree).
            child_node.terminal_reward = -1.0
        else:
            child_node.terminal_reward = step_reward if child_node.done else 0.0

        # Initialize the child node's untried actions (empty if terminal)
        child_node.untried_actions = [] if child_node.done else list(range(self.sim_env.action_space.n))

        return child_node, step_reward

    def rollout(self, node: Node) -> float:
        cloned_env = self.clone_env_state(node.state)
        # Epsilon-greedy: with prob epsilon take random action, else take greedy (best immediate reward)
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
        # One-step lookahead using the transition model — no env cloning needed
        # P[state][action] = list of (prob, next_state, reward, done)
        # Randomly break ties so rollouts stay exploratory when all rewards are equal (e.g. reward=0)

        # Compute current and goal (x, y) positions on the grid
        row, col = divmod(state, self.grid_size)
        goal = self.grid_size * self.grid_size - 1
        goal_row, goal_col = divmod(goal, self.grid_size)

        scores = []
        # For each action calculate score
        for action in range(self.sim_env.action_space.n):
            expected_score = 0
            # Average over all stochastic transitions for this action
            for prob, next_state, reward, done in self.sim_env.unwrapped.P[state][action]:
                next_row, next_col = divmod(next_state, self.grid_size)

                # Negative distance so higher score = closer to goal
                dist = abs(next_row - goal_row) + abs(next_col - goal_col)

                # Heavy penalty for landing in a hole (done=True, reward=0)
                hole_penalty = -10 if (done and reward == 0) else 0

                expected_score += prob * (-dist + hole_penalty)
            scores.append(expected_score)

        # Pick uniformly among all actions that tie for the best score
        max_score = max(scores)
        best_actions = [a for a, s in enumerate(scores) if s == max_score]
        return np.random.choice(best_actions)

    def backpropagate(self, node: Node, reward: float):
        # Propagate the reward up the tree
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_action_probabilities(self, root: Node):
        action_probabilities = np.zeros(self.sim_env.action_space.n)
        total_visits = sum(child.visits for child in root.children)
        if total_visits == 0:
            return action_probabilities

        for child in root.children:
            action_probabilities[child.action] = child.visits / total_visits
        return action_probabilities

    def clone_env_state(self, state) -> gym.Env:
        cloned_env: gym.Env = deepcopy(self.sim_env)
        cloned_env.reset()
        # overwrites the state to the specific one you want
        cloned_env.unwrapped.s = state
        return cloned_env
