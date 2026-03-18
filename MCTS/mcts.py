import gymnasium as gym
import numpy as np
from helper.node import Node


class MCTS:
    """Configurable Monte Carlo Tree Search — plug in selection, expansion, rollout,
    backpropagation, and final action selection strategies."""

    def __init__(
        self,
        env: gym.Env,
        num_simulations,
        selection_strategy,
        expansion_strategy,
        rollout_policy,
        backprop_strategy,
        final_action_strategy,
        verbose=False,
    ):
        sim_kwargs = {k: v for k, v in env.unwrapped.spec.kwargs.items() if k != "render_mode"}
        self.sim_env: gym.Env = gym.make(env.unwrapped.spec.id, **sim_kwargs)
        self.num_simulations = num_simulations
        self.strategy = selection_strategy
        self.expand = expansion_strategy
        self.rollout_policy = rollout_policy
        self.backpropagate = backprop_strategy
        self.select_final_action = final_action_strategy
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)

    def search(self, state):
        self.strategy.reset()
        root = Node(state=state, parent=None, action=None, prior=self.expand.prior)
        root.untried_actions = list(range(self.sim_env.action_space.n))

        # PUCT needs parent visits > 0 for sqrt(N_parent) in the formula
        if root.visits == 0:
            root.visits = 1

        for _ in range(self.num_simulations):
            node = self.select(root)
            if node.is_terminal():
                self.backpropagate(node, node.terminal_reward)
            elif not node.is_fully_expanded():
                child_node, step_reward = self.expand(node)
                node.children.append(child_node)
                reward = (
                    child_node.terminal_reward
                    if child_node.is_terminal()
                    else step_reward + self.rollout_policy(child_node)
                )
                self.strategy.update(reward)
                self.backpropagate(child_node, reward)

        best_child = self.select_final_action(root)

        if self.verbose:
            action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
            self.log(f"\n--- Search from state {state} (root visits={root.visits}) ---")
            for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
                q = child.value / child.visits if child.visits > 0 else 0.0
                score = self.strategy.score(child) if child.visits > 0 else float("inf")
                terminal_tag = (
                    f" [TERMINAL reward={child.terminal_reward:.2f}]" if child.is_terminal() else ""
                )
                chosen_tag = " <-- CHOSEN" if child is best_child else ""
                self.log(
                    f"  action={action_names.get(child.action, child.action):<5} "
                    f"-> state={child.state:<4} "
                    f"visits={child.visits:<6} "
                    f"Q={q:+.6f}  "
                    f"score={score:+.6f}"
                    f"{terminal_tag}{chosen_tag}"
                )

        return best_child.action

    def select(self, node: Node) -> Node:
        current = node
        while current.is_fully_expanded() and not current.is_terminal():
            best = self.strategy.best_child(current)
            if best is None:
                break
            current = best
        return current
