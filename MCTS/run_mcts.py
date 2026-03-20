# Plug-and-play MCTS runner — select different strategies via CLI flags

import argparse
from calendar import EPOCH
import json
from os import path
from time import perf_counter
import gymnasium as gym
from tqdm import tqdm

from mcts import MCTS
from helper.selection_strategy import (
    UCTStrategy,
    UCB1Strategy,
    PUCTStrategy_Heuristic,
    PUCTStrategy_Softmax,
    PUCTStrategy_Uniform,
)
from helper.rollout import RandomRollout, EpsilonGreedyRollout, ValueNetworkRollout
from helper.value_function import ValueMLP, ValueFunctionOnly, HeuristicValueFunction
from helper.expansion import FullExpansion, StandardExpansion, ProgressiveWideningExpansion
from helper.backprop import StandardBackprop, MaxBackprop
from helper.final_action import RobustChild, MaxValue
from metrics.plot import plot_progress, plot_time_stats

GAMMA = 0.99  # Discount factor for value iteration and rollout blending
EPOCHS = 1000  # Training epochs for value network
LEARNING_RATE = 1e-3  # Learning rate for value network training
LAMBDA = 0.5  # Blending factor for value network in rollout (0 = pure rollout, 1 = pure value fn)
ALPHA = 0.5  # Alpha for progressive widening (not used in this code but can be set when building the expansion strategy)

# Load generated maps from maps.json (run helper/map_generator.py to regenerate)
_maps_file = path.join(path.dirname(__file__), "maps.json")
with open(_maps_file) as _f:
    _generated_maps = json.load(_f)

MAPS = {
    4: _generated_maps["4"],
    8: _generated_maps["8"],
    16: _generated_maps["16"],
    32: _generated_maps["32"],
    64: _generated_maps["64"],
}

verbose = False

# --- Strategy factory maps ---

SELECTION_CHOICES = ["uct", "ucb1", "puct_uniform", "puct_heuristic", "puct_softmax"]
ROLLOUT_CHOICES = ["random", "epsilon_greedy", "value_network", "mlp_value_network", "alphazero"]
FINAL_ACTION_CHOICES = ["robust_child", "max_value"]
EXPANSION_CHOICES = ["standard", "full", "progressive_widening"]
BACKPROP_CHOICES = ["standard", "max"]


def build_expansion(name, sim_env, prior):
    if name == "standard":
        return StandardExpansion(sim_env, prior=prior)
    if name == "full":
        return FullExpansion(sim_env, prior=prior)
    if name == "progressive_widening":
        return ProgressiveWideningExpansion(sim_env, prior=prior, alpha=ALPHA)
    raise ValueError(f"Unknown expansion strategy: {name}")


def build_selection(name, exploration_constant, grid_size):
    if name == "uct":
        return UCTStrategy(exploration_constant)
    if name == "ucb1":
        return UCB1Strategy(exploration_constant)
    if name == "puct_uniform":
        return PUCTStrategy_Uniform(exploration_constant)
    if name == "puct_heuristic":
        return PUCTStrategy_Heuristic(exploration_constant, grid_size)
    if name == "puct_softmax":
        return PUCTStrategy_Softmax(exploration_constant, grid_size)
    raise ValueError(f"Unknown selection strategy: {name}")


def build_rollout(name, sim_env, depth, env, grid_size):
    if name == "random":
        return RandomRollout(sim_env, depth)
    if name == "epsilon_greedy":
        return EpsilonGreedyRollout(sim_env, depth, grid_size, epsilon=0.1)
    if name == "value_network":
        base_rollout = RandomRollout(sim_env, depth)
        value_fn = HeuristicValueFunction(grid_size)
        return ValueNetworkRollout(value_fn, base_rollout, lam=LAMBDA)
    if name == "mlp_value_network":
        print(f"Running value iteration + training MLP for {grid_size}x{grid_size} grid...")
        base_rollout = RandomRollout(sim_env, depth)
        mlp = ValueMLP(hidden_size=64)
        value_fn, _ = mlp.train_value_network(
            env, grid_size, gamma=GAMMA, epochs=EPOCHS, lr=LEARNING_RATE, verbose=True
        )
        return ValueNetworkRollout(value_fn, base_rollout, lam=LAMBDA)
    if name == "alphazero":
        print(
            f"Running value iteration + training MLP (AlphaZero-style, no rollout) for {grid_size}x{grid_size} grid..."
        )
        mlp = ValueMLP(hidden_size=64)
        value_fn, _ = mlp.train_value_network(
            env, grid_size, gamma=GAMMA, epochs=EPOCHS, lr=LEARNING_RATE, verbose=True
        )
        return ValueFunctionOnly(value_fn)
    raise ValueError(f"Unknown rollout policy: {name}")


def build_final_action(name):
    if name == "robust_child":
        return RobustChild()
    if name == "max_value":
        return MaxValue()
    raise ValueError(f"Unknown final action strategy: {name}")


def build_agent(env, args):
    sim_count = {4: 1000, 8: 3000, 16: 8000, 32: 15000, 64: 30000}[args.grid]
    rollout_depth = {4: 100, 8: 200, 16: 400, 32: 800, 64: 1600}[args.grid]

    # Build a headless sim env for components that need it
    sim_kwargs = {k: v for k, v in env.unwrapped.spec.kwargs.items() if k != "render_mode"}
    sim_env = gym.make(env.unwrapped.spec.id, **sim_kwargs)

    selection = build_selection(args.selection, args.exploration_constant, args.grid)

    # PUCT needs a prior on each node; uniform = 1/num_actions; softmax sets priors lazily
    prior = (1.0 / sim_env.action_space.n) if args.selection in ("puct_uniform", "puct_softmax") else 0.0
    expansion = build_expansion(args.expansion, sim_env, prior)

    rollout = build_rollout(args.rollout, sim_env, rollout_depth, env, args.grid)
    backprop = MaxBackprop() if args.backprop == "max" else StandardBackprop()
    final_action = build_final_action(args.final_action)

    return MCTS(
        env=env,
        num_simulations=sim_count,
        selection_strategy=selection,
        expansion_strategy=expansion,
        rollout_policy=rollout,
        backprop_strategy=backprop,
        final_action_strategy=final_action,
        verbose=args.verbose,
    )


def log(message):
    if verbose:
        print(f"{message}")


def evaluate_mcts(env, agent, episodes, args):
    """Run MCTS for a fixed number of episodes and report success rate and avg reward."""
    total_rewards = []
    episode_times = []
    steps_per_episode = []
    avg_search_times = []
    successes = 0

    for episode in tqdm(range(episodes), desc="Evaluating MCTS"):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        episode_search_time = 0.0

        log(f"--- Starting episode {episode + 1} ---")
        episode_start = perf_counter()
        while not done:
            search_start = perf_counter()
            action = agent.search(obs)
            episode_search_time += perf_counter() - search_start

            log(f"Taking action: {action} at state: {obs}")
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            if terminated:
                log(f"Episode ended with reward: {reward} (terminated)")
            done = terminated or truncated
        episode_times.append(perf_counter() - episode_start)

        total_rewards.append(episode_reward)
        steps_per_episode.append(steps)
        avg_search_times.append(episode_search_time / steps if steps > 0 else 0.0)
        if episode_reward > 0:
            successes += 1

    print(f"\n=== MCTS Evaluation over {episodes} episodes ===")
    print(f"Selection:          {args.selection}")
    print(f"Rollout:            {args.rollout}")
    print(f"Final action:       {args.final_action}")
    print(f"Exploration C:      {args.exploration_constant}")
    print(f"Success rate:       {successes / episodes * 100:.1f}%")
    print(f"Avg reward:         {sum(total_rewards) / episodes:.4f}")
    print(f"Min/Max reward:     {min(total_rewards):.4f} / {max(total_rewards):.4f}")
    print(f"Avg episode time:   {sum(episode_times) / episodes:.2f}s")
    print(f"Avg steps/episode:  {sum(steps_per_episode) / episodes:.1f}")
    print(f"Avg search time:    {sum(avg_search_times) / episodes * 1000:.1f}ms/step")
    plot_progress(total_rewards, args)
    plot_time_stats(episode_times, steps_per_episode, avg_search_times, args)
    return total_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plug-and-play MCTS for FrozenLake")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of evaluation episodes")

    # Strategy selection
    parser.add_argument(
        "--selection",
        choices=SELECTION_CHOICES,
        default="uct",
        help="Selection strategy (default: uct)",
    )
    parser.add_argument(
        "--rollout",
        choices=ROLLOUT_CHOICES,
        default="random",
        help="Rollout policy (default: random)",
    )
    parser.add_argument(
        "--final_action",
        choices=FINAL_ACTION_CHOICES,
        default="robust_child",
        help="Final action selection (default: robust_child)",
    )
    parser.add_argument(
        "--backprop",
        choices=BACKPROP_CHOICES,
        default="standard",
        help="Backpropagation strategy (default: standard)",
    )
    parser.add_argument(
        "-c", "--exploration_constant", type=float, default=1.4, help="Exploration constant C (default: 1.4)"
    )

    # Environment
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--human", action="store_true", default=False, help="Enable human render")
    parser.add_argument(
        "-g", "--grid", type=int, choices=[4, 8, 16, 32, 64], default=4, help="Grid size for FrozenLake"
    )
    parser.add_argument(
        "-s", "--slip", action="store_true", default=False, help="Use slippery version of FrozenLake"
    )
    args = parser.parse_args()
    verbose = args.verbose

    selected_map = MAPS[args.grid]

    # Set up the environment
    if args.human:
        env = gym.make("FrozenLake-v1", desc=selected_map, is_slippery=args.slip, render_mode="human")
    else:
        env = gym.make("FrozenLake-v1", desc=selected_map, is_slippery=args.slip)

    # Build and evaluate
    agent = build_agent(env, args)
    evaluate_mcts(env, agent, args.episodes, args)
