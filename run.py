# Update this file to incorporate any new agent or implementation you create

import argparse
import os
from multiprocessing import Pool
from time import sleep, perf_counter
import gymnasium as gym
from stable_baselines3 import A2C
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from A2C.base_A2C import ModA2C
from MCTS.mcts_greedy_rollout import MCTS_GreedyRollout
from MCTS.mcts_ucb1 import MCTS_UCB1
from MCTS.mcts_base import MCTSBase
from metrics.plot import plot_progress, plot_time_stats

four_x_four_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

eight_x_eight_map = [
    "SFHFFFHF",
    "FFFHFFFH",
    "HFFFHFHF",
    "FFHFFHFF",
    "FHFFFFHF",
    "FFFHFFFH",
    "HFHFFHFF",
    "FFFHFFHG",
]

sixteen_x_sixteen_map = [
    "SFFHFFFHFFFHFFFF",
    "FHFFFHFFHFFHFFHF",
    "FFHFFFFFHFFHFFFF",
    "HFFFHFFHFFFHFHFF",
    "FFFHFFFHFHFFHFFF",
    "FHFFHFFFHFFFHFHF",
    "FFFHFFFFHFHFFHFF",
    "HFFFHFHFFFHFFFHF",
    "FFHFFFHFHFFFHFFF",
    "FHFFHFFFHFHFFHFF",
    "FFFHFFFHFFHFHFFF",
    "HFHFFFHFHFFFHFFH",
    "FFFHFHFFFHFFHFFF",
    "FHFFFHFFHFHFFFHF",
    "FFHFHFFFHFFFHFFF",
    "HFFFHFHFFFHFFHFG",
]

verbose = False


def selected_agent(args):
    # Scale simulations and rollout depth with grid size so the tree has enough signal
    sim_count = {4: 1000, 8: 3000, 16: 8000}[args.grid]
    rollout_depth = {4: 100, 8: 200, 16: 400}[args.grid]

    if args.mcts_base:
        # Using the specs from the project proposal for MCTS
        return MCTSBase(
            env=env,
            num_simulations=sim_count,
            exploration_constant=1.4,
            max_rollout_depth=rollout_depth,
            verbose=args.verbose,
        )
    elif args.mcts_ucb1:
        return MCTS_UCB1(
            env=env,
            num_simulations=sim_count,
            exploration_constant=1.4,
            max_rollout_depth=rollout_depth,
            verbose=args.verbose,
        )
    elif args.mcts_greedy:
        return MCTS_GreedyRollout(
            env=env,
            num_simulations=sim_count,
            exploration_constant=1.4,
            max_rollout_depth=rollout_depth,
            epsilon=0.1,  # 10% random actions in rollout
            verbose=args.verbose,
        )

    return ModA2C("MlpPolicy", env)


def log(message):
    if verbose:
        print(f"{message}")


def evaluate_mcts(env, agent, episodes, verbose=False):
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
    print(f"Success rate:       {successes / episodes * 100:.1f}%")
    print(f"Avg reward:         {sum(total_rewards) / episodes:.4f}")
    print(f"Min/Max reward:     {min(total_rewards):.4f} / {max(total_rewards):.4f}")
    print(f"Avg episode time:   {sum(episode_times) / episodes:.2f}s")
    print(f"Avg steps/episode:  {sum(steps_per_episode) / episodes:.1f}")
    print(f"Avg search time:    {sum(avg_search_times) / episodes * 1000:.1f}ms/step")
    alg_name = type(agent).__name__
    plot_progress(total_rewards, alg_name)
    plot_time_stats(episode_times, steps_per_episode, avg_search_times, alg_name)
    return total_rewards


def train_agent(env, agent, episodes):
    if isinstance(agent, (MCTSBase, MCTS_UCB1, MCTS_GreedyRollout)):
        evaluate_mcts(env, agent, episodes, verbose=verbose)
    else:
        # Train the agent
        num_episodes = args.episodes
        obs, info = env.reset()

        for episode in tqdm(range(num_episodes), desc="Training"):
            done = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.learn(obs, action, reward, next_obs, terminated)
                obs = next_obs
                done = terminated or truncated
                sleep(3)
            obs, info = env.reset()


def test_agent(env, agent):
    # Test the trained agent
    obs, info = env.reset()
    for step in tqdm(range(1000)):
        action = agent.select_action(obs)
        print(f"\n--- Step {step + 1} ---")
        print(f"Current observation: {obs}")
        print(f"Action taken: {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Next observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            obs, info = env.reset()
            print(f"Environment reset. New observation: {obs}")


def canTest(args):
    if args.mcts_base or args.mcts_ucb1 or args.mcts_greedy:
        return False
    else:
        return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--mcts_base", action="store_true", help="Use MCTS as the base agent")
    parser.add_argument("--mcts_ucb1", action="store_true", help="Use MCTS with UCB1 for selection")
    parser.add_argument("--mcts_greedy", action="store_true", help="Use MCTS with greedy rollout policy")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--human", default=False, action="store_true", help="Enable human render")
    parser.add_argument(
        "-g", "--grid", type=int, choices=[4, 8, 16], default=4, help="Grid size for FrozenLake"
    )
    parser.add_argument(
        "-s", "--slip", action="store_true", default=False, help="Use slippery version of FrozenLake"
    )
    args = parser.parse_args()
    verbose = args.verbose

    # Select the map based on the grid size argument
    if args.grid == 4:
        selected_map = four_x_four_map
    elif args.grid == 8:
        selected_map = eight_x_eight_map
    else:
        selected_map = sixteen_x_sixteen_map

    # Set up the environment
    if args.human:
        env = gym.make("FrozenLake-v1", desc=selected_map, is_slippery=args.slip, render_mode="human")
    else:
        env = gym.make("FrozenLake-v1", desc=selected_map, is_slippery=args.slip)

    # Initialize agent
    agent = selected_agent(args)

    # Train the agent
    train_agent(env, agent, args.episodes)

    # Test the agent
    if canTest(args):
        test_agent(env, agent)
